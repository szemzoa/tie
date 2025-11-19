#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include "main.h"
#include "engine.h"
#include "model.h"
#include "math_dispatch.h"
#include "math_scalar.h"
#include "vision.h"


void dispatch_memcpy(MemType *dest, MemType *src, size_t src_element_offset, size_t element_count);

typedef struct __attribute__((packed)) {
	uint16_t bfType;
	uint32_t bfSize;
	uint16_t bfReserved1;
	uint16_t bfReserved2;
	uint32_t bfOffBits;
} BMPHeader;

typedef struct __attribute__((packed)) {
	uint32_t biSize;
	int32_t biWidth;
	int32_t biHeight;
	uint16_t biPlanes;
	uint16_t biBitCount;
	uint32_t biCompression;
	uint32_t biSizeImage;
	int32_t biXPelsPerMeter;
	int32_t biYPelsPerMeter;
	uint32_t biClrUsed;
	uint32_t biClrImportant;
} DIBHeader;


int load_bmp_clip(const char *filename, struct TIEContext *ctx)
{
	ClipVisionMeta *meta = &ctx->model_vision->meta;

	FILE *f = fopen(filename, "rb");
	if (!f) {
		fprintf(stderr, "Cannot open %s\n", filename);
		return -1;
	}

	BMPHeader bmp;
	DIBHeader dib;
	fread(&bmp, sizeof(bmp), 1, f);
	fread(&dib, sizeof(dib), 1, f);

	if (bmp.bfType != 0x4D42 || dib.biCompression != 0) {
		fclose(f);
		return -1;
	}

	int width = dib.biWidth;
	int height = dib.biHeight;

	// --- DYNAMIC RESOLUTION LOGIC ---
	int patch_size = ctx->model_vision->patch_size;

	// Optionally clamp to max size if needed (e.g. downscale huge images)
	// For now, just ensure multiple of patch_size
	int target_w = (width / patch_size) * patch_size;
	int target_h = (height / patch_size) * patch_size;

	if (target_w == 0)
		target_w = patch_size;
	if (target_h == 0)
		target_h = patch_size;

	ctx->vision_mem.image_raw_width = target_w;
	ctx->vision_mem.image_raw_height = target_h;

	const int bpp = dib.biBitCount;
	const size_t row_padded = ((width * (bpp / 8) + 3) & ~3);
	unsigned char *rowbuf = malloc(row_padded);
	if (!rowbuf) {
		fclose(f);
		return -1;
	}

	// Reallocate image_raw buffer if necessary (or rely on max allocation)
	// Assuming image_raw is large enough (768x768 float32) to hold this dynamic image.
	float *pixels = ctx->vision_mem.image_raw.data;

	// Resize/Crop loop
	for (int y = 0; y < target_h; y++) {
		// Nearest Neighbor Sampling
		int sample_y = (int)((float)(height - 1) * y / (target_h - 1));
		int src_y = height - 1 - sample_y; // BMP inversion

		fseek(f, bmp.bfOffBits + (long)row_padded * src_y, SEEK_SET);
		fread(rowbuf, 1, row_padded, f);

		for (int x = 0; x < target_w; x++) {
			int src_x = (int)((float)(width - 1) * x / (target_w - 1));
			const unsigned char *px = rowbuf + src_x * (bpp / 8);

			float r = px[2] / 255.0f;
			float g = px[1] / 255.0f;
			float b = px[0] / 255.0f;

			r = (r - meta->image_mean[0]) / meta->image_std[0];
			g = (g - meta->image_mean[1]) / meta->image_std[1];
			b = (b - meta->image_mean[2]) / meta->image_std[2];

			int idx = (y * target_w + x);
			int plane_stride = target_w * target_h;

			pixels[idx] = r;
			pixels[plane_stride + idx] = g;
			pixels[2 * plane_stride + idx] = b;
		}
	}

	free(rowbuf);
	fclose(f);
	return 0;
}

void dispatch_add_bias(MemType *matrix, const Tensor *bias, int rows, int cols)
{
	float *matrix_data = (float *)matrix->data;
	const float *bias_data = (const float *)bias->mem.data;

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			matrix_data[i * cols + j] += bias_data[j];
		}
	}
}

void dispatch_add_and_store(MemType *dest, const MemType *src1, const MemType *src2, int size)
{
	float *dest_data = (float *)dest->data;
	const float *src1_data = (const float *)src1->data;
	const float *src2_data = (const float *)src2->data;

	for (int i = 0; i < size; ++i) {
		dest_data[i] = src1_data[i] + src2_data[i];
	}
}

void dispatch_layer_norm(MemType *dest, const MemType *src, const Tensor *weight, const Tensor *bias, int size,
			 float eps)
{
	const float *src_data = (const float *)src->data;
	float *dest_data = (float *)dest->data;

	float sum = 0.0;
	for (int i = 0; i < size; ++i) {
		sum += src_data[i];
	}
	const float mean = sum / size;

	float sum_sq_diff = 0.0;
	for (int i = 0; i < size; ++i) {
		float diff = src_data[i] - mean;
		sum_sq_diff = fmaf(diff, diff, sum_sq_diff);
	}
	const float variance = sum_sq_diff / size;
	const float inv_std = 1.0f / sqrtf(variance + eps);

	const float *weight_data = (const float *)weight->mem.data;
	const float *bias_data = (const float *)bias->mem.data;

	for (int i = 0; i < size; ++i) {
		float normalized_val = (src_data[i] - mean) * inv_std;
		dest_data[i] = fmaf(normalized_val, weight_data[i], bias_data[i]);
	}
}

void dispatch_transpose(MemType *dest, const MemType *src, int rows, int cols)
{
	// Don't transpose between different types
	if (src->type != dest->type) {
		fprintf(stderr, "Error: Mismatched types in dispatch_transpose (%d != %d)\n", src->type, dest->type);
		return;
	}

	switch (src->type) {
	case GGML_TYPE_F32: {
		const float *src_data = (const float *)src->data;
		float *dest_data = (float *)dest->data;

		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				dest_data[j * rows + i] = src_data[i * cols + j];
			}
		}
		break;
	}

	case GGML_TYPE_F16:
	case GGML_TYPE_BF16: {
		const uint16_t *src_data = (const uint16_t *)src->data;
		uint16_t *dest_data = (uint16_t *)dest->data;
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				dest_data[j * rows + i] = src_data[i * cols + j];
			}
		}
		break;
	}

	default: {
		fprintf(stderr, "Error: Unsupported tensor type %d in dispatch_transpose.\n", src->type);
		return;
	}
	}
}

void dispatch_memcpy(MemType *dest, MemType *src, size_t src_element_offset, size_t element_count)
{
	int size_multiplier;

	if (src->type != dest->type) {
		fprintf(stderr, "Error: Mismatched types in dispatch_memcpy (%d != %d)\n", src->type, dest->type);
		return;
	}

	// Get size of one element
	switch (src->type) {
	case GGML_TYPE_F32: {
		size_multiplier = sizeof(float);
		break;
	}
	case GGML_TYPE_F16:
	case GGML_TYPE_BF16: {
		size_multiplier = sizeof(uint16_t);
		break;
	}
	default: {
		fprintf(stderr, "Error: Unsupported MemType %d in dispatch_memcpy\n", src->type);
		return;
	}
	}

	// Calculate all offsets and sizes in BYTES
	const size_t src_byte_offset = src_element_offset * size_multiplier;
	const size_t total_bytes_to_copy = element_count * size_multiplier;

	// Perform the copy
	memcpy((void *)dest->data, (void *)src->data + src_byte_offset, total_bytes_to_copy);
}

static void vision_attention(struct TIEContext *ctx, int seq_len)
{
	VisionModel *vm = ctx->model_vision;
	MemLayoutVision *mem = &ctx->vision_mem;

	const int embed_dim = vm->embed_dim;	    // 1152
	const int num_heads = vm->num_heads;	    // 16
	const int head_dim = embed_dim / num_heads; // 72
	const float attn_scale = 1.0f / sqrtf((float)head_dim);

	const float *Q_data = (const float *)mem->Q.data;
	const float *K_data = (const float *)mem->K.data;
	const float *V_data = (const float *)mem->V.data;
	float *attn_output_data = (float *)mem->attn_output.data;

	// Allocate buffers for ONE head
	MemType q_head, k_head, v_head, v_head_t, q_head_t, scores, output_head;

	alloc_memtype(&q_head, GGML_TYPE_F32, seq_len * head_dim);
	alloc_memtype(&k_head, GGML_TYPE_F32, seq_len * head_dim);
	alloc_memtype(&v_head, GGML_TYPE_F32, seq_len * head_dim);
	alloc_memtype(&q_head_t, GGML_TYPE_F32, head_dim * seq_len);
	alloc_memtype(&scores, GGML_TYPE_F32, seq_len * seq_len);
	alloc_memtype(&output_head, GGML_TYPE_F32, seq_len * head_dim);
	alloc_memtype(&v_head_t, GGML_TYPE_F32, head_dim * seq_len);

	float *q_head_data = (float *)q_head.data;
	float *k_head_data = (float *)k_head.data;
	float *v_head_data = (float *)v_head.data;
	float *scores_data = (float *)scores.data;
	float *output_head_data = (float *)output_head.data;

	memset(attn_output_data, 0, seq_len * embed_dim * sizeof(float));

	for (int h = 0; h < num_heads; ++h) {
		// Extract Q, K, V for the current head
		for (int i = 0; i < seq_len; ++i) {
			for (int j = 0; j < head_dim; ++j) {
				int src_idx = i * embed_dim + h * head_dim + j;
				int dst_idx = i * head_dim + j;
				q_head_data[dst_idx] = Q_data[src_idx];
				k_head_data[dst_idx] = K_data[src_idx];
				v_head_data[dst_idx] = V_data[src_idx];
			}
		}

		// Transpose Q: q_head_t = Q^T
		// Input q_head is [seq_len, head_dim]. Output q_head_t is [head_dim, seq_len]
		dispatch_transpose(&q_head_t, &q_head, seq_len, head_dim);

		// Calculate scores = Q @ K^T
		Tensor k_head_tensor = {.mem = k_head};
		dispatch_mat_mat(ctx, &q_head, &k_head_tensor, &scores, seq_len, head_dim, seq_len, 0);

		// Apply scaling and softmax
		for (int i = 0; i < seq_len * seq_len; ++i)
			scores_data[i] *= attn_scale;

		for (int i = 0; i < seq_len; ++i)
			softmax(scores_data + i * seq_len, seq_len);

		// Transpose V: v_head_t = V^T
		dispatch_transpose(&v_head_t, &v_head, seq_len, head_dim);

		// Calculate output_head = scores @ V
		Tensor v_head_t_tensor = {.mem = v_head_t};
		dispatch_mat_mat(ctx, &scores, &v_head_t_tensor, &output_head, seq_len, seq_len, head_dim, 0);

		// Scatter result back
		for (int i = 0; i < seq_len; ++i) {
			for (int j = 0; j < head_dim; ++j) {
				int src_idx = i * head_dim + j;
				int dst_idx = i * embed_dim + h * head_dim + j;
				attn_output_data[dst_idx] = output_head_data[src_idx];
			}
		}
	}

	free_memtype(&q_head);
	free_memtype(&k_head);
	free_memtype(&v_head);
	free_memtype(&q_head_t);
	free_memtype(&v_head_t);
	free_memtype(&scores);
	free_memtype(&output_head);
}


/* =================== Qwen3-VL =================== */

// Dynamically interpolates and shuffles positional embeddings
void vision_build_pos_embeds_dynamic(struct TIEContext *ctx, MemType *dest_buffer, int h_patches, int w_patches)
{
	VisionModel *vm = ctx->model_vision;
	float *out_data = (float *)dest_buffer->data;

	// Get the raw pre-trained grid dimensions
	int num_grid_per_side = vm->image_size / vm->patch_size; // 768 / 16 = 48

	// Setup for Permutation (Spatial Merge)
	int merge_size = vm->spatial_merge_size;
	//	int merged_h = h_patches / merge_size;
	int merged_w = w_patches / merge_size;

	// Iterate over the TARGET grid (h_patches x w_patches)
	// We generate the embedding for (h, w) and place it in the correct shuffled slot.
	for (int h = 0; h < h_patches; h++) {
		for (int w = 0; w < w_patches; w++) {

			// Interpolation Logic (linspace)
			// Calculate the source coordinates in the grid
			// Safety check for 1xN or Nx1 grids
			float r_h =
				(h_patches > 1) ? (float)h * (num_grid_per_side - 1) / (float)(h_patches - 1) : 0.0f;
			float r_w =
				(w_patches > 1) ? (float)w * (num_grid_per_side - 1) / (float)(w_patches - 1) : 0.0f;

			int h0 = (int)floorf(r_h);
			int h1 = (int)ceilf(r_h);
			int w0 = (int)floorf(r_w);
			int w1 = (int)ceilf(r_w);

			// Clamp to be safe
			if (h1 >= num_grid_per_side)
				h1 = num_grid_per_side - 1;
			if (w1 >= num_grid_per_side)
				w1 = num_grid_per_side - 1;

			float dh = r_h - h0;
			float dw = r_w - w0;

			// Calculate Shuffle Index
			// Map linear (h, w) to the 2x2 blocked layout
			int h_big = h / merge_size;
			int w_big = w / merge_size;
			int h_sub = h % merge_size;
			int w_sub = w % merge_size;

			// [merged_h][merged_w][2][2]
			int dest_idx = (h_big * merged_w * merge_size * merge_size) + (w_big * merge_size * merge_size)
				       + (h_sub * merge_size) + (w_sub);

			float *dest_vec = out_data + dest_idx * vm->embed_dim;

			// Bilinear Interpolation & Write
			// Get pointers to the 4 corner vectors in the pre-trained table
			float *v00 =
				(float *)vm->position_embd.mem.data + (h0 * num_grid_per_side + w0) * vm->embed_dim;
			float *v01 =
				(float *)vm->position_embd.mem.data + (h0 * num_grid_per_side + w1) * vm->embed_dim;
			float *v10 =
				(float *)vm->position_embd.mem.data + (h1 * num_grid_per_side + w0) * vm->embed_dim;
			float *v11 =
				(float *)vm->position_embd.mem.data + (h1 * num_grid_per_side + w1) * vm->embed_dim;

			// w00 = (1-dh)(1-dw), w01 = (1-dh)(dw), etc.
			float weight00 = (1.0f - dh) * (1.0f - dw);
			float weight01 = (1.0f - dh) * dw;
			float weight10 = dh * (1.0f - dw);
			float weight11 = dh * dw;

			for (int d = 0; d < vm->embed_dim; d++) {
				dest_vec[d] =
					v00[d] * weight00 + v01[d] * weight01 + v10[d] * weight10 + v11[d] * weight11;
			}
		}
	}
}

void vision_rope_cache_build_dynamic(struct TIEContext *ctx, int h_patches, int w_patches)
{
	VisionModel *vm = ctx->model_vision;
	MRopeCacheType *cache = &vm->mrope_cache;

	int head_dim = vm->embed_dim / vm->num_heads; // 64
	int half_dim = head_dim / 2;		      // 32
	int mrope_dim = head_dim / 4;		      // 16 (This was missing in dynamic version)

	int seq_len = h_patches * w_patches;

	// 1. Calculate standard 1D inv_freq table (Size 16, not 32)
	float inv_freq[mrope_dim];
	for (int i = 0; i < mrope_dim; i++) {
		// Use half_dim (32) as denominator, same as static version
		inv_freq[i] = 1.0f / powf(10000.0f, (float)(2 * i) / (float)half_dim);
	}

	// 2. Iterate patches in the SHUFFLED order
	int merge_size = vm->spatial_merge_size;
	int merged_w = w_patches / merge_size;

	for (int i = 0; i < seq_len; i++) {
		// ... (Decode logic is correct, keep it) ...
		int tokens_per_block = merge_size * merge_size;
		int block_idx = i / tokens_per_block;
		int intra_idx = i % tokens_per_block;

		int w_block = block_idx % merged_w;
		int h_block = block_idx / merged_w;
		int w_sub = intra_idx % merge_size;
		int h_sub = intra_idx / merge_size;

		int h_coord = h_block * merge_size + h_sub;
		int w_coord = w_block * merge_size + w_sub;

		// 3. Build Frequencies [H, W, H, W]
		float *cos_vec = cache->cos_table + i * head_dim;
		float *sin_vec = cache->sin_table + i * head_dim;

		for (int d = 0; d < mrope_dim; d++) {
			// Calculate freq for this dimension
			float freq_h = (float)h_coord * inv_freq[d];
			float freq_w = (float)w_coord * inv_freq[d];

			float ch = cosf(freq_h);
			float sh = sinf(freq_h);
			float cw = cosf(freq_w);
			float sw = sinf(freq_w);

			// Layout: [H (0-15), W (16-31), H (32-47), W (48-63)]

			// First Half (0-31)
			cos_vec[d] = ch; // H at 0..15
			sin_vec[d] = sh;
			cos_vec[d + mrope_dim] = cw; // W at 16..31
			sin_vec[d + mrope_dim] = sw;

			// Second Half (Duplicate for rotate_half logic)
			cos_vec[d + half_dim] = ch; // H at 32..47
			sin_vec[d + half_dim] = sh;
			cos_vec[d + half_dim + mrope_dim] = cw; // W at 48..63
			sin_vec[d + half_dim + mrope_dim] = sw;
		}
	}
}

void vision_create_embeddings_qwen3vl(struct TIEContext *ctx, int h_patches, int w_patches)
{
	VisionModel *vm = ctx->model_vision;
	MemLayoutVision *mem = &ctx->vision_mem;

	// Calculate Dynamic Dimensions
	const int patch_size = vm->patch_size; // 16
	const int embed_dim = vm->embed_dim;   // 2304
	const int num_patches = h_patches * w_patches;

	// Image dimensions derived from patches
	const int img_h = h_patches * patch_size;
	const int img_w = w_patches * patch_size;

	// Allocate Temps (Dynamic Size)
	MemType conv_output_mem_t0, conv_output_mem_t1, pos_embeds_mem, permuted_patched_mem;
	alloc_memtype(&conv_output_mem_t0, GGML_TYPE_F32, (size_t)embed_dim * num_patches);
	alloc_memtype(&conv_output_mem_t1, GGML_TYPE_F32, (size_t)embed_dim * num_patches);
	alloc_memtype(&permuted_patched_mem, GGML_TYPE_F32, (size_t)embed_dim * num_patches);

	// Allocate temp buffer for the pos_embedss
	alloc_memtype(&pos_embeds_mem, GGML_TYPE_F32, (size_t)embed_dim * num_patches);

	// 3. Run Conv2D (Kernel T=0 & T=1) using Dynamic Sizes
	dispatch_conv_2d_scalar(&conv_output_mem_t0, &mem->image_raw, &vm->patch_embd, NULL, img_h, img_w, patch_size,
				0);
	dispatch_conv_2d_scalar(&conv_output_mem_t1, &mem->image_raw, &vm->patch_embd_1, NULL, img_h, img_w, patch_size,
				0);

	// Sum T=0, T=1
	dispatch_add_and_store(&conv_output_mem_t0, &conv_output_mem_t0, &conv_output_mem_t1, embed_dim * num_patches);
	free_memtype(&conv_output_mem_t1);

	// Dynamic Permutation Logic
	// The goal: Interleave 2x2 blocks.
	// Calculate strides dynamically
	const int merge_size = vm->spatial_merge_size; // 2
						       //	int merged_h = h_patches / merge_size;
	int merged_w = w_patches / merge_size;

	// Stride logic:
	// We move through the output sequence one "merged block" at a time.
	// A merged block contains (merge_size * merge_size) patches.
	// A "row" of merged blocks contains `merged_w` blocks.
	// So, skipping one "row" of blocks means skipping `merged_w * 4` patches.
	const int stride_w_sub = 1;
	const int stride_h_sub = merge_size;		    // 2
	const int stride_w_big = (merge_size * merge_size); // 4
	const int stride_h_big = merged_w * stride_w_big;   // e.g., 24 * 4 = 96

	float *permuted_data = (float *)permuted_patched_mem.data;
	const float *conv_data = (const float *)conv_output_mem_t0.data;

	for (int c = 0; c < embed_dim; ++c) {
		for (int h = 0; h < h_patches; ++h) {
			for (int w = 0; w < w_patches; ++w) {
				// Read linear from Conv2D [C, H, W]
				float val = conv_data[c * h_patches * w_patches + h * w_patches + w];

				int h_sub = h % merge_size;
				int w_sub = w % merge_size;
				int h_big = h / merge_size;
				int w_big = w / merge_size;

				int patch_idx = h_big * stride_h_big + w_big * stride_w_big + h_sub * stride_h_sub
						+ w_sub * stride_w_sub;

				permuted_data[patch_idx * embed_dim + c] = val;
			}
		}
	}
	free_memtype(&conv_output_mem_t0);

	// Add Patch Bias
	dispatch_add_bias(&permuted_patched_mem, &vm->patch_embd_bias, num_patches, embed_dim);

	// Generate the interpolated grid for (h_patches, w_patches)
	vision_build_pos_embeds_dynamic(ctx, &pos_embeds_mem, h_patches, w_patches);

	// Add them together
	dispatch_add_and_store(&mem->hidden_state, &permuted_patched_mem, &pos_embeds_mem, num_patches * embed_dim);

	free_memtype(&permuted_patched_mem);
	free_memtype(&pos_embeds_mem);
}

/*
 *  - Pointer to the START of the current Q head [head_dim = 64]
 *  - Pointer to the START of the current K head [head_dim = 64]
 *  - The current token/patch index (0..2303)
 *  - 64
 */
void apply_mrope_cache(float *q_head_ptr, float *k_head_ptr, const MRopeCacheType *cache, int patch_idx, int head_dim)
{
	const int half_dim = head_dim / 2; // 32

	// Get pointers to the pre-calculated cos/sin values for this patch
	const float *cos_ptr = cache->cos_table + (patch_idx * head_dim);
	const float *sin_ptr = cache->sin_table + (patch_idx * head_dim);

	// This loop `(x * cos) + (rotate_half(x) * sin)`
	for (int d = 0; d < half_dim; ++d) {

		float q0 = q_head_ptr[d];
		float q1 = q_head_ptr[d + half_dim];
		float k0 = k_head_ptr[d];
		float k1 = k_head_ptr[d + half_dim];

		float cos0 = cos_ptr[d];
		float cos1 = cos_ptr[d + half_dim];
		float sin0 = sin_ptr[d];
		float sin1 = sin_ptr[d + half_dim];

		// q[d] = (q0 * cos0) + ((-q1) * sin0)
		q_head_ptr[d] = fmaf(-q1, sin0, q0 * cos0);

		// q[d + half_dim] = (q1 * cos1) + (q0 * sin1)
		q_head_ptr[d + half_dim] = fmaf(q0, sin1, q1 * cos1);

		// k[d] = (k0 * cos0) + ((-k1) * sin0)
		k_head_ptr[d] = fmaf(-k1, sin0, k0 * cos0);

		// k[d + half_dim] = (k1 * cos1) + (k0 * sin1)
		k_head_ptr[d + half_dim] = fmaf(k0, sin1, k1 * cos1);
	}
}

bool is_deepstack_layer(struct TIEContext *ctx, int layer_idx)
{
	VisionModel *vm = ctx->model_vision;
	bool *isdl = (bool *)vm->is_deepstack_layers.data;

	if (vm->num_deepstack_layers == 0)
		return false;

	return isdl[layer_idx];
}

// Output: [576, 2048], Input: [2304, 1024], layer_idx,  DeepStack=true, Final=false
void vision_run_patch_merger(struct TIEContext *ctx, MemType *dest_feature, MemType *src_hidden_state, int layer_idx,
			     int seq_len, bool use_postshuffle_norm)
{
	VisionModel *vm = ctx->model_vision;
	MemLayoutVision *mem = &ctx->vision_mem;

	// Get the weights for this specific layer
	VisionLayerWeights *weights = &vm->layers[layer_idx];

	// Get Dimensions
	//	const int seq_len = (vm->image_size / vm->patch_size) * (vm->image_size / vm->patch_size); // 2304
	const int embed_dim = vm->embed_dim;					  // 1024
	const int proj_dim = vm->projection_dim;				  // 2048
	const int merge_factor = vm->spatial_merge_size * vm->spatial_merge_size; // 4
	const int merged_seq_len = seq_len / merge_factor;			  // 576
	const int merged_dim = embed_dim * merge_factor;			  // 4096

	// This is the buffer we'll use as input for FC1
	MemType *norm_output_buffer;

	// LayerNorm
	if (use_postshuffle_norm) {
		// DEEPSTACK PATH (post-shuffle norm)
		// Norm is applied to the [576, 4096] reshaped tensor
		// The output to mem->merger_norm_buf
		for (int i = 0; i < merged_seq_len; i++) {
			MemType src_slice = mem_slice(src_hidden_state, i * merged_dim);
			MemType dest_slice = mem_slice(&mem->merger_norm_buf, i * merged_dim);

			dispatch_layer_norm(&dest_slice, &src_slice, &weights->ds_norm_weight, &weights->ds_norm_bias,
					    merged_dim, vm->norm_eps);
		}

		norm_output_buffer = &mem->merger_norm_buf;

		// FC1 (Linear + Bias)
		// Output: mem->merger_fc1_buf [576, 4096]
		dispatch_mat_mat(ctx, norm_output_buffer, &weights->ds_fc1_weight, &mem->merger_fc1_buf, merged_seq_len,
				 merged_dim, merged_dim, true);

		dispatch_add_bias(&mem->merger_fc1_buf, &weights->ds_fc1_bias, merged_seq_len, merged_dim);

		// Activation (GELU)
		dispatch_gelu_inplace(&mem->merger_fc1_buf, merged_seq_len * merged_dim);

		// FC2 (Linear + Bias)
		// Input: mem->merger_fc1_buf [576, 4096]
		// Output: dest_feature [576, 2048]
		dispatch_mat_mat(ctx, &mem->merger_fc1_buf, &weights->ds_fc2_weight, dest_feature, merged_seq_len,
				 merged_dim, proj_dim, true);

		dispatch_add_bias(dest_feature, &weights->ds_fc2_bias, merged_seq_len, proj_dim);

	} else {
		// FINAL MERGER PATH (pre-shuffle norm)
		// Norm is applied to the original [2304, 1024] tensor.
		for (int i = 0; i < seq_len; i++) {
			MemType src_slice = mem_slice(src_hidden_state, i * embed_dim);
			MemType dest_slice = mem_slice(&mem->merger_norm_buf, i * embed_dim);

			dispatch_layer_norm(&dest_slice, &src_slice, &vm->post_ln, &vm->post_ln_bias, embed_dim,
					    vm->norm_eps);
		}
		norm_output_buffer = &mem->merger_norm_buf;

		// FC1 (Linear + Bias)
		// Output: mem->merger_fc1_buf [576, 4096]
		dispatch_mat_mat(ctx, norm_output_buffer, &vm->mm_0_weight, &mem->merger_fc1_buf, merged_seq_len,
				 merged_dim, merged_dim, true);

		dispatch_add_bias(&mem->merger_fc1_buf, &vm->mm_0_bias, merged_seq_len, merged_dim);

		// Activation (GELU)
		dispatch_gelu_inplace(&mem->merger_fc1_buf, merged_seq_len * merged_dim);

		// 5. FC2 (Linear + Bias)
		// Input: mem->merger_fc1_buf [576, 4096]
		// Output: dest_feature [576, 2048]
		dispatch_mat_mat(ctx, &mem->merger_fc1_buf, &vm->mm_2_weight, dest_feature, merged_seq_len, merged_dim,
				 proj_dim, true);

		dispatch_add_bias(dest_feature, &vm->mm_2_bias, merged_seq_len, proj_dim);
	}
}

void vision_transformer_layer_qwen3vl(struct TIEContext *ctx, int layer_idx, int seq_len)
{
	VisionModel *vm = ctx->model_vision;
	MemLayoutVision *mem = &ctx->vision_mem;
	VisionLayerWeights *l = &vm->layers[layer_idx];

	const int embed_dim = vm->embed_dim;
	const int ffn_dim = vm->ffn_dim;
	const int total_size = seq_len * embed_dim;
	const int qkv_width = embed_dim * 3;


	// Pre-Attention LayerNorm & Residual
	memcpy(mem->residual_scratch.data, mem->hidden_state.data, seq_len * embed_dim * sizeof(float));

	for (int i = 0; i < seq_len; ++i) {
		MemType dest_slice = mem_slice(&mem->normed_input, i * embed_dim);
		MemType src_slice = mem_slice(&mem->hidden_state, i * embed_dim);
		dispatch_layer_norm(&dest_slice, &src_slice, &l->ln1, &l->ln1_bias, embed_dim, vm->norm_eps);
	}

	// Multi-Head Attention Fused weight
	dispatch_mat_mat(ctx, &mem->normed_input, &l->attn_qkv, &mem->QKV_fused, seq_len, embed_dim, qkv_width, true);
	dispatch_add_bias(&mem->QKV_fused, &l->attn_qkv_bias, seq_len, qkv_width);

	float *qkv_fused_ptr = (float *)mem->QKV_fused.data;
	float *q_ptr = (float *)mem->Q.data;
	float *k_ptr = (float *)mem->K.data;
	float *v_ptr = (float *)mem->V.data;

	const int num_elements_per_patch = embed_dim * 3; // Total size of one QKV block
	const int num_elements_per_qkv = embed_dim;	  // Size of Q, K, or V

	/* interleaved format */
	for (int i = 0; i < seq_len; ++i) {
		// Source pointers
		float *qkv_patch_src = qkv_fused_ptr + i * num_elements_per_patch;
		float *q_src = qkv_patch_src;
		float *k_src = qkv_patch_src + num_elements_per_qkv;
		float *v_src = qkv_patch_src + 2 * num_elements_per_qkv;

		// Destination pointers
		float *q_patch_dst = q_ptr + i * num_elements_per_qkv;
		float *k_patch_dst = k_ptr + i * num_elements_per_qkv;
		float *v_patch_dst = v_ptr + i * num_elements_per_qkv;

		// Copy the data
		memcpy(q_patch_dst, q_src, num_elements_per_qkv * sizeof(float));
		memcpy(k_patch_dst, k_src, num_elements_per_qkv * sizeof(float));
		memcpy(v_patch_dst, v_src, num_elements_per_qkv * sizeof(float));
	}

	// Apply M-RoPE
	const int num_heads = vm->num_heads;		// 16
	const int head_dim = vm->embed_dim / num_heads; // 64

	float *q_data = (float *)mem->Q.data;
	float *k_data = (float *)mem->K.data;

	// Loop over every patch (token) in the sequence
	for (int patch_idx = 0; patch_idx < seq_len; ++patch_idx) {

		// Loop over all heads
		for (int h_idx = 0; h_idx < num_heads; ++h_idx) {

			// Calculate the correct offset within the [seq_len, embed_dim] layout
			int head_offset = (patch_idx * embed_dim) + (h_idx * head_dim);

			float *q_head_ptr = q_data + head_offset;
			float *k_head_ptr = k_data + head_offset;

			// Apply the M-RoPE in-place
			apply_mrope_cache(q_head_ptr, k_head_ptr, &ctx->model_vision->mrope_cache, patch_idx, head_dim);
		}
	}

	// Multi-Head Attention (Standard)
	// It takes Q, K, V and produces attn_output.
	vision_attention(ctx, seq_len);

	// Attention Output Projection
	dispatch_mat_mat(ctx, &mem->attn_output, &l->attn_out, &mem->attn_proj_output, seq_len, embed_dim, embed_dim,
			 true);
	dispatch_add_bias(&mem->attn_proj_output, &l->attn_out_bias, seq_len, embed_dim);

	// First Residual Connection
	dispatch_add_and_store(&mem->hidden_state, &mem->residual_scratch, &mem->attn_proj_output, total_size);

	// Pre-FFN LayerNorm & Residual
	memcpy(mem->residual_scratch.data, mem->hidden_state.data, total_size * sizeof(float));

	for (int i = 0; i < seq_len; ++i) {
		MemType dest_slice = mem_slice(&mem->normed_input, i * embed_dim);
		MemType src_slice = mem_slice(&mem->hidden_state, i * embed_dim);
		dispatch_layer_norm(&dest_slice, &src_slice, &l->ln2, &l->ln2_bias, embed_dim, vm->norm_eps);
	}

	// Feed-Forward Network
	dispatch_mat_mat(ctx, &mem->normed_input, &l->ffn_up, &mem->ffn_up_output, seq_len, embed_dim, ffn_dim, true);
	dispatch_add_bias(&mem->ffn_up_output, &l->ffn_up_bias, seq_len, ffn_dim);

	// Activation
	dispatch_gelu_inplace(&mem->ffn_up_output, seq_len * ffn_dim);

	dispatch_mat_mat(ctx, &mem->ffn_up_output, &l->ffn_down, &mem->ffn_down_output, seq_len, ffn_dim, embed_dim,
			 true);
	dispatch_add_bias(&mem->ffn_down_output, &l->ffn_down_bias, seq_len, embed_dim);

	// Second Residual Connection
	dispatch_add_and_store(&mem->hidden_state, &mem->residual_scratch, &mem->ffn_down_output, total_size);
}

MemType *process_image_vision_qwen3vl(struct TIEContext *ctx)
{
	VisionModel *vm = ctx->model_vision;
	MemLayoutVision *mem = &ctx->vision_mem;
	int deepstack_merger_index = -1;
	int h_patches, w_patches;


	printf("Processing image QWEN3-VL\n");

	if (ctx->vision_mem.image_raw_width > vm->image_size || ctx->vision_mem.image_raw_height > vm->image_size) {
		printf("invalid image size, supports max %u x %u\n", vm->image_size, vm->image_size);
		return NULL;
	}

	if ((ctx->vision_mem.image_raw_width % vm->patch_size != 0)
	    || (ctx->vision_mem.image_raw_height % vm->patch_size != 0)) {
		printf("invalid image size, supports %upx grid only\n", vm->patch_size);
		return NULL;
	}

	// Project raw patches into `mem->patch_embeds`
	w_patches = ctx->vision_mem.image_raw_width / vm->patch_size;
	h_patches = ctx->vision_mem.image_raw_height / vm->patch_size;

	vision_create_embeddings_qwen3vl(ctx, h_patches, w_patches);

	// The loop logic (seq_len) must use (h_patches * w_patches)
	int seq_len = h_patches * w_patches;

	// Build Vision RoPE for THIS resolution
	vision_rope_cache_build_dynamic(ctx, h_patches, w_patches);

	// Run the Vision Transformer layers.
	for (int layer_idx = 0; layer_idx < vm->num_layers; layer_idx++) {

		vision_transformer_layer_qwen3vl(ctx, layer_idx, seq_len);

		// DEEPSTACK
		if (is_deepstack_layer(ctx, layer_idx) == true) {

			deepstack_merger_index++;
			// printf("%u layer is deepstack, merger_idx: %u\n", layer_idx, deepstack_merger_index);

			// Get the destination buffer
			MemType *dest_feature = &mem->deepstack_features[deepstack_merger_index];

			// Run the merger
			vision_run_patch_merger(ctx, dest_feature, &mem->hidden_state, layer_idx, seq_len, true);
		}

		printf(".");
		fflush(stdout);
	}

	// Final merger
	vision_run_patch_merger(ctx, &mem->projected_embeddings, &mem->hidden_state, 0, seq_len, false);

	printf("done\n");
	fflush(stdout);

	return &mem->projected_embeddings;
}

/* =================== Gemma-3 =================== */
void vision_create_embeddings_gemma3(struct TIEContext *ctx)
{
	VisionModel *vm = ctx->model_vision;
	MemLayoutVision *mem = &ctx->vision_mem;

	const int image_size = vm->image_size; // 896
	const int patch_size = vm->patch_size; // 14
	const int embed_dim = vm->embed_dim;   // 1152

	const int num_patches_side = image_size / patch_size;	     // 64 (H_out, W_out)
	const int num_patches = num_patches_side * num_patches_side; // 4096 (seq_len)

	//	printf("%s, image_size: %u, patch_size: %u, embed_dim: %u, num_patches_side: %u, num_patches: %u\n",
	//	       __FUNCTION__, image_size, patch_size, embed_dim, num_patches_side, num_patches);

	// Allocate a temporary buffer for the Conv2D output
	// The raw conv output is [C_out, H_out, W_out] = [1152, 64, 64].
	MemType conv_output_mem;
	alloc_memtype(&conv_output_mem, GGML_TYPE_F32, (size_t)embed_dim * num_patches);
	if (!conv_output_mem.data) {
		fprintf(stderr, "Failed to allocate memory for conv output\n");
		return;
	}

	//	printf("Running Conv2D patch embedding\n");
	dispatch_conv_2d_scalar(&conv_output_mem,     // Dest: [1152, 64, 64]
				&mem->image_raw,      // Src: [3, 896, 896]
				&vm->patch_embd,      // Kernel: [1152, 3, 14, 14]
				&vm->patch_embd_bias, // Bias: [1152]
				image_size,	      // H_in
				image_size,	      // W_in
				patch_size,	      // Stride (14)
				0);		      // Padding (0)

	// Permute and flatten the result
	// The output of conv is [C_out, H_out, W_out] = [1152, 64, 64] (planar)
	// The transformer needs [seq_len, C_out] = [4096, 1152]
	// We must permute [1152, 64, 64] -> [64, 64, 1152] and then flatten.

	//	printf("Permuting Conv2D output to [seq_len, embed_dim]...\n");
	float *dest_data = (float *)mem->patch_embeds.data;			// [4096, 1152]
	const float *conv_data = (const float *)conv_output_mem.data;		// [1152, 64, 64]
	const size_t H_out_W_out = (size_t)num_patches_side * num_patches_side; // 4096

	for (int c = 0; c < embed_dim; ++c) {
		const float *conv_plane_ptr = conv_data + c * H_out_W_out;
		for (int i = 0; i < num_patches; ++i) { // i = y*W_out + x
			dest_data[i * embed_dim + c] = conv_plane_ptr[i];
		}
	}

	// Create the initial hidden state by adding bias and positional embeddings.
	const float *projected_patches = (const float *)mem->patch_embeds.data;
	const float *pos_embeds_data = (const float *)vm->position_embd.mem.data;
	float *hidden_state_data = (float *)mem->hidden_state.data;

	for (int i = 0; i < num_patches; ++i) {
		for (int j = 0; j < embed_dim; ++j) {
			int idx = i * embed_dim + j;
			hidden_state_data[idx] = projected_patches[idx] + pos_embeds_data[idx];
		}
	}

	free_memtype(&conv_output_mem);
}

static void vision_downsample_and_project(struct TIEContext *ctx)
{
	VisionModel *vm = ctx->model_vision;
	MemLayoutVision *mem = &ctx->vision_mem;

	const int image_size = vm->image_size;
	const int patch_size = vm->patch_size;
	const int embed_dim = vm->embed_dim;
	const int proj_dim = vm->projection_dim;
	const int scale_factor = vm->proj_scale_factor;

	const int num_patches_side = image_size / patch_size;
	const int num_patches = num_patches_side * num_patches_side;

	const int pooled_patches_side = num_patches_side / scale_factor;
	const int pooled_seq_len = pooled_patches_side * pooled_patches_side;

	// Apply the final Post-Transformer LayerNorm
	// The ViT has one last normalization the final layer
	for (int i = 0; i < num_patches; ++i) {
		MemType slice = mem_slice(&mem->hidden_state, i * embed_dim);
		dispatch_layer_norm(&slice, &slice, &vm->post_ln, &vm->post_ln_bias, embed_dim, vm->norm_eps);
	}

	// Perform 2D Average Pooling
	// This downsamples the 64x64 grid of embeddings to a 16x16 grid.
	float *pooled_data = (float *)mem->pooled_embeddings.data;
	const float *hidden_state_data = (const float *)mem->hidden_state.data;

	for (int y_out = 0; y_out < pooled_patches_side; ++y_out) {
		for (int x_out = 0; x_out < pooled_patches_side; ++x_out) {
			float *dest_vec = pooled_data + (y_out * pooled_patches_side + x_out) * embed_dim;
			memset(dest_vec, 0, embed_dim * sizeof(float));

			for (int y_in = 0; y_in < scale_factor; ++y_in) {
				for (int x_in = 0; x_in < scale_factor; ++x_in) {
					int src_y = y_out * scale_factor + y_in;
					int src_x = x_out * scale_factor + x_in;
					const float *src_vec =
						hidden_state_data + (src_y * num_patches_side + src_x) * embed_dim;
					for (int d = 0; d < embed_dim; ++d) {
						dest_vec[d] += src_vec[d];
					}
				}
			}
			float inv_pool_size = 1.0f / (float)(scale_factor * scale_factor);
			for (int d = 0; d < embed_dim; ++d) {
				dest_vec[d] *= inv_pool_size;
			}
		}
	}

	// Apply the Projector's RMSNorm
	// Loop over each token in the pooled sequence (pooled_seq_len = 256)
	for (int i = 0; i < pooled_seq_len; ++i) {
		// Get a slice for the i-th token [1, 1152]
		MemType slice = mem_slice(&mem->pooled_embeddings, i * embed_dim);

		dispatch_rms_norm(&slice, &vm->soft_embd_norm, &slice, embed_dim, vm->norm_eps);
	}

	// Final Projection
	// This projects the [256, 1152] pooled embeddings to the final [256, 2560] tensor.
	dispatch_mat_mat(ctx, &mem->pooled_embeddings, &vm->input_projection, &mem->projected_embeddings,
			 pooled_seq_len, embed_dim, proj_dim, true);
}

void vision_transformer_layer_gemma3(struct TIEContext *ctx, int layer_idx)
{
	VisionModel *vm = ctx->model_vision;
	MemLayoutVision *mem = &ctx->vision_mem;
	VisionLayerWeights *l = &vm->layers[layer_idx];


	const int seq_len = (vm->image_size / vm->patch_size) * (vm->image_size / vm->patch_size);
	const int embed_dim = vm->embed_dim;
	const int ffn_dim = vm->ffn_dim;
	const int total_size = seq_len * embed_dim;

	// Pre-Attention LayerNorm & Residual
	memcpy(mem->residual_scratch.data, mem->hidden_state.data, seq_len * embed_dim * sizeof(float));

	for (int i = 0; i < seq_len; ++i) {
		MemType dest_slice = mem_slice(&mem->normed_input, i * embed_dim);
		MemType src_slice = mem_slice(&mem->hidden_state, i * embed_dim);
		dispatch_layer_norm(&dest_slice, &src_slice, &l->ln1, &l->ln1_bias, embed_dim, vm->norm_eps);
	}

	// Multi-Head Attention
	dispatch_mat_mat(ctx, &mem->normed_input, &l->attn_q, &mem->Q, seq_len, embed_dim, embed_dim, true);
	dispatch_add_bias(&mem->Q, &l->attn_q_bias, seq_len, embed_dim);

	dispatch_mat_mat(ctx, &mem->normed_input, &l->attn_k, &mem->K, seq_len, embed_dim, embed_dim, true);
	dispatch_add_bias(&mem->K, &l->attn_k_bias, seq_len, embed_dim);

	dispatch_mat_mat(ctx, &mem->normed_input, &l->attn_v, &mem->V, seq_len, embed_dim, embed_dim, true);
	dispatch_add_bias(&mem->V, &l->attn_v_bias, seq_len, embed_dim);

	vision_attention(ctx, seq_len);

	dispatch_mat_mat(ctx, &mem->attn_output, &l->attn_out, &mem->attn_proj_output, seq_len, embed_dim, embed_dim,
			 true);
	dispatch_add_bias(&mem->attn_proj_output, &l->attn_out_bias, seq_len, embed_dim);

	// First Residual Connection
	dispatch_add_and_store(&mem->hidden_state, &mem->residual_scratch, &mem->attn_proj_output, total_size);

	// Pre-FFN LayerNorm & Residual
	// Save the result of the first residual connection before the FFN block.
	memcpy(mem->residual_scratch.data, mem->hidden_state.data, total_size * sizeof(float));

	for (int i = 0; i < seq_len; ++i) {
		MemType dest_slice = mem_slice(&mem->normed_input, i * embed_dim);
		MemType src_slice = mem_slice(&mem->hidden_state, i * embed_dim);
		dispatch_layer_norm(&dest_slice, &src_slice, &l->ln2, &l->ln2_bias, embed_dim, vm->norm_eps);
	}

	// Feed-Forward Network
	dispatch_mat_mat(ctx, &mem->normed_input, &l->ffn_up, &mem->ffn_up_output, seq_len, embed_dim, ffn_dim, true);
	dispatch_add_bias(&mem->ffn_up_output, &l->ffn_up_bias, seq_len, ffn_dim);

	// Activation
	dispatch_gelu_inplace(&mem->ffn_up_output, seq_len * ffn_dim);

	dispatch_mat_mat(ctx, &mem->ffn_up_output, &l->ffn_down, &mem->ffn_down_output, seq_len, ffn_dim, embed_dim,
			 true);
	dispatch_add_bias(&mem->ffn_down_output, &l->ffn_down_bias, seq_len, embed_dim);

	// Second Residual Connection
	dispatch_add_and_store(&mem->hidden_state, &mem->residual_scratch, &mem->ffn_down_output, total_size);
}

MemType *process_image_vision_gemma3(struct TIEContext *ctx)
{
	VisionModel *vm = ctx->model_vision;
	MemLayoutVision *mem = &ctx->vision_mem;

	printf("Processing image Gemma3");

	if (ctx->vision_mem.image_raw_width != vm->image_size || ctx->vision_mem.image_raw_height != vm->image_size) {
		printf("invalid image size, supports %u x %u only\n", vm->image_size, vm->image_size);
		return NULL;
	}

	// Project raw patches into `mem->patch_embeds`
	vision_create_embeddings_gemma3(ctx);

	// Run the Vision Transformer layers.
	for (int layer_idx = 0; layer_idx < vm->num_layers; layer_idx++) {

		vision_transformer_layer_gemma3(ctx, layer_idx);

		printf(".");
		fflush(stdout);
	}

	// Final downsampling
	vision_downsample_and_project(ctx);

	printf("done\n");
	fflush(stdout);

	return &mem->projected_embeddings;
}
