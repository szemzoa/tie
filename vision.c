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

#define DEBUG_SKIP_VIT_LAYERS 0

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

//clip_vision_meta_t clip_vision_meta;

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

	const int width = dib.biWidth;
	const int height = dib.biHeight;
	const int bpp = dib.biBitCount;
	if (bpp != 24 && bpp != 32) {
		fclose(f);
		return -1;
	}

	const int target = meta->image_size;

	if (width != target || height != target)
		printf("Warning: Image is not target size. Using nearest-neighbor scaling.\n");

	const size_t row_padded = ((width * (bpp / 8) + 3) & ~3);
	unsigned char *rowbuf = malloc(row_padded);
	if (!rowbuf) {
		fclose(f);
		return -1;
	}

	float *pixels = ctx->vision_mem.image_raw.data;

	for (int y = 0; y < target; y++) {
		int sample_y = (int)((float)(height - 1) * y / (target - 1));

		// Invert the source Y-coordinate to account for BMP's bottom-to-top storage.
		int src_y = height - 1 - sample_y;

		fseek(f, bmp.bfOffBits + (long)row_padded * src_y, SEEK_SET);
		fread(rowbuf, 1, row_padded, f);

		for (int x = 0; x < target; x++) {
			int src_x = (int)((float)(width - 1) * x / (target - 1));
			const unsigned char *px = rowbuf + src_x * (bpp / 8);

			float r = px[2] / 255.0f;
			float g = px[1] / 255.0f;
			float b = px[0] / 255.0f;

			r = (r - meta->image_mean[0]) / meta->image_std[0];
			g = (g - meta->image_mean[1]) / meta->image_std[1];
			b = (b - meta->image_mean[2]) / meta->image_std[2];

			int idx = (y * target + x);
			pixels[idx] = r;
			pixels[target * target + idx] = g;
			pixels[2 * target * target + idx] = b;
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

static void vision_create_embeddings(struct TIEContext *ctx)
{
	VisionModel *vm = ctx->model_vision;
	MemLayoutVision *mem = &ctx->vision_mem;

	const int image_size = vm->image_size; // 896
	const int patch_size = vm->patch_size; // 14
	const int embed_dim = vm->embed_dim;   // 1152

	const int num_patches_side = image_size / patch_size;	     // 64 (H_out, W_out)
	const int num_patches = num_patches_side * num_patches_side; // 4096 (seq_len)

	// Allocate a temporary buffer for the Conv2D output
	// The raw conv output is [C_out, H_out, W_out] = [1152, 64, 64].
	MemType conv_output_mem;
	alloc_memtype(&conv_output_mem, GGML_TYPE_F32, (size_t)embed_dim * num_patches);
	if (!conv_output_mem.data) {
		fprintf(stderr, "Failed to allocate memory for conv output\n");
		return;
	}

	//	printf("Running Conv2D patch embedding...\n");
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

	free_memtype(&conv_output_mem);
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

static void vision_attention(struct TIEContext *ctx)
{
	VisionModel *vm = ctx->model_vision;
	MemLayoutVision *mem = &ctx->vision_mem;

	const int seq_len = (vm->image_size / vm->patch_size) * (vm->image_size / vm->patch_size); // 4096
	const int embed_dim = vm->embed_dim;							   // 1152
	const int num_heads = vm->num_heads;							   // 16
	const int head_dim = embed_dim / num_heads;						   // 72
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

void dispatch_add_and_store(MemType *dest, const MemType *src1, const MemType *src2, int size)
{
	float *dest_data = (float *)dest->data;
	const float *src1_data = (const float *)src1->data;
	const float *src2_data = (const float *)src2->data;

	for (int i = 0; i < size; ++i) {
		dest_data[i] = src1_data[i] + src2_data[i];
	}
}

void vision_transformer_layer(struct TIEContext *ctx, int layer_idx)
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

	vision_attention(ctx);

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

MemType *process_image_vision(struct TIEContext *ctx)
{
	VisionModel *vm = ctx->model_vision;
	MemLayoutVision *mem = &ctx->vision_mem;

	printf("Processing image");
	fflush(stdout);

	// Project raw patches into `mem->patch_embeds`
	vision_create_embeddings(ctx);

	// Create the initial hidden state by adding bias and positional embeddings.
	const float *projected_patches = (const float *)mem->patch_embeds.data;
	const float *pos_embeds_data = (const float *)vm->position_embd.mem.data;
	float *hidden_state_data = (float *)mem->hidden_state.data;

	const int num_patches = (vm->image_size / vm->patch_size) * (vm->image_size / vm->patch_size);
	const int embed_dim = vm->embed_dim;
	//	const size_t total_elements = (size_t)num_patches * embed_dim;

	for (int i = 0; i < num_patches; ++i) {
		for (int j = 0; j < embed_dim; ++j) {
			int idx = i * embed_dim + j;
			hidden_state_data[idx] = projected_patches[idx] + pos_embeds_data[idx];
		}
	}

#if DEBUG_SKIP_VIT_LAYERS
	// Load final layer output directly
	printf("DEBUG: Skipping ViT layers 0-26. Loading result from file...\n");
	int load_status = load_tensor_from_file("tensor_dumps/Layer_26_-_Final_Output.bin",
						(float *)mem->hidden_state.data, (size_t)num_patches * embed_dim);
	if (load_status != 0) {
		fprintf(stderr, "FATAL: Could not load debug tensor. Exiting.\n");
		exit(EXIT_FAILURE);
	}

#else
	// Run the Vision Transformer layers.
	for (int i = 0; i < vm->num_layers; i++) {
		vision_transformer_layer(ctx, i);
		printf(".");
		fflush(stdout);
	}
#endif
	// Final downsampling
	vision_downsample_and_project(ctx);

	printf("done\n");
	fflush(stdout);

	return &mem->projected_embeddings;
}
