#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "math_scalar.h"
#include "engine.h"
#include "threadpool.h"


inline float bf16_to_fp32(uint16_t bf16)
{
	union {
		uint32_t u;
		float f;
	} converter;
	converter.u = ((uint32_t)bf16) << 16;
	return converter.f;
}

inline uint16_t fp32_to_bf16(float f)
{
	union {
		float f;
		uint32_t u;
	} u = {.f = f};
	return (uint16_t)(u.u >> 16);
}

inline uint16_t fp32_to_bf16_rne(float f)
{
	union {
		uint32_t u;
		float f;
	} v = {.f = f};

	uint32_t x = v.u;

	// add 0x7FFF + LSB of upper word for tie-to-even
	uint32_t lsb = (x >> 16) & 1;
	uint32_t rounding_bias = 0x7FFF + lsb;

	return (uint16_t)((x + rounding_bias) >> 16);
}

inline float fp16_to_fp32(uint16_t h)
{
	uint32_t sign = ((uint32_t)(h >> 15) & 1) << 31;
	int exponent = (h >> 10) & 0x1F;
	uint32_t mantissa = (h & 0x3FF) << 13;

	if (exponent == 0x1F) { // Inf/NaN
		exponent = 0xFF;
	} else if (exponent != 0) { // Normalized
		exponent += 112;    // 127 - 15
	} else if (mantissa != 0) { // Denormal
		exponent = 113;	    // 127 - 14
		while ((mantissa & 0x7F800000) == 0) {
			mantissa <<= 1;
			exponent--;
		}
		mantissa &= 0x007FFFFF;
	}

	uint32_t f32 = sign | ((uint32_t)exponent << 23) | mantissa;
	return *(float *)&f32;
}

inline uint16_t fp32_to_fp16(float f)
{
	uint32_t x;
	memcpy(&x, &f, sizeof(f)); // Safe bitcast

	uint16_t h = (x >> 16) & 0x8000; // Sign
	uint32_t e = (x >> 23) & 0xff;	 // Exponent
	uint32_t m = x & 0x7fffff;	 // Mantissa

	if (e < 103) { // Underflow
		return h;
	}
	if (e > 142) { // Overflow
		return h | 0x7c00;
	}

	e = e - 112;
	m = m >> 13;

	return h | (e << 10) | m;
}

inline void get_scale_min_k4(int j, const uint8_t *scales, uint8_t *scale, uint8_t *min)
{
	if (j < 4) {
		*scale = scales[j] & 0x3F;
		*min = scales[j + 4] & 0x3F;
	} else {
		*scale = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
		*min = (scales[j + 4] >> 4) | ((scales[j - 0] >> 6) << 4);
	}
}

void convert_f32_bf16_scalar(const void *src, void *dest, int size)
{
	float *S = (float *)src;
	uint16_t *D = (uint16_t *)dest;

	for (int i = 0; i < size; i++) {
		D[i] = fp32_to_bf16(S[i]);
	}
}

void convert_bf16_f32_scalar(const void *src, void *dest, int size)
{
	uint16_t *S = (uint16_t *)src;
	float *D = (float *)dest;

	for (int i = 0; i < size; i++) {
		D[i] = bf16_to_fp32(S[i]);
	}
}

void convert_bf16_bf16_scalar(const void *src, void *dest, int size)
{
	memcpy(dest, src, size * sizeof(uint16_t));
}

void convert_f32_f32_scalar(const void *src, void *dest, int size)
{
	memcpy(dest, src, size * sizeof(float));
}

void apply_residual_f32_f32_scalar(void *acc_void, const void *residual_void, int size)
{
	float *acc = (float *)acc_void;
	float *residual = (float *)residual_void;

	for (int i = 0; i < size; i++) {
		acc[i] = fmaf(residual[i], 1.0f, acc[i]);
	}
}

void apply_rope_cache_f32_scalar(RopeCacheType *rope_cache, void *X, int pos, int head_dim)
{
	float *x = (float *)X;
	int h_dim_half = head_dim / 2;

	if (pos >= rope_cache->max_pos) {
		fprintf(stderr, "Position %d exceeds rope cache max_pos %d\n", pos, rope_cache->max_pos);
		return;
	}

	const float *sin_vals = rope_cache->sin + pos * h_dim_half;
	const float *cos_vals = rope_cache->cos + pos * h_dim_half;

	int i = 0;
	for (; i <= h_dim_half - 2; i += 2) {
		float x_r0 = x[i], x_i0 = x[i + h_dim_half];
		float x_r1 = x[i + 1], x_i1 = x[i + 1 + h_dim_half];

		float sin0 = sin_vals[i], cos0 = cos_vals[i];
		float sin1 = sin_vals[i + 1], cos1 = cos_vals[i + 1];

		x[i] = fmaf(x_r0, cos0, -x_i0 * sin0);
		x[i + h_dim_half] = fmaf(x_r0, sin0, x_i0 * cos0);

		x[i + 1] = fmaf(x_r1, cos1, -x_i1 * sin1);
		x[i + 1 + h_dim_half] = fmaf(x_r1, sin1, x_i1 * cos1);
	}

	// Handle tail if odd head_dim
	for (; i < h_dim_half; ++i) {
		float x_real = x[i];
		float x_imag = x[i + h_dim_half];
		float sin = sin_vals[i], cos = cos_vals[i];

		x[i] = fmaf(x_real, cos, -x_imag * sin);
		x[i + h_dim_half] = fmaf(x_real, sin, x_imag * cos);
	}
}

void dequantize_row_q4_k_f32_scalar(const void *__restrict__ q_void, float *__restrict__ y, int k)
{
	const block_q4_k *blocks = (const block_q4_k *)q_void;
	const int nb = k / QK_K; // k is the total number of elements, e.g., model dimension

	// Iterate over the blocks that make up the row
	for (int i = 0; i < nb; ++i) {
		const block_q4_k *blk = &blocks[i];
		const float d = fp16_to_fp32(blk->d);
		const float dmin = fp16_to_fp32(blk->dmin);
		const uint8_t *q = blk->qs;
		int is = 0;

		float *y_block = y + i * QK_K;

		for (int j = 0; j < QK_K; j += 64) {
			uint8_t s1, m1, s2, m2;

			get_scale_min_k4(is + 0, blk->scales, &s1, &m1);
			const float d1 = d * s1;
			const float m1f = dmin * m1;

			get_scale_min_k4(is + 1, blk->scales, &s2, &m2);
			const float d2 = d * s2;
			const float m2f = dmin * m2;

			// Dequantize 32 low-nibble values
			for (int l = 0; l < 32; ++l) {
				y_block[j + l] = d1 * (float)(q[l] & 0x0F) - m1f;
			}

			// Dequantize 32 high-nibble values
			for (int l = 0; l < 32; ++l) {
				y_block[j + 32 + l] = d2 * (float)(q[l] >> 4) - m2f;
			}
			q += 32;
			is += 2;
		}
	}
}

void dequantize_row_q6_k_f32_scalar(const void *__restrict__ q_void, float *__restrict__ y, int k)
{
	const block_q6_k *x = (const block_q6_k *)q_void;
	const int64_t nb = k / QK_K;

	for (int i = 0; i < nb; i++) {
		const float d = fp16_to_fp32(x[i].d);

		const uint8_t *ql = x[i].ql;
		const uint8_t *qh = x[i].qh;
		const int8_t *sc = x[i].scales;

		for (int n = 0; n < QK_K; n += 128) {
			for (int l = 0; l < 32; ++l) {
				int is = l / 16;
				const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
				const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
				const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
				const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
				y[l + 0] = d * sc[is + 0] * q1;
				y[l + 32] = d * sc[is + 2] * q2;
				y[l + 64] = d * sc[is + 4] * q3;
				y[l + 96] = d * sc[is + 6] * q4;
			}
			y += 128;
			ql += 64;
			qh += 32;
			sc += 8;
		}
	}
}

// Implements: output = (x / sqrt(variance + eps)) * weight
void rms_norm_f32_f32_f32_scalar(void *__restrict O, const void *__restrict X, const Tensor *__restrict W, int size,
				 float eps)
{
	float *o = (float *)O;
	float *x = (float *)X;
	float *weight = W->mem.data;

	// Unrolled accumulation for faster reduction
	float ss0 = 0.0f, ss1 = 0.0f, ss2 = 0.0f, ss3 = 0.0f;
	int i = 0;
	for (; i <= size - 4; i += 4) {
		ss0 = fmaf(x[i + 0], x[i + 0], ss0);
		ss1 = fmaf(x[i + 1], x[i + 1], ss1);
		ss2 = fmaf(x[i + 2], x[i + 2], ss2);
		ss3 = fmaf(x[i + 3], x[i + 3], ss3);
	}
	float ss = ss0 + ss1 + ss2 + ss3;
	for (; i < size; i++) {
		ss = fmaf(x[i], x[i], ss);
	}

	float inv_rms = 1.0f / sqrtf(ss / size + eps);

	// Apply weight and scale
	for (int i = 0; i < size; ++i) {
		o[i] = x[i] * weight[i] * inv_rms;
	}
}

float dot_product_f32_f32_scalar(const void *__restrict a, const void *__restrict b, int size)
{
	float *A = (float *)a;
	float *B = (float *)b;

	float sum = 0.0f;
	for (int j = 0; j < size; j++)
		sum = fmaf(A[j], B[j], sum);

	return sum;
}

float dot_product_f32_bf16_scalar(const void *__restrict a, const void *__restrict b, int size)
{
	float *A = (float *)a;
	uint16_t *B = (uint16_t *)b;

	float sum = 0.0f;
	for (int j = 0; j < size; j++)
		sum = fmaf(A[j], bf16_to_fp32(B[j]), sum);

	return sum;
}

void accumulate_weighted_V_f32_bf16_scalar(void *O, float weight, const void *V, int size)
{
	float *out = (float *)O;
	uint16_t *v = (uint16_t *)V;

	for (int i = 0; i < size; i++) {
		out[i] = fmaf(weight, bf16_to_fp32(v[i]), out[i]);
	}
}

void accumulate_weighted_V_f32_f32_scalar(void *O, float weight, const void *V, int size)
{
	float *out = (float *)O;
	float *v = (float *)V;

	for (int i = 0; i < size; i++) {
		out[i] = fmaf(weight, v[i], out[i]);
	}
}

float dot_product_f32_q6k_scalar(const float *x, const block_q6_k *block)
{
	float sum = 0.0f;
	const float d = fp16_to_fp32(block->d);
	const uint8_t *ql = block->ql;
	const uint8_t *qh = block->qh;
	const int8_t *sc = block->scales;

	for (int n = 0; n < QK_K; n += 128) {
		for (int l = 0; l < 32; ++l) {
			int is = l / 16;
			int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
			int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
			int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
			int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

			sum += d * sc[is + 0] * q1 * x[l + 0];
			sum += d * sc[is + 2] * q2 * x[l + 32];
			sum += d * sc[is + 4] * q3 * x[l + 64];
			sum += d * sc[is + 6] * q4 * x[l + 96];
		}
		ql += 64;
		qh += 32;
		sc += 8;
		x += 128;
	}

	return sum;
}

float dot_product_f32_q4k_scalar(const float *x, const block_q4_k *blk)
{
	const float d = fp16_to_fp32(blk->d);
	const float dmin = fp16_to_fp32(blk->dmin);

	const uint8_t *q = blk->qs;
	float sum = 0.0f;
	int is = 0;

	for (int j = 0; j < 256; j += 64) {
		uint8_t s1, m1, s2, m2;

		get_scale_min_k4(is + 0, blk->scales, &s1, &m1);
		float d1 = d * (float)s1;
		float m1f = dmin * (float)m1;

		get_scale_min_k4(is + 1, blk->scales, &s2, &m2);
		float d2 = d * (float)s2;
		float m2f = dmin * (float)m2;

		// 32 low nibbles
		for (int l = 0; l < 32; l++) {
			uint8_t qval = q[l] & 0x0F;
			float v = d1 * (float)qval - m1f;
			sum += x[j + l] * v;
		}

		// 32 high nibbles
		for (int l = 0; l < 32; l++) {
			uint8_t qval = q[l] >> 4;
			float v = d2 * (float)qval - m2f;
			sum += x[j + 32 + l] * v;
		}

		q += 32; // Advance 64 values (32 bytes)
		is += 2;
	}

	return sum;
}

void mat_vec_row_f32_q4k_f32_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row, int end_row)
{
	const block_q4_k *w = (const block_q4_k *)w_void;
	const int nb = in_dim / QK_K;
	const float *x = (const float *)X;
	float *o = (float *)O;

	for (int i = start_row; i < end_row; i++) {
		const block_q4_k *w_row = w + (long long)i * nb;
		float sum = 0.0f;
		for (int j = 0; j < nb; j++) {
			// Explicitly call the scalar version of the dot product
			sum += dot_product_f32_q4k_scalar(x + j * QK_K, &w_row[j]);
		}
		o[i] = sum;
	}
}

/**
 * @brief Scalar matrix-vector multiplication for Q6_K weights.
 */
void mat_vec_row_f32_q6k_f32_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row, int end_row)
{
	const block_q6_k *w = (const block_q6_k *)w_void;
	const int nb = in_dim / QK_K;
	const float *x = (const float *)X;
	float *o = (float *)O;

	for (int i = start_row; i < end_row; i++) {
		const block_q6_k *w_row = w + (long long)i * nb;
		float sum = 0.0f;
		for (int j = 0; j < nb; j++) {
			// Explicitly call the scalar version of the dot product
			sum += dot_product_f32_q6k_scalar(x + j * QK_K, &w_row[j]);
		}
		o[i] = sum;
	}
}


void mat_vec_row_f32_bf16_f32_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row, int end_row)
{
	float *x = (float *)X;
	float *o = (float *)O;
	uint16_t *w_bf16 = (uint16_t *)w_void;

	for (int i = start_row; i < end_row; i++) {
		float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
		int j = 0;
		const uint16_t *w_row = &w_bf16[i * in_dim];

		for (; j <= in_dim - 4; j += 4) {
			uint32_t w_bits0 = ((uint32_t)w_row[j]) << 16;
			uint32_t w_bits1 = ((uint32_t)w_row[j + 1]) << 16;
			uint32_t w_bits2 = ((uint32_t)w_row[j + 2]) << 16;
			uint32_t w_bits3 = ((uint32_t)w_row[j + 3]) << 16;

			float w0 = *((float *)&w_bits0);
			float w1 = *((float *)&w_bits1);
			float w2 = *((float *)&w_bits2);
			float w3 = *((float *)&w_bits3);

			sum0 = fmaf(x[j], w0, sum0);
			sum1 = fmaf(x[j + 1], w1, sum1);
			sum2 = fmaf(x[j + 2], w2, sum2);
			sum3 = fmaf(x[j + 3], w3, sum3);
		}

		float sum = sum0 + sum1 + sum2 + sum3;

		// Tail case
		for (; j < in_dim; j++) {
			uint32_t w_bits = ((uint32_t)w_row[j]) << 16;
			float w = *((float *)&w_bits);
			sum = fmaf(x[j], w, sum);
		}

		o[i] = sum;
	}
}

// COLUMN VISE !!//
void mat_vec_row_f32_f16_f32_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row, int end_row)
{
	const float *x = (const float *)X;
	float *o = (float *)O;
	const uint16_t *w_f16 = (const uint16_t *)w_void;

	// 'end_row' is effectively the output dimension (out_dim)
	const int out_dim = end_row;

	for (int i = start_row; i < end_row; i++) {
		float sum = 0.0f;

		for (int j = 0; j < in_dim; j++) {
			float w = fp16_to_fp32(w_f16[j * out_dim + i]);
			sum = fmaf(x[j], w, sum);
		}
		o[i] = sum;
	}
}

void mat_vec_row_f32_f32_f32_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row, int end_row)
{
	float *x = (float *)X;
	float *o = (float *)O;
	float *w = (float *)w_void;

	for (int i = start_row; i < end_row; i++) {
		float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
		int j = 0;
		for (; j <= in_dim - 4; j += 4) {
			sum0 += x[j] * w[i * in_dim + j];
			sum1 += x[j + 1] * w[i * in_dim + j + 1];
			sum2 += x[j + 2] * w[i * in_dim + j + 2];
			sum3 += x[j + 3] * w[i * in_dim + j + 3];
		}

		float sum = sum0 + sum1 + sum2 + sum3;
		for (; j < in_dim; j++) {
			sum = fmaf(x[j], w[i * in_dim + j], sum);
		}
		o[i] = sum;
	}
}

void get_embedding_row_q4k_f32_scalar(const Tensor *W, int row_index, void *dest, int embed_dim)
{
	int blocks_per_row = embed_dim / QK_K;
	block_q4_k *src = (block_q4_k *)W->mem.data;
	long long row_block_offset = (long long)row_index * blocks_per_row;

	for (int block_idx = 0; block_idx < blocks_per_row; block_idx++)
		dequantize_row_q4_k_f32_scalar(&src[row_block_offset + block_idx], (float *)dest + block_idx * QK_K,
					       QK_K);
}

void get_embedding_row_q6k_f32_scalar(const Tensor *W, int row_index, void *dest, int embed_dim)
{
	int blocks_per_row = embed_dim / QK_K;

	block_q6_k *src = (block_q6_k *)W->mem.data;
	long long row_block_offset = (long long)row_index * blocks_per_row;

	for (int block_idx = 0; block_idx < blocks_per_row; block_idx++) {
		dequantize_row_q6_k_f32_scalar(&src[row_block_offset + block_idx], (float *)dest + block_idx * QK_K,
					       QK_K);
	}
}

void get_embedding_row_f32_f32_scalar(const Tensor *W, int row_index, void *dest, int embed_dim)
{
	float *src = (float *)W->mem.data;
	long long row_offset = (long long)row_index * embed_dim;

	memcpy((float *)dest, src + row_offset, embed_dim * sizeof(float));
}

void get_embedding_row_bf16_f32_scalar(const Tensor *W, int row_index, void *dest, int embed_dim)
{
	uint16_t *src = (uint16_t *)W->mem.data;
	long long row_offset = (long long)row_index * embed_dim;
	float *d_f32 = (float *)dest;

	for (int i = 0; i < embed_dim; i++) {
		d_f32[i] = bf16_to_fp32(src[row_offset + i]);
	}
}

void store_KV_cache_f32_bf16_scalar(struct TIEContext *ctx, int layer_idx, int start_pos, int batch_len)
{
	LayerKVCache *cache = &ctx->kv_cache[layer_idx];
	int kv_dim = ctx->model->num_kv_heads * ctx->model->head_dim;
	long long cache_offset = (long long)start_pos * kv_dim;
	long long batch_size_elements = (long long)batch_len * kv_dim;

	// Get the raw data pointers from the MemType structs
	uint16_t *k_cache_data = (uint16_t *)cache->k.data;
	uint16_t *v_cache_data = (uint16_t *)cache->v.data;
	float *K_mem_data = (float *)ctx->mem.K.data;
	float *V_mem_data = (float *)ctx->mem.V.data;

	for (long long i = 0; i < batch_size_elements; i++) {
		k_cache_data[cache_offset + i] = fp32_to_bf16(K_mem_data[i]);
		v_cache_data[cache_offset + i] = fp32_to_bf16(V_mem_data[i]);
	}
}

void store_KV_cache_f32_f32_scalar(struct TIEContext *ctx, int layer_idx, int start_pos, int batch_len)
{
	LayerKVCache *cache = &ctx->kv_cache[layer_idx];
	int kv_dim = ctx->model->num_kv_heads * ctx->model->head_dim;
	long long cache_offset = (long long)start_pos * kv_dim;
	long long batch_size_elements = (long long)batch_len * kv_dim;

	// Get the raw data pointers from the MemType structs
	float *k_cache_data = (float *)cache->k.data;
	float *v_cache_data = (float *)cache->v.data;
	float *K_mem_data = (float *)ctx->mem.K.data;
	float *V_mem_data = (float *)ctx->mem.V.data;

	for (long long i = 0; i < batch_size_elements; i++) {
		k_cache_data[cache_offset + i] = K_mem_data[i];
		v_cache_data[cache_offset + i] = V_mem_data[i];
	}
}

// SwiGLU activation FP32, FP32
void swiglu_activation_f32_f32_scalar(void *gate, const void *up, int size)
{
	float *gate_fp32 = (float *)gate;
	float *up_fp32 = (float *)up;

	for (int i = 0; i < size; i++) {
		gate_fp32[i] = silu_lookup(gate_fp32[i]) * up_fp32[i];
	}
}

void get_embedding_row_bf16_bf16_scalar(const Tensor *W, int row_index, void *dest, int embed_dim)
{
	uint16_t *src = (uint16_t *)W->mem.data;
	long long row_offset = (long long)row_index * embed_dim;
	uint16_t *d_u16 = (uint16_t *)dest;

	for (int i = 0; i < embed_dim; i++) {
		d_u16[i] = src[row_offset + i];
	}
}

void rms_norm_bf16_f32_bf16_scalar(void *O, const void *X, const Tensor *W, int size, float eps)
{
	const float *weight = (const float *)W->mem.data;
	const uint16_t *x_bf16 = (const uint16_t *)X;
	uint16_t *o_bf16 = (uint16_t *)O;

	// First pass: accumulate sum of squares
	float ss = 0.0f;
	for (int i = 0; i < size; i++) {
		float x = bf16_to_fp32(x_bf16[i]);
		ss += x * x;
	}

	float inv_rms = 1.0f / sqrtf(ss / size + eps);

	// Second pass: normalize, apply weight, convert back to bf16
	for (int i = 0; i < size; i++) {
		float x = bf16_to_fp32(x_bf16[i]);
		float o = x * inv_rms * weight[i];
		o_bf16[i] = fp32_to_bf16_rne(o);
	}
}

void rms_norm_bf16_f32_f32_scalar(void *O, const void *X, const Tensor *W, int size, float eps)
{
	float *weight = (float *)W->mem.data;
	uint16_t *x_bf16 = (uint16_t *)X;
	float *o_f32 = (float *)O;
	float sum = 0.0f;
	for (int i = 0; i < size; i++) {
		float x = bf16_to_fp32(x_bf16[i]);
		sum += x * x;
	}
	//	float rms = sqrtf(sum / size + eps);
	float inv_rms = 1.0f / sqrtf(sum / size + eps);
	for (int i = 0; i < size; i++) {
		//		o_f32[i] = bf16_to_fp32(x_bf16[i]) / rms * weight[i];
		o_f32[i] = bf16_to_fp32(x_bf16[i]) * inv_rms * weight[i];
	}
}

float dot_product_bf16_q4k_scalar(const uint16_t *x_bf16, const block_q4_k *blk)
{
	const float d = fp16_to_fp32(blk->d);
	const float dmin = fp16_to_fp32(blk->dmin);
	const uint8_t *q = blk->qs;
	float sum = 0.0f;
	int is = 0;

	for (int j = 0; j < QK_K; j += 64) {
		uint8_t s1, m1, s2, m2;
		get_scale_min_k4(is + 0, blk->scales, &s1, &m1);
		float d1 = d * (float)s1;
		float m1f = dmin * (float)m1;
		get_scale_min_k4(is + 1, blk->scales, &s2, &m2);
		float d2 = d * (float)s2;
		float m2f = dmin * (float)m2;

		for (int l = 0; l < 32; l++) {
			float x_val = bf16_to_fp32(x_bf16[j + l]);
			float v = d1 * (float)(q[l] & 0x0F) - m1f;
			sum += x_val * v;
		}
		for (int l = 0; l < 32; l++) {
			float x_val = bf16_to_fp32(x_bf16[j + 32 + l]);
			float v = d2 * (float)(q[l] >> 4) - m2f;
			sum += x_val * v;
		}
		q += 32;
		is += 2;
	}
	return sum;
}

void mat_vec_row_bf16_q4k_f32_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row, int end_row)
{
	const block_q4_k *w = (const block_q4_k *)w_void;
	const int nb = in_dim / QK_K;
	const uint16_t *x = (const uint16_t *)X;
	float *o = (float *)O;

	for (int i = start_row; i < end_row; i++) {
		const block_q4_k *w_row = w + (long long)i * nb;
		float sum = 0.0f;
		for (int j = 0; j < nb; j++) {
			sum += dot_product_bf16_q4k_scalar(x + j * QK_K, &w_row[j]);
		}
		o[i] = sum;
	}
}

void mat_vec_row_bf16_q4k_bf16_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row,
				      int end_row)
{
	const block_q4_k *w = (const block_q4_k *)w_void;
	const int nb = in_dim / QK_K;
	const uint16_t *x = (const uint16_t *)X;
	uint16_t *o = (uint16_t *)O;

	for (int i = start_row; i < end_row; i++) {
		const block_q4_k *w_row = w + (long long)i * nb;
		float sum = 0.0f;
		for (int j = 0; j < nb; j++) {
			sum += dot_product_bf16_q4k_scalar(x + j * QK_K, &w_row[j]);
		}
		o[i] = fp32_to_bf16_rne(sum);
	}
}

void mat_vec_row_f32_q4k_bf16_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row, int end_row)
{
	const block_q4_k *w = (const block_q4_k *)w_void;
	const int nb = in_dim / QK_K;
	const float *x = (const float *)X;
	uint16_t *o = (uint16_t *)O;

	for (int i = start_row; i < end_row; i++) {
		const block_q4_k *w_row = w + (long long)i * nb;
		float sum = 0.0f;
		for (int j = 0; j < nb; j++) {
			sum += dot_product_f32_q4k_scalar(x + j * QK_K, &w_row[j]);
		}
		o[i] = fp32_to_bf16_rne(sum);
	}
}

void mat_vec_row_bf16_f32_f32_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row, int end_row)
{
	uint16_t *x = (uint16_t *)X;
	float *o = (float *)O;
	float *w = (float *)w_void;

	for (int i = start_row; i < end_row; i++) {
		float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
		int j = 0;
		for (; j <= in_dim - 4; j += 4) {
			sum0 += bf16_to_fp32(x[j]) * w[i * in_dim + j];
			sum1 += bf16_to_fp32(x[j + 1]) * w[i * in_dim + j + 1];
			sum2 += bf16_to_fp32(x[j + 2]) * w[i * in_dim + j + 2];
			sum3 += bf16_to_fp32(x[j + 3]) * w[i * in_dim + j + 3];
		}

		float sum = sum0 + sum1 + sum2 + sum3;
		for (; j < in_dim; j++) {
			sum = fmaf(bf16_to_fp32(x[j]), w[i * in_dim + j], sum);
		}
		o[i] = sum;
	}
}

float dot_product_bf16_q6k_scalar(const uint16_t *x, const block_q6_k *block)
{
	const float d = fp16_to_fp32(block->d);
	const uint8_t *ql = block->ql;
	const uint8_t *qh = block->qh;
	const int8_t *sc = block->scales;
	float sum = 0.0f;

	// A block has 256 elements, which we process in two 128-element halves.
	for (int half = 0; half < 2; ++half) {
		// Calculate the base pointers for this half
		const uint8_t *ql_half = ql + half * 64;
		const uint8_t *qh_half = qh + half * 32;
		const int8_t *sc_half = sc + half * 8;
		const uint16_t *x_half = x + half * 128;

		// Inner loop processes 4 elements at a time across the 128-element half
		for (int l = 0; l < 32; ++l) {
			int is = l / 16; // Scale index (0 for first 16, 1 for second 16)

			// Dequantize the 4 values for this step
			int8_t q1 = ((ql_half[l] & 0x0F) | ((qh_half[l] >> 0) & 3) << 4) - 32;
			int8_t q2 = ((ql_half[l + 32] & 0x0F) | ((qh_half[l] >> 2) & 3) << 4) - 32;
			int8_t q3 = ((ql_half[l] >> 4) | ((qh_half[l] >> 4) & 3) << 4) - 32;
			int8_t q4 = ((ql_half[l + 32] >> 4) | ((qh_half[l] >> 6) & 3) << 4) - 32;

			// Multiply with the CORRECT corresponding x values from the input vector
			sum += d * sc_half[is + 0] * q1 * bf16_to_fp32(x_half[l]);
			sum += d * sc_half[is + 2] * q2 * bf16_to_fp32(x_half[l + 32]);
			sum += d * sc_half[is + 4] * q3 * bf16_to_fp32(x_half[l + 64]);
			sum += d * sc_half[is + 6] * q4 * bf16_to_fp32(x_half[l + 96]);
		}
	}

	return sum;
}

void mat_vec_row_bf16_q6k_f32_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row, int end_row)
{
	const block_q6_k *w = (const block_q6_k *)w_void;
	const int nb = in_dim / QK_K;
	const uint16_t *x = (const uint16_t *)X;
	float *o = (float *)O;

	for (int i = start_row; i < end_row; i++) {
		const block_q6_k *w_row = w + (long long)i * nb;
		float sum = 0.0f;
		for (int j = 0; j < nb; j++) {
			// Explicitly call the scalar version of the dot product
			sum += dot_product_bf16_q6k_scalar(x + j * QK_K, &w_row[j]);
		}
		o[i] = sum;
	}
}

void mat_vec_row_bf16_bf16_bf16_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row,
				       int end_row)
{
	uint16_t *x = (uint16_t *)X;
	uint16_t *o = (uint16_t *)O;
	uint16_t *w = (uint16_t *)w_void;

	for (int i = start_row; i < end_row; i++) {
		float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
		int j = 0;
		for (; j <= in_dim - 4; j += 4) {
			sum0 += bf16_to_fp32(x[j]) * bf16_to_fp32(w[i * in_dim + j]);
			sum1 += bf16_to_fp32(x[j + 1]) * bf16_to_fp32(w[i * in_dim + j + 1]);
			sum2 += bf16_to_fp32(x[j + 2]) * bf16_to_fp32(w[i * in_dim + j + 2]);
			sum3 += bf16_to_fp32(x[j + 3]) * bf16_to_fp32(w[i * in_dim + j + 3]);
		}

		float sum = sum0 + sum1 + sum2 + sum3;
		for (; j < in_dim; j++) {
			sum = fmaf(bf16_to_fp32(x[j]), bf16_to_fp32(w[i * in_dim + j]), sum);
		}
		o[i] = fp32_to_bf16_rne(sum);
	}
}

void mat_vec_row_bf16_bf16_f32_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row,
				      int end_row)
{
	uint16_t *x = (uint16_t *)X;
	float *o = (float *)O;
	uint16_t *w = (uint16_t *)w_void;

	for (int i = start_row; i < end_row; i++) {
		float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
		int j = 0;
		for (; j <= in_dim - 4; j += 4) {
			sum0 += bf16_to_fp32(x[j]) * bf16_to_fp32(w[i * in_dim + j]);
			sum1 += bf16_to_fp32(x[j + 1]) * bf16_to_fp32(w[i * in_dim + j + 1]);
			sum2 += bf16_to_fp32(x[j + 2]) * bf16_to_fp32(w[i * in_dim + j + 2]);
			sum3 += bf16_to_fp32(x[j + 3]) * bf16_to_fp32(w[i * in_dim + j + 3]);
		}

		float sum = sum0 + sum1 + sum2 + sum3;
		for (; j < in_dim; j++) {
			sum = fmaf(bf16_to_fp32(x[j]), bf16_to_fp32(w[i * in_dim + j]), sum);
		}
		o[i] = sum;
	}
}

void mat_vec_row_bf16_q6k_bf16_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row,
				      int end_row)
{
	const block_q6_k *w = (const block_q6_k *)w_void;
	const int nb = in_dim / QK_K;
	const uint16_t *x = (const uint16_t *)X;
	uint16_t *o = (uint16_t *)O;

	for (int i = start_row; i < end_row; i++) {
		const block_q6_k *w_row = w + (long long)i * nb;
		float sum = 0.0f;
		for (int j = 0; j < nb; j++) {
			// Explicitly call the scalar version of the dot product
			sum += dot_product_bf16_q6k_scalar(x + j * QK_K, &w_row[j]);
		}
		o[i] = fp32_to_bf16_rne(sum);
	}
}

void mat_vec_row_f32_q6k_bf16_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row, int end_row)
{
	const block_q6_k *w = (const block_q6_k *)w_void;
	const int nb = in_dim / QK_K;
	const float *x = (const float *)X;
	uint16_t *o = (uint16_t *)O;

	for (int i = start_row; i < end_row; i++) {
		const block_q6_k *w_row = w + (long long)i * nb;
		float sum = 0.0f;
		for (int j = 0; j < nb; j++) {
			// Explicitly call the scalar version of the dot product
			sum += dot_product_f32_q6k_scalar(x + j * QK_K, &w_row[j]);
		}
		o[i] = fp32_to_bf16_rne(sum);
	}
}

void dequantize_row_q4_k_bf16_scalar(const void *__restrict__ q_void, uint16_t *__restrict__ y, int k)
{
	const block_q4_k *blocks = (const block_q4_k *)q_void;
	const int nb = k / QK_K; // k is the total number of elements, e.g., model dimension

	// Iterate over the blocks that make up the row
	for (int i = 0; i < nb; ++i) {
		const block_q4_k *blk = &blocks[i];
		const float d = fp16_to_fp32(blk->d);
		const float dmin = fp16_to_fp32(blk->dmin);
		const uint8_t *q = blk->qs;
		int is = 0;

		uint16_t *y_block = y + i * QK_K;

		for (int j = 0; j < QK_K; j += 64) {
			uint8_t s1, m1, s2, m2;

			get_scale_min_k4(is + 0, blk->scales, &s1, &m1);
			const float d1 = d * s1;
			const float m1f = dmin * m1;

			get_scale_min_k4(is + 1, blk->scales, &s2, &m2);
			const float d2 = d * s2;
			const float m2f = dmin * m2;

			// Dequantize 32 low-nibble values
			for (int l = 0; l < 32; ++l) {
				y_block[j + l] = fp32_to_bf16_rne(d1 * (float)(q[l] & 0x0F) - m1f);
			}

			// Dequantize 32 high-nibble values
			for (int l = 0; l < 32; ++l) {
				y_block[j + 32 + l] = fp32_to_bf16_rne(d2 * (float)(q[l] >> 4) - m2f);
			}
			q += 32;
			is += 2;
		}
	}
}

void dequantize_row_q6_k_bf16_scalar(const void *__restrict__ q_void, uint16_t *__restrict__ y, int k)
{
	const block_q6_k *x = (const block_q6_k *)q_void;
	const int64_t nb = k / QK_K;

	for (int i = 0; i < nb; i++) {
		const float d = fp16_to_fp32(x[i].d);

		const uint8_t *ql = x[i].ql;
		const uint8_t *qh = x[i].qh;
		const int8_t *sc = x[i].scales;

		for (int n = 0; n < QK_K; n += 128) {
			for (int l = 0; l < 32; ++l) {
				int is = l / 16;
				const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
				const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
				const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
				const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
				y[l + 0] = fp32_to_bf16_rne(d * sc[is + 0] * q1);
				y[l + 32] = fp32_to_bf16_rne(d * sc[is + 2] * q2);
				y[l + 64] = fp32_to_bf16_rne(d * sc[is + 4] * q3);
				y[l + 96] = fp32_to_bf16_rne(d * sc[is + 6] * q4);
			}
			y += 128;
			ql += 64;
			qh += 32;
			sc += 8;
		}
	}
}

void get_embedding_row_q4k_bf16_scalar(const Tensor *W, int row_index, void *dest, int embed_dim)
{
	int blocks_per_row = embed_dim / QK_K;
	block_q4_k *src = (block_q4_k *)W->mem.data;
	long long row_block_offset = (long long)row_index * blocks_per_row;

	for (int block_idx = 0; block_idx < blocks_per_row; block_idx++)
		dequantize_row_q4_k_bf16_scalar(&src[row_block_offset + block_idx], (uint16_t *)dest + block_idx * QK_K,
						QK_K);
}

void get_embedding_row_q6k_bf16_scalar(const Tensor *W, int row_index, void *dest, int embed_dim)
{
	int blocks_per_row = embed_dim / QK_K;

	block_q6_k *src = (block_q6_k *)W->mem.data;
	long long row_block_offset = (long long)row_index * blocks_per_row;

	for (int block_idx = 0; block_idx < blocks_per_row; block_idx++) {
		dequantize_row_q6_k_bf16_scalar(&src[row_block_offset + block_idx], (uint16_t *)dest + block_idx * QK_K,
						QK_K);
	}
}

void apply_residual_bf16_f32_scalar(void *acc_void, const void *residual_void, int size)
{
	float *acc = (float *)acc_void;
	uint16_t *residual = (uint16_t *)residual_void;

	for (int i = 0; i < size; i++) {
		acc[i] = fmaf(bf16_to_fp32(residual[i]), 1.0f, acc[i]);
	}
}

void apply_residual_bf16_bf16_scalar(void *acc_void, const void *residual_void, int size)
{
	uint16_t *acc = (uint16_t *)acc_void;
	uint16_t *residual = (uint16_t *)residual_void;

	for (int i = 0; i < size; i++) {
		acc[i] = fp32_to_bf16_rne(fmaf(bf16_to_fp32(residual[i]), 1.0f, bf16_to_fp32(acc[i])));
	}
}

void apply_residual_f32_bf16_scalar(void *acc_void, const void *residual_void, int size)
{
	uint16_t *acc = (uint16_t *)acc_void;
	float *residual = (float *)residual_void;
	float acc_32;

	for (int i = 0; i < size; i++) {
		acc_32 = fmaf(residual[i], 1.0f, bf16_to_fp32(acc[i]));
		acc[i] = fp32_to_bf16_rne(acc_32);
	}
}

void accumulate_weighted_V_bf16_bf16_scalar(void *O, float weight, const void *V, int size)
{
	uint16_t *out = (uint16_t *)O;
	uint16_t *v = (uint16_t *)V;

	for (int i = 0; i < size; i++) {
		out[i] = fp32_to_bf16_rne(fmaf(weight, bf16_to_fp32(v[i]), bf16_to_fp32(out[i])));
	}
}

void swiglu_activation_bf16_bf16_scalar(void *gate_void, const void *up_void, int size)
{
	uint16_t *gate = (uint16_t *)gate_void;
	const uint16_t *up = (const uint16_t *)up_void;

	for (int i = 0; i < size; i++) {
		float gate_f32 = bf16_to_fp32(gate[i]);
		float up_f32 = bf16_to_fp32(up[i]);
		gate[i] = fp32_to_bf16_rne(silu_lookup(gate_f32) * up_f32);
	}
}

void store_KV_cache_bf16_bf16_scalar(struct TIEContext *ctx, int layer_idx, int start_pos, int batch_len)
{
	LayerKVCache *cache = &ctx->kv_cache[layer_idx];
	int kv_dim = ctx->model->num_kv_heads * ctx->model->head_dim;
	long long cache_offset = (long long)start_pos * kv_dim;
	long long batch_size_elements = (long long)batch_len * kv_dim;

	// Get the raw data pointers from the MemType structs
	uint16_t *k_cache_data = (uint16_t *)cache->k.data;
	uint16_t *v_cache_data = (uint16_t *)cache->v.data;
	uint16_t *K_mem_data = (uint16_t *)ctx->mem.K.data;
	uint16_t *V_mem_data = (uint16_t *)ctx->mem.V.data;

	for (long long i = 0; i < batch_size_elements; i++) {
		k_cache_data[cache_offset + i] = K_mem_data[i];
		v_cache_data[cache_offset + i] = V_mem_data[i];
	}
}

void apply_rope_cache_bf16_scalar(RopeCacheType *rope_cache, void *X, int pos, int head_dim)
{
	int h_dim_half = head_dim / 2;
	uint16_t *x = (uint16_t *)X;

	const float *sin_vals = rope_cache->sin + pos * h_dim_half;
	const float *cos_vals = rope_cache->cos + pos * h_dim_half;

	for (int i = 0; i < h_dim_half; ++i) {
		float x_real = bf16_to_fp32(x[i]);
		float x_imag = bf16_to_fp32(x[i + h_dim_half]);
		float sin_val = sin_vals[i];
		float cos_val = cos_vals[i];

		float new_real = x_real * cos_val - x_imag * sin_val;
		float new_imag = x_real * sin_val + x_imag * cos_val;

		x[i] = fp32_to_bf16_rne(new_real);
		x[i + h_dim_half] = fp32_to_bf16_rne(new_imag);
	}
}

inline float gelu_fast(float x)
{
	return 0.5f * x * (1.0f + tanhf(0.79788456f * x * (1.0f + 0.044715f * x * x)));
}

void geglu_activation_f32_f32_scalar(void *gate, const void *up, int size)
{
	float *gate_fp32 = (float *)gate;
	float *up_fp32 = (float *)up;

	for (int i = 0; i < size; i++) {
		gate_fp32[i] = gelu_fast(gate_fp32[i]) * up_fp32[i];
	}
}

void dispatch_conv_2d_scalar(MemType *dest, const MemType *src_image, const Tensor *kernel_tensor,
			     const Tensor *bias_tensor, int H_in, int W_in, int stride, int padding)
{
	// --- 1. Get pointers and dimensions ---
	float *dest_data = (float *)dest->data;
	const float *src_data = (const float *)src_image->data;
	const float *kernel_data = (const float *)kernel_tensor->mem.data;
	const float *bias_data = (const float *)bias_tensor->mem.data;

	// KERNEL DIMENSIONS (from GGUF metadata [K_H, K_W, C_in, C_out])
	const int K_H = kernel_tensor->dimensions[0];	// 14
	const int K_W = kernel_tensor->dimensions[1];	// 14
	const int C_in = kernel_tensor->dimensions[2];	// 3
	const int C_out = kernel_tensor->dimensions[3]; // 1152

	// INPUT DIMENSIONS (planar [C, H, W])
	const size_t src_plane_size = (size_t)H_in * W_in;

	// OUTPUT DIMENSIONS
	const int H_out = (H_in + 2 * padding - K_H) / stride + 1; // 64
	const int W_out = (W_in + 2 * padding - K_W) / stride + 1; // 64
	const size_t dest_plane_size = (size_t)H_out * W_out;	   // 4096

	// --- KERNEL MEMORY LAYOUT (Actual layout in memory is [C_out, C_in, K_H, K_W]) ---
	// We must use this layout for calculating offsets
	const size_t kernel_stride_K_W = 1;
	const size_t kernel_stride_K_H = (size_t)K_W * kernel_stride_K_W;
	const size_t kernel_stride_C_in = (size_t)K_H * kernel_stride_K_H;
	const size_t kernel_stride_C_out = (size_t)C_in * kernel_stride_C_in;

	// --- 2. Perform Convolution ---
	for (int c_out = 0; c_out < C_out; ++c_out) {

		const float bias = bias_data[c_out];
		float *dest_plane_ptr = dest_data + (c_out * dest_plane_size);

		// Get the pointer to the start of this filter's weights
		// [c_out, 0, 0, 0]
		const float *kernel_filter_ptr = kernel_data + (c_out * kernel_stride_C_out);

		for (int y_out = 0; y_out < H_out; ++y_out) {
			for (int x_out = 0; x_out < W_out; ++x_out) {

				float sum = bias;

				// --- 3. Apply the 3D kernel [C_in, K_H, K_W] ---
				for (int c_in = 0; c_in < C_in; ++c_in) {

					const float *src_plane_ptr = src_data + (c_in * src_plane_size);

					// Get pointer to kernel[c_out][c_in][0][0]
					const float *kernel_plane_ptr = kernel_filter_ptr + (c_in * kernel_stride_C_in);

					for (int ky = 0; ky < K_H; ++ky) {
						const int y_in = y_out * stride + ky - padding;

						// Boundary check for Y
						if (y_in < 0 || y_in >= H_in) {
							continue;
						}

						// Get pointer to kernel[c_out][c_in][ky][0]
						const float *kernel_row_ptr =
							kernel_plane_ptr + (ky * kernel_stride_K_H);

						for (int kx = 0; kx < K_W; ++kx) {
							const int x_in = x_out * stride + kx - padding;

							// Boundary check for X
							if (x_in < 0 || x_in >= W_in) {
								continue;
							}

							// KERNEL INDEX CALCULATION (THE FIX)
							// This is just kernel_row_ptr[kx]
							const float kernel_val = kernel_row_ptr[kx * kernel_stride_K_W];

							// Input index
							size_t src_idx = (size_t)y_in * W_in + x_in;

							// Accumulate: sum += input * weight
							sum = fmaf(src_plane_ptr[src_idx], kernel_val, sum);

						} // kx
					} // ky
				} // c_in

				// Store the final convoluted pixel value
				dest_plane_ptr[y_out * W_out + x_out] = sum;

			} // x_out
		} // y_out
	} // c_out
}
