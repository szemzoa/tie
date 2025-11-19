#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include "config.h"
#include "engine.h"
#include "math_avx2.h"
#include "math_scalar.h"

#ifdef CONFIG_ENABLE_AVX2
#include <immintrin.h>

__attribute__((target("avx2"))) static inline float hsum_f32_avx2(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(_mm_hadd_ps(sum, sum));
}

__attribute__((target("avx2"))) inline __m256 load_bf16_as_f32(const uint16_t *src)
{
	// Load 8 BF16s into 128-bit register
	__m128i bf16_vals = _mm_loadu_si128((const __m128i *)src);

	// Zero-extend to 32-bit ints
	__m256i extended = _mm256_cvtepu16_epi32(bf16_vals); // 8 x uint32_t

	// Shift left by 16 bits to align to FP32 format
	extended = _mm256_slli_epi32(extended, 16);

	// Reinterpret as float
	return _mm256_castsi256_ps(extended);
}

__attribute__((target("avx2"))) static inline void store_f32_as_bf16_avx2(uint16_t *dst, __m256 src_ps)
{
	__m256i val_i32 = _mm256_castps_si256(src_ps);
	__m256i shifted = _mm256_srli_epi32(val_i32, 16);

	__m128i low_lane = _mm256_castsi256_si128(shifted);
	__m128i high_lane = _mm256_extracti128_si256(shifted, 1);

	__m128i mask = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 9, 8, 5, 4, 1, 0);

	__m128i packed_low = _mm_shuffle_epi8(low_lane, mask);
	__m128i packed_high = _mm_shuffle_epi8(high_lane, mask);

	__m128i final_result = _mm_or_si128(packed_low, _mm_slli_si128(packed_high, 8));

	_mm_storeu_si128((__m128i *)dst, final_result);
}


__attribute__((target("avx2"))) void rms_norm_f32_f32_f32_avx2(void *__restrict O, const void *__restrict X,
							       const Tensor *__restrict W, int size, float eps)
{
	float *o = (float *)O;
	float *x = (float *)X;
	float *weight = W->mem.data;

	__m256 sum = _mm256_setzero_ps();
	int i = 0;

	// Sum of squares using 8-wide FMA
	for (; i <= size - 8; i += 8) {
		__m256 x_vec = _mm256_loadu_ps(&x[i]);
		sum = _mm256_fmadd_ps(x_vec, x_vec, sum);
	}

	// Horizontal reduction of 8 floats
	float ss = hsum_f32_avx2(sum);

	// Tail loop
	for (; i < size; ++i) {
		ss = fmaf(x[i], x[i], ss);
	}

	float inv_rms = 1.0f / sqrtf(ss / size + eps);

	// Scale + weight
	i = 0;
	__m256 scale = _mm256_set1_ps(inv_rms);
	for (; i <= size - 8; i += 8) {
		__m256 x_vec = _mm256_loadu_ps(&x[i]);
		__m256 w_vec = _mm256_loadu_ps(&weight[i]);
		__m256 out = _mm256_mul_ps(x_vec, w_vec);
		out = _mm256_mul_ps(out, scale);
		_mm256_storeu_ps(&o[i], out);
	}

	// Tail
	for (; i < size; ++i) {
		o[i] = x[i] * weight[i] * inv_rms;
	}
}

__attribute__((target("avx2"))) void rms_norm_bf16_f32_f32_avx2(void *__restrict o_void, const void *__restrict x_void,
								const Tensor *__restrict W, int size, float eps)
{
	const uint16_t *x = (const uint16_t *)x_void;
	uint16_t *o = (uint16_t *)o_void;
	const float *weight = (const float *)W->mem.data;

	// --- 1. Sum of squares (computation is in FP32) ---
	__m256 sum_sq_vec = _mm256_setzero_ps();
	int i = 0;
	for (; i <= size - 8; i += 8) {
		// Load 8 BF16 values and convert to FP32
		__m256 x_vec_f32 = load_bf16_as_f32(&x[i]);
		// Fused multiply-add: sum += x * x
		sum_sq_vec = _mm256_fmadd_ps(x_vec_f32, x_vec_f32, sum_sq_vec);
	}

	// Horizontal reduction of the sum vector to a single float 'ss'
	float ss = hsum_f32_avx2(sum_sq_vec);

	// Scalar tail loop for sum of squares
	for (; i < size; ++i) {
		float val = bf16_to_fp32(x[i]);
		ss += val * val;
	}

	// --- 2. Scale and Store ---
	const float inv_rms = 1.0f / sqrtf(ss / size + eps);
	const __m256 scale_vec = _mm256_set1_ps(inv_rms);

	i = 0;
	for (; i <= size - 8; i += 8) {
		__m256 x_vec_f32 = load_bf16_as_f32(&x[i]);
		__m256 w_vec_f32 = _mm256_loadu_ps(&weight[i]);

		// Normalize and scale in FP32
		__m256 out_f32 = _mm256_mul_ps(_mm256_mul_ps(x_vec_f32, w_vec_f32), scale_vec);

		// Convert back to BF16 and store
		store_f32_as_bf16_avx2(&o[i], out_f32);
	}

	// Scalar tail loop for scaling
	for (; i < size; ++i) {
		float val = bf16_to_fp32(x[i]);
		o[i] = fp32_to_bf16(val * weight[i] * inv_rms);
	}
}

void mat_vec_row_f32_bf16_f32_avx2(const void *__restrict X, const void *__restrict w_void, void *__restrict O,
				   int in_dim, int start_row, int end_row)
{
	const float *x = (const float *)X;
	const uint16_t *w_bf16 = (const uint16_t *)w_void;
	float *o = (float *)O;

	for (int i = start_row; i < end_row; i++) {
		const uint16_t *w_row = &w_bf16[i * in_dim];
		o[i] = dot_product_f32_bf16_avx2(x, w_row, in_dim);
	}
}

__attribute__((target("avx2"))) float dot_product_f32_f32_avx2(const void *__restrict A, const void *__restrict B,
							       int size)
{
	float *a = (float *)A;
	float *b = (float *)B;

	__m256 sum_vec = _mm256_setzero_ps(); // Initialize 8-lane sum to zero
	int i;

	// Process 8 elements at a time
	for (i = 0; i <= size - 8; i += 8) {
		__m256 a_vec = _mm256_loadu_ps(a + i);		  // Unaligned load of 8 floats from a
		__m256 b_vec = _mm256_loadu_ps(b + i);		  // Unaligned load of 8 floats from b
		sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec); // sum += a * b
	}

	// Horizontal sum of 8 lanes
	float sum = hsum_f32_avx2(sum_vec);

	// Handle remaining elements (size % 8)
	for (; i < size; i++) {
		sum = fmaf(a[i], b[i], sum);
	}

	return sum;
}

__attribute__((target("avx2"))) float dot_product_f32_bf16_avx2(const void *__restrict A, const void *__restrict B,
								int size)
{
	float *a = (float *)A;
	uint16_t *b = (uint16_t *)B;

	__m256 acc = _mm256_setzero_ps();

	int i = 0;
	for (; i + 8 <= size; i += 8) {
		// Load 8 BF16 values (16-bit integers)
		__m128i b_half = _mm_loadu_si128((const __m128i *)(b + i)); // 8 * uint16_t

		// Zero-extend to 32-bit integers
		__m256i b_i32 = _mm256_cvtepu16_epi32(b_half); // 8 * uint32_t

		// Shift left by 16 bits → BF16 to FP32 bits
		__m256i b_fp32_bits = _mm256_slli_epi32(b_i32, 16);

		// Reinterpret as float
		__m256 b_fp32 = _mm256_castsi256_ps(b_fp32_bits);

		// Load 8 floats from a
		__m256 a_fp32 = _mm256_loadu_ps(a + i);

		// Multiply-accumulate
		acc = _mm256_fmadd_ps(a_fp32, b_fp32, acc);
	}

	float result = hsum_f32_avx2(acc);

	// Tail loop
	for (; i < size; i++) {
		float b_val = bf16_to_fp32(b[i]);
		result = fmaf(a[i], b_val, result);
	}

	return result;
}

__attribute__((target("avx2"))) void mat_vec_row_f32_f32_f32_avx2(const void *X, const void *w_void, void *O,
								  int in_dim, int start_row, int end_row)
{
	float *x = (float *)X;
	float *w = (float *)w_void;
	float *o = (float *)O;

	for (int i = start_row; i < end_row; i++) {
		const float *w_row = &w[i * in_dim];
		__m256 acc = _mm256_setzero_ps();

		int j = 0;
		for (; j <= in_dim - 8; j += 8) {
			__m256 w_vec = _mm256_loadu_ps(&w_row[j]); // 8 floats from weight row
			__m256 x_vec = _mm256_loadu_ps(&x[j]);	   // 8 floats from input vector
			acc = _mm256_fmadd_ps(x_vec, w_vec, acc);  // acc += x * w
		}

		// Horizontal sum of acc
		float sum = hsum_f32_avx2(acc);

		// Tail
		for (; j < in_dim; j++) {
			sum = fmaf(x[j], w_row[j], sum);
		}

		o[i] = sum;
	}
}

__attribute__((target("avx2"))) static void dequantize_row_q6k_f32_avx2(const void *__restrict__ q_void,
									float *__restrict__ y, int k)
{
	const block_q6_k *x = (const block_q6_k *)q_void;
	const int64_t nb = k / QK_K;

	for (int i = 0; i < nb; i++) {
		const float d = fp16_to_fp32(x[i].d);
		const uint8_t *ql = x[i].ql;
		const uint8_t *qh = x[i].qh;
		const int8_t *sc = x[i].scales;

		__m256 d_vec = _mm256_set1_ps(d);

		for (int n = 0; n < QK_K; n += 128) {
			for (int l = 0; l < 32; l += 8) {
				int is = l / 16;

				// Process q1 (y[l + 0])
				{
					__m128i ql_vec = _mm_loadu_si128((const __m128i *)(ql + l));
					__m128i qh_vec = _mm_loadu_si128((const __m128i *)(qh + l));
					__m128i ql_low = _mm_and_si128(ql_vec, _mm_set1_epi8(0x0F));
					__m128i qh_01 = _mm_and_si128(qh_vec, _mm_set1_epi8(3));
					__m128i qh_shift = _mm_slli_epi16(qh_01, 4);
					__m128i q_vec = _mm_or_si128(ql_low, qh_shift);
					__m128i q_int = _mm_sub_epi8(q_vec, _mm_set1_epi8(32));
					__m256i q_32 = _mm256_cvtepi8_epi32(q_int);
					__m256 q1_vec = _mm256_cvtepi32_ps(q_32);
					__m256 scale_vec = _mm256_set1_ps((float)sc[is]);
					q1_vec = _mm256_mul_ps(q1_vec, _mm256_mul_ps(d_vec, scale_vec));
					_mm256_storeu_ps(y + l, q1_vec);
				}

				// Process q2 (y[l + 32])
				{
					__m128i ql_vec = _mm_loadu_si128((const __m128i *)(ql + l + 32));
					__m128i qh_vec = _mm_loadu_si128((const __m128i *)(qh + l));
					__m128i ql_low = _mm_and_si128(ql_vec, _mm_set1_epi8(0x0F));
					__m128i qh_23 = _mm_and_si128(_mm_srli_epi16(qh_vec, 2), _mm_set1_epi8(3));
					__m128i qh_shift = _mm_slli_epi16(qh_23, 4);
					__m128i q_vec = _mm_or_si128(ql_low, qh_shift);
					__m128i q_int = _mm_sub_epi8(q_vec, _mm_set1_epi8(32));
					__m256i q_32 = _mm256_cvtepi8_epi32(q_int);
					__m256 q2_vec = _mm256_cvtepi32_ps(q_32);
					__m256 scale_vec = _mm256_set1_ps((float)sc[is + 2]);
					q2_vec = _mm256_mul_ps(q2_vec, _mm256_mul_ps(d_vec, scale_vec));
					_mm256_storeu_ps(y + l + 32, q2_vec);
				}

				// Process q3 (y[l + 64])
				{
					__m128i ql_vec = _mm_loadu_si128((const __m128i *)(ql + l));
					__m128i qh_vec = _mm_loadu_si128((const __m128i *)(qh + l));
					__m128i ql_high = _mm_and_si128(_mm_srli_epi16(ql_vec, 4), _mm_set1_epi8(0x0F));
					__m128i qh_45 = _mm_and_si128(_mm_srli_epi16(qh_vec, 4), _mm_set1_epi8(3));
					__m128i qh_shift = _mm_slli_epi16(qh_45, 4);
					__m128i q_vec = _mm_or_si128(ql_high, qh_shift);
					__m128i q_int = _mm_sub_epi8(q_vec, _mm_set1_epi8(32));
					__m256i q_32 = _mm256_cvtepi8_epi32(q_int);
					__m256 q3_vec = _mm256_cvtepi32_ps(q_32);
					__m256 scale_vec = _mm256_set1_ps((float)sc[is + 4]);
					q3_vec = _mm256_mul_ps(q3_vec, _mm256_mul_ps(d_vec, scale_vec));
					_mm256_storeu_ps(y + l + 64, q3_vec);
				}

				// Process q4 (y[l + 96])
				{
					__m128i ql_vec = _mm_loadu_si128((const __m128i *)(ql + l + 32));
					__m128i qh_vec = _mm_loadu_si128((const __m128i *)(qh + l));
					__m128i ql_high = _mm_and_si128(_mm_srli_epi16(ql_vec, 4), _mm_set1_epi8(0x0F));
					__m128i qh_67 = _mm_and_si128(_mm_srli_epi16(qh_vec, 6), _mm_set1_epi8(3));
					__m128i qh_shift = _mm_slli_epi16(qh_67, 4);
					__m128i q_vec = _mm_or_si128(ql_high, qh_shift);
					__m128i q_int = _mm_sub_epi8(q_vec, _mm_set1_epi8(32));
					__m256i q_32 = _mm256_cvtepi8_epi32(q_int);
					__m256 q4_vec = _mm256_cvtepi32_ps(q_32);
					__m256 scale_vec = _mm256_set1_ps((float)sc[is + 6]);
					q4_vec = _mm256_mul_ps(q4_vec, _mm256_mul_ps(d_vec, scale_vec));
					_mm256_storeu_ps(y + l + 96, q4_vec);
				}
			}
			y += 128;
			ql += 64;
			qh += 32;
			sc += 8;
		}
	}
}

__attribute__((target("avx2"))) static void dequantize_row_q4k_f32_avx2(const void *__restrict__ q_void,
									float *__restrict__ y, int k)
{
	const block_q4_k *blocks = (const block_q4_k *)q_void;
	const int nb = k / QK_K;

	// Iterate over the blocks that make up the row
	for (int i = 0; i < nb; ++i) {
		const block_q4_k *blk = &blocks[i];
		const float d = fp16_to_fp32(blk->d);
		const float dmin = fp16_to_fp32(blk->dmin);

		// Pre-calculate scales and mins for the block
		float scales[8], mins[8];
		for (int s_idx = 0; s_idx < 8; s_idx += 2) {
			uint8_t s1, m1, s2, m2;
			get_scale_min_k4(s_idx + 0, blk->scales, &s1, &m1);
			get_scale_min_k4(s_idx + 1, blk->scales, &s2, &m2);
			scales[s_idx] = d * s1;
			mins[s_idx] = dmin * m1;
			scales[s_idx + 1] = d * s2;
			mins[s_idx + 1] = dmin * m2;
		}

		const uint8_t *q = blk->qs;
		float *y_block = y + i * QK_K;
		int is = 0;

		for (int j = 0; j < QK_K; j += 64) {
			// Load scales and mins into AVX registers
			const __m256 d1_vec = _mm256_set1_ps(scales[is]);
			const __m256 m1_vec = _mm256_set1_ps(mins[is]);
			const __m256 d2_vec = _mm256_set1_ps(scales[is + 1]);
			const __m256 m2_vec = _mm256_set1_ps(mins[is + 1]);

			// Process 32 bytes (64 nibbles) at a time
			for (int offset = 0; offset < 32; offset += 8) {
				// Load 8 bytes, containing 16 4-bit nibbles
				const __m128i q_i8 = _mm_loadu_si64(q + offset);

				// --- Vectorized Unpacking ---
				const __m128i mask_lo = _mm_set1_epi8(0x0F);
				const __m128i q_lo_i8 = _mm_and_si128(q_i8, mask_lo);
				const __m128i q_hi_i8 = _mm_and_si128(_mm_srli_epi16(q_i8, 4), mask_lo);

				// Convert 8-bit nibbles to 32-bit integers, then to floats
				const __m256 qv_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_lo_i8));
				const __m256 qv_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi_i8));

				// Dequantize: y = d * q - m
				const __m256 y_lo = _mm256_sub_ps(_mm256_mul_ps(qv_lo, d1_vec), m1_vec);
				const __m256 y_hi = _mm256_sub_ps(_mm256_mul_ps(qv_hi, d2_vec), m2_vec);

				// Store results
				_mm256_storeu_ps(y_block + j + offset, y_lo);
				_mm256_storeu_ps(y_block + j + 32 + offset, y_hi);
			}
			q += 32;
			is += 2;
		}
	}
}

__attribute__((target("avx2"))) float dot_product_f32_q6k_avx2(const float *x, const block_q6_k *block)
{
	__m256 sum_vec = _mm256_setzero_ps();
	const float d = fp16_to_fp32(block->d);
	__m256 d_vec = _mm256_set1_ps(d);
	const uint8_t *ql = block->ql;
	const uint8_t *qh = block->qh;
	const int8_t *sc = block->scales;

	for (int n = 0; n < 2; n++) { // Two halves: n=0 for 0-127, n=1 for 128-255
		const float *x_half = x + n * 128;
		const uint8_t *ql_half = ql + n * 64; // ql[0:64] or ql[64:128]
		const uint8_t *qh_half = qh + n * 32; // qh[0:32] or qh[32:64]
		const int8_t *sc_half = sc + n * 8;   // sc[0:8] or sc[8:16]

		for (int l = 0; l < 32; l += 8) {
			int is = (l >= 16) ? 1 : 0; // is=0 for l=0-15, is=1 for l=16-31

			// Load 8 elements of x for q1, q2, q3, q4
			__m256 x1 = _mm256_loadu_ps(x_half + l);
			__m256 x2 = _mm256_loadu_ps(x_half + l + 32);
			__m256 x3 = _mm256_loadu_ps(x_half + l + 64);
			__m256 x4 = _mm256_loadu_ps(x_half + l + 96);

			// Load quantization data for l to l+7
			__m128i ql_vec = _mm_loadu_si128((const __m128i *)(ql_half + l));
			__m128i qh_vec = _mm_loadu_si128((const __m128i *)(qh_half + l));
			__m128i ql_vec32 = _mm_loadu_si128((const __m128i *)(ql_half + l + 32));

			// q1: (ql[l] & 0xF) | ((qh[l] & 3) << 4) - 32
			__m128i ql_low = _mm_and_si128(ql_vec, _mm_set1_epi8(0x0F));
			__m128i qh_01 = _mm_and_si128(qh_vec, _mm_set1_epi8(3));
			__m128i qh_shift1 = _mm_slli_epi16(qh_01, 4);
			__m128i q_vec1 = _mm_or_si128(ql_low, qh_shift1);
			__m128i q_int1 = _mm_sub_epi8(q_vec1, _mm_set1_epi8(32));
			__m256 q1_vec = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_int1));
			__m256 scale_vec1 = _mm256_set1_ps((float)sc_half[is]);
			__m256 w1_vec = _mm256_mul_ps(d_vec, _mm256_mul_ps(scale_vec1, q1_vec));
			sum_vec = _mm256_fmadd_ps(w1_vec, x1, sum_vec);

			// q2: (ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4) - 32
			__m128i ql_low2 = _mm_and_si128(ql_vec32, _mm_set1_epi8(0x0F));
			__m128i qh_23 = _mm_and_si128(_mm_srli_epi16(qh_vec, 2), _mm_set1_epi8(3));
			__m128i qh_shift2 = _mm_slli_epi16(qh_23, 4);
			__m128i q_vec2 = _mm_or_si128(ql_low2, qh_shift2);
			__m128i q_int2 = _mm_sub_epi8(q_vec2, _mm_set1_epi8(32));
			__m256 q2_vec = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_int2));
			__m256 scale_vec2 = _mm256_set1_ps((float)sc_half[is + 2]);
			__m256 w2_vec = _mm256_mul_ps(d_vec, _mm256_mul_ps(scale_vec2, q2_vec));
			sum_vec = _mm256_fmadd_ps(w2_vec, x2, sum_vec);

			// q3: (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4) - 32
			__m128i ql_high = _mm_and_si128(_mm_srli_epi16(ql_vec, 4), _mm_set1_epi8(0x0F));
			__m128i qh_45 = _mm_and_si128(_mm_srli_epi16(qh_vec, 4), _mm_set1_epi8(3));
			__m128i qh_shift3 = _mm_slli_epi16(qh_45, 4);
			__m128i q_vec3 = _mm_or_si128(ql_high, qh_shift3);
			__m128i q_int3 = _mm_sub_epi8(q_vec3, _mm_set1_epi8(32));
			__m256 q3_vec = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_int3));
			__m256 scale_vec3 = _mm256_set1_ps((float)sc_half[is + 4]);
			__m256 w3_vec = _mm256_mul_ps(d_vec, _mm256_mul_ps(scale_vec3, q3_vec));
			sum_vec = _mm256_fmadd_ps(w3_vec, x3, sum_vec);

			// q4: (ql[l+32] >> 4) | (((qh[l] >> 6) & 3) << 4) - 32
			__m128i ql_high2 = _mm_and_si128(_mm_srli_epi16(ql_vec32, 4), _mm_set1_epi8(0x0F));
			__m128i qh_67 = _mm_and_si128(_mm_srli_epi16(qh_vec, 6), _mm_set1_epi8(3));
			__m128i qh_shift4 = _mm_slli_epi16(qh_67, 4);
			__m128i q_vec4 = _mm_or_si128(ql_high2, qh_shift4);
			__m128i q_int4 = _mm_sub_epi8(q_vec4, _mm_set1_epi8(32));
			__m256 q4_vec = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_int4));
			__m256 scale_vec4 = _mm256_set1_ps((float)sc_half[is + 6]);
			__m256 w4_vec = _mm256_mul_ps(d_vec, _mm256_mul_ps(scale_vec4, q4_vec));
			sum_vec = _mm256_fmadd_ps(w4_vec, x4, sum_vec);
		}
	}

	// Reduce sum_vec to scalar
	float sum = hsum_f32_avx2(sum_vec);

	return sum;
}

__attribute__((target("avx2"))) void accumulate_weighted_V_f32_bf16_avx2(void *__restrict O, float weight,
									 const void *__restrict V, int size)
{
	float *out = (float *)O;
	uint16_t *v_bf16 = (uint16_t *)V;

	__m256 weight_ps = _mm256_set1_ps(weight);

	int i = 0;
	for (; i + 8 <= size; i += 8) {
		// Load 8 uint16_t BF16 values
		__m128i v_half = _mm_loadu_si128((const __m128i *)(v_bf16 + i));

		// Convert to 8 * uint32_t
		__m256i v_i32 = _mm256_cvtepu16_epi32(v_half);

		// Shift left 16 bits → convert to FP32 bit pattern
		__m256i v_bits = _mm256_slli_epi32(v_i32, 16);

		// Cast to float
		__m256 v_ps = _mm256_castsi256_ps(v_bits);

		// Load output
		__m256 out_ps = _mm256_loadu_ps(out + i);

		// Fused multiply-add
		__m256 result = _mm256_fmadd_ps(weight_ps, v_ps, out_ps);

		// Store result
		_mm256_storeu_ps(out + i, result);
	}

	// Handle remaining tail
	for (; i < size; i++) {
		out[i] = fmaf(weight, bf16_to_fp32(v_bf16[i]), out[i]);
	}
}

__attribute__((target("avx2"))) float dot_product_f32_q4k_avx2(const float *x, const block_q4_k *blk)
{
	const float d = fp16_to_fp32(blk->d);
	const float dmin = fp16_to_fp32(blk->dmin);

	// Pre-calculate scales for the entire block
	float scales[8], mins[8];
	for (int i = 0; i < 8; i += 2) {
		uint8_t s1, m1, s2, m2;
		get_scale_min_k4(i + 0, blk->scales, &s1, &m1);
		get_scale_min_k4(i + 1, blk->scales, &s2, &m2);
		scales[i] = d * s1;
		mins[i] = dmin * m1;
		scales[i + 1] = d * s2;
		mins[i + 1] = dmin * m2;
	}

	__m256 acc = _mm256_setzero_ps();
	const uint8_t *q = blk->qs;
	int is = 0;

	// Process the block of 256 values
	for (int j = 0; j < 256; j += 64) {
		// Broadcast scales and mins for the current 64-element chunk
		__m256 d1_vec = _mm256_set1_ps(scales[is]);
		__m256 m1_vec = _mm256_set1_ps(mins[is]);
		__m256 d2_vec = _mm256_set1_ps(scales[is + 1]);
		__m256 m2_vec = _mm256_set1_ps(mins[is + 1]);

		// Process 32 bytes (64 nibbles) at a time
		for (int offset = 0; offset < 32; offset += 8) {
			// Load 8 bytes, containing 16 4-bit nibbles
			__m128i q_i8 = _mm_loadu_si64(q + offset);

			// Mask for low nibbles (0x0F0F...)
			const __m128i mask_lo = _mm_set1_epi8(0x0F);
			// Get low nibbles: (q & 0x0F)
			__m128i q_lo_i8 = _mm_and_si128(q_i8, mask_lo);
			// Get high nibbles: ((q >> 4) & 0x0F)
			__m128i q_hi_i8 = _mm_and_si128(_mm_srli_epi16(q_i8, 4), mask_lo);

			// Convert 8-bit nibbles to 32-bit integers, then to floats
			__m256 qv_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_lo_i8));
			__m256 qv_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi_i8));

			// Load corresponding x vectors
			__m256 xv_lo = _mm256_loadu_ps(x + j + offset);
			__m256 xv_hi = _mm256_loadu_ps(x + j + 32 + offset);

			// Dequantize and multiply: acc += x * (d * q - m)
			acc = _mm256_fmadd_ps(xv_lo, _mm256_sub_ps(_mm256_mul_ps(qv_lo, d1_vec), m1_vec), acc);
			acc = _mm256_fmadd_ps(xv_hi, _mm256_sub_ps(_mm256_mul_ps(qv_hi, d2_vec), m2_vec), acc);
		}

		q += 32;
		is += 2;
	}

	// Reduce the 8-lane sum to a single float
	return hsum_f32_avx2(acc);
}


__attribute__((target("avx2"))) void apply_rope_cache_f32_avx2(RopeCacheType *rope_cache, void *X, int pos,
							       int head_dim)
{
	int h_dim_half = head_dim / 2;
	float *x = (float *)X;

	if (pos >= rope_cache->max_pos) {
		fprintf(stderr, "Position %d exceeds rope cache max_pos %d\n", pos, rope_cache->max_pos);
		return;
	}

	const float *sin_vals = rope_cache->sin + pos * h_dim_half;
	const float *cos_vals = rope_cache->cos + pos * h_dim_half;

	int i = 0;
	// Process 8 floats (16 total elements: 8 real, 8 imaginary) at a time
	for (; i <= h_dim_half - 8; i += 8) {
		// Load 8 "real" and 8 "imaginary" parts from the input vector x
		__m256 x_r = _mm256_loadu_ps(&x[i]);
		__m256 x_i = _mm256_loadu_ps(&x[i + h_dim_half]);

		// Load 8 sin and cos values from the cache
		__m256 sin_v = _mm256_loadu_ps(&sin_vals[i]);
		__m256 cos_v = _mm256_loadu_ps(&cos_vals[i]);

		// Calculate the two main components of the rotation
		__m256 x_r_cos = _mm256_mul_ps(x_r, cos_v);
		__m256 x_r_sin = _mm256_mul_ps(x_r, sin_v);
		__m256 x_i_cos = _mm256_mul_ps(x_i, cos_v);
		__m256 x_i_sin = _mm256_mul_ps(x_i, sin_v);

		// Perform the rotation for all 8 elements in parallel
		// new_x_real = x_real * cos - x_imag * sin
		__m256 new_x_r = _mm256_sub_ps(x_r_cos, x_i_sin);
		// new_x_imag = x_real * sin + x_imag * cos
		__m256 new_x_i = _mm256_add_ps(x_r_sin, x_i_cos);

		// Store the results back into the x vector
		_mm256_storeu_ps(&x[i], new_x_r);
		_mm256_storeu_ps(&x[i + h_dim_half], new_x_i);
	}

	// Scalar tail loop for remaining elements
	for (; i < h_dim_half; ++i) {
		float x_real = x[i];
		float x_imag = x[i + h_dim_half];
		float sin = sin_vals[i];
		float cos = cos_vals[i];

		x[i] = fmaf(x_real, cos, -x_imag * sin);
		x[i + h_dim_half] = fmaf(x_real, sin, x_imag * cos);
	}
}

__attribute__((target("avx2"))) void apply_residual_f32_f32_avx2(void *acc_void, const void *residual_void, int size)
{
	float *acc = (float *)acc_void;
	float *residual = (float *)residual_void;
	int i = 0;
	for (; i <= size - 8; i += 8) {
		// Load 8 floats from the accumulator (current hidden state)
		__m256 acc_vec = _mm256_loadu_ps(&acc[i]);
		// Load 8 floats from the residual vector
		__m256 res_vec = _mm256_loadu_ps(&residual[i]);
		// Add them together
		acc_vec = _mm256_add_ps(acc_vec, res_vec);
		// Store the result back
		_mm256_storeu_ps(&acc[i], acc_vec);
	}

	// Handle any remaining elements
	for (; i < size; ++i) {
		acc[i] += residual[i];
	}
}

__attribute__((target("avx2"))) void mat_vec_row_f32_q4k_f32_avx2(const void *X, const void *w_void, void *O,
								  int in_dim, int start_row, int end_row)
{
	const block_q4_k *w = (const block_q4_k *)w_void;
	const int nb = in_dim / QK_K;
	const float *x = (const float *)X;
	float *o = (float *)O;

	for (int i = start_row; i < end_row; i++) {
		const block_q4_k *w_row = w + (long long)i * nb;
		float sum = 0.0f;
		for (int j = 0; j < nb; j++) {
			sum += dot_product_f32_q4k_avx2(x + j * QK_K, &w_row[j]);
		}
		o[i] = sum;
	}
}

__attribute__((target("avx2"))) void mat_vec_row_f32_q6k_f32_avx2(const void *X, const void *w_void, void *O,
								  int in_dim, int start_row, int end_row)
{
	const block_q6_k *w = (const block_q6_k *)w_void;
	const int nb = in_dim / QK_K;
	const float *x = (const float *)X;
	float *o = (float *)O;

	for (int i = start_row; i < end_row; i++) {
		const block_q6_k *w_row = w + (long long)i * nb;
		float sum = 0.0f;
		for (int j = 0; j < nb; j++) {
			sum += dot_product_f32_q6k_avx2(x + j * QK_K, &w_row[j]);
		}
		o[i] = sum;
	}
}

__attribute__((target("avx2"))) void get_embedding_row_q4k_f32_avx2(const Tensor *W, int row_index, void *dest,
								    int embed_dim)
{
	int blocks_per_row = embed_dim / QK_K;
	block_q4_k *src = (block_q4_k *)W->mem.data;
	long long row_block_offset = (long long)row_index * blocks_per_row;

	for (int block_idx = 0; block_idx < blocks_per_row; block_idx++)
		dequantize_row_q4k_f32_avx2(&src[row_block_offset + block_idx], (float *)dest + block_idx * QK_K, QK_K);
}

__attribute__((target("avx2"))) void get_embedding_row_q6k_f32_avx2(const Tensor *W, int row_index, void *dest,
								    int embed_dim)
{
	int blocks_per_row = embed_dim / QK_K;

	block_q6_k *src = (block_q6_k *)W->mem.data;
	long long row_block_offset = (long long)row_index * blocks_per_row;

	for (int block_idx = 0; block_idx < blocks_per_row; block_idx++) {
		dequantize_row_q6k_f32_avx2(&src[row_block_offset + block_idx], (float *)dest + block_idx * QK_K, QK_K);
	}
}

__attribute__((target("avx2"))) void get_embedding_row_bf16_bf16_avx2(const Tensor *W, int row_index, void *dest,
								      int embed_dim)
{
	const uint16_t *src = (const uint16_t *)W->mem.data;
	long long row_offset = (long long)row_index * embed_dim;
	const uint16_t *row_src = src + row_offset;
	uint16_t *row_dst = (uint16_t *)dest;

	int i = 0;
	// Process 16 BF16 values = 32 bytes per AVX2 register
	for (; i + 16 <= embed_dim; i += 16) {
		__m256i v = _mm256_loadu_si256((const __m256i *)(row_src + i));
		_mm256_storeu_si256((__m256i *)(row_dst + i), v);
	}
	// Tail
	for (; i < embed_dim; i++) {
		row_dst[i] = row_src[i];
	}
}

__attribute__((target("avx2"))) void get_embedding_row_bf16_f32_avx2(const Tensor *W, int row_index, void *dest,
								     int embed_dim)
{
	const uint16_t *src = (const uint16_t *)W->mem.data;
	long long row_offset = (long long)row_index * (long long)embed_dim;
	const uint16_t *row_src = src + row_offset;
	float *row_dst = (float *)dest;

	int i = 0;

	// Main loop: 16 BF16 -> 16 FP32 per iteration
	for (; i + 16 <= embed_dim; i += 16) {
		// load 16 u16 as two 128-bit halves
		__m128i v128_lo = _mm_loadu_si128((const __m128i *)(row_src + i + 0)); // covers elements i..i+7
		__m128i v128_hi = _mm_loadu_si128((const __m128i *)(row_src + i + 8)); // covers elements i+8..i+15

		// expand 8x u16 -> 8x u32 lanes in 256-bit registers
		__m256i vlo32 = _mm256_cvtepu16_epi32(v128_lo); // lanes 0..7 (u32)
		__m256i vhi32 = _mm256_cvtepu16_epi32(v128_hi); // lanes 8..15 (u32)

		// shift left 16: place BF16 bits into FP32 exponent/mantissa position
		vlo32 = _mm256_slli_epi32(vlo32, 16);
		vhi32 = _mm256_slli_epi32(vhi32, 16);

		// bitcast to floats
		__m256 f_lo = _mm256_castsi256_ps(vlo32);
		__m256 f_hi = _mm256_castsi256_ps(vhi32);

		// store 8 floats each
		_mm256_storeu_ps(row_dst + i + 0, f_lo);
		_mm256_storeu_ps(row_dst + i + 8, f_hi);
	}

	// 8-wide remainder (if any)
	if (i + 8 <= embed_dim) {
		__m128i v128 = _mm_loadu_si128((const __m128i *)(row_src + i)); // 8 u16
		__m256i v32 = _mm256_cvtepu16_epi32(v128);
		v32 = _mm256_slli_epi32(v32, 16);
		__m256 f = _mm256_castsi256_ps(v32);
		_mm256_storeu_ps(row_dst + i, f);
		i += 8;
	}

	// scalar tail
	for (; i < embed_dim; ++i) {
		uint16_t bf = row_src[i];
		uint32_t bits = ((uint32_t)bf) << 16;
		float f;
		memcpy(&f, &bits, sizeof(f)); // safe bitcast
		row_dst[i] = f;
	}
}

__attribute__((target("avx2"))) void convert_f32_bf16_avx2(const void *src, void *dest, int size)
{
	const float *s = (const float *)src;
	uint16_t *d = (uint16_t *)dest;
	int i = 0;
	for (; i <= size - 8; i += 8) {
		__m256 s_vec = _mm256_loadu_ps(s + i);
		store_f32_as_bf16_avx2(d + i, s_vec);
	}
	// Scalar tail loop for remaining elements
	for (; i < size; ++i) {
		d[i] = fp32_to_bf16(s[i]);
	}
}

__attribute__((target("avx2"))) void convert_bf16_f32_avx2(const void *S, void *D, int n)
{
	int i = 0;
	uint16_t *src = (uint16_t *)S;
	float *dst = (float *)D;

	// Process 8 elements per iteration (256 bits / 32-bit lanes)
	const int V = 8;
	for (; i + V <= n; i += V) {
		// load 8 bf16 -> 8 f32 in a __m256
		__m256 v = load_bf16_as_f32(src + i);

		// store 8 floats to dst
		_mm256_storeu_ps(dst + i, v);
	}

	// tail: handle remaining elements scalar
	for (; i < n; ++i) {
		dst[i] = bf16_to_fp32(src[i]);
	}
}

__attribute__((target("avx2"))) void convert_f32_f32_avx2(const void *S, void *D, int n)
{
	int i = 0;
	float *src = (float *)S;
	float *dst = (float *)D;

	// Process 8 elements per iteration (256 bits / 32-bit lanes)
	const int V = 8;
	for (; i + V <= n; i += V) {
		__m256 s_vec = _mm256_loadu_ps(src + i);

		// store 8 floats to dst
		_mm256_storeu_ps(dst + i, s_vec);
	}

	// tail: handle remaining elements scalar
	for (; i < n; ++i) {
		dst[i] = src[i];
	}
}

__attribute__((target("avx2"))) void convert_bf16_bf16_avx2(const void *S, void *D, int size)
{
	int i = 0;
	uint16_t *src = (uint16_t *)S;
	uint16_t *dst = (uint16_t *)D;

	// process 16 bf16 = 32 bytes per iteration
	for (; i + 16 <= size; i += 16) {
		__m256i v = _mm256_loadu_si256((const __m256i *)(src + i));
		_mm256_storeu_si256((__m256i *)(dst + i), v);
	}

	// tail (not multiple of 16)
	for (; i < size; i++) {
		dst[i] = src[i];
	}
}

__attribute__((target("avx2"))) void swiglu_activation_f32_f32_avx2(void *gate_void, const void *up_void, int size)
{
	float *gate = (float *)gate_void;
	const float *up = (const float *)up_void;

	const __m256 range = _mm256_set1_ps(SILU_X_MAX - SILU_X_MIN);
	const __m256 x_min = _mm256_set1_ps(SILU_X_MIN);
	const __m256 table_size_minus_1 = _mm256_set1_ps(SILU_TABLE_SIZE - 1);

	const __m256i zero_idx = _mm256_setzero_si256();
	const __m256i max_idx = _mm256_set1_epi32(SILU_TABLE_SIZE - 2);

	int i = 0;
	for (; i <= size - 8; i += 8) {
		__m256 x_vec = _mm256_loadu_ps(gate + i);
		__m256 up_vec = _mm256_loadu_ps(up + i);

		// --- Vectorized Lookup ---
		__m256 pos_f = _mm256_div_ps(_mm256_mul_ps(_mm256_sub_ps(x_vec, x_min), table_size_minus_1), range);
		__m256i idx_i = _mm256_cvtps_epi32(pos_f);

		// --- Clamp indices to the valid range before gathering ---
		idx_i = _mm256_max_epi32(idx_i, zero_idx); // Clamp min to 0
		idx_i = _mm256_min_epi32(idx_i, max_idx);  // Clamp max to TABLE_SIZE - 2

		__m256 frac_f = _mm256_sub_ps(pos_f, _mm256_cvtepi32_ps(idx_i));

		__m256 y1 = _mm256_i32gather_ps(silu_table, idx_i, sizeof(float));
		// The index for y2 is now guaranteed to be safe (max is TABLE_SIZE - 1)
		__m256 y2 =
			_mm256_i32gather_ps(silu_table, _mm256_add_epi32(idx_i, _mm256_set1_epi32(1)), sizeof(float));

		__m256 silu_val = _mm256_fmadd_ps(_mm256_sub_ps(y2, y1), frac_f, y1);

		__m256 result = _mm256_mul_ps(silu_val, up_vec);
		_mm256_storeu_ps(gate + i, result);
	}

	// Scalar tail loop
	for (; i < size; i++) {
		gate[i] = silu_lookup(gate[i]) * up[i];
	}
}

__attribute__((target("avx2"))) void store_KV_cache_f32_bf16_avx2(struct TIEContext *ctx, int layer_idx, int start_pos,
								  int batch_len)
{
	LayerKVCache *cache = &ctx->kv_cache[layer_idx];
	int kv_dim = ctx->model->num_kv_heads * ctx->model->head_dim;
	uint16_t *k_cache_data = (uint16_t *)cache->k.data;
	uint16_t *v_cache_data = (uint16_t *)cache->v.data;
	float *K_mem_data = (float *)ctx->mem.K.data;
	float *V_mem_data = (float *)ctx->mem.V.data;

	for (int b = 0; b < batch_len; b++) {
		long long cache_offset = (long long)(start_pos + b) * kv_dim;
		long long mem_offset = (long long)b * kv_dim;
		long long i = 0;
		for (; i <= kv_dim - 8; i += 8) {
			__m256 k_vec = _mm256_load_ps(K_mem_data + mem_offset + i);
			__m256 v_vec = _mm256_load_ps(V_mem_data + mem_offset + i);
			store_f32_as_bf16_avx2(k_cache_data + cache_offset + i, k_vec);
			store_f32_as_bf16_avx2(v_cache_data + cache_offset + i, v_vec);
		}
		for (; i < kv_dim; i++) {
			k_cache_data[cache_offset + i] = fp32_to_bf16(K_mem_data[mem_offset + i]);
			v_cache_data[cache_offset + i] = fp32_to_bf16(V_mem_data[mem_offset + i]);
		}
	}
}


__attribute__((target("avx2"))) void rms_norm_bf16_f32_bf16_avx2(void *O, const void *X, const Tensor *W, int size,
								 float eps)
{
	const uint16_t *x = (const uint16_t *)X;
	uint16_t *o = (uint16_t *)O;
	const float *w = (const float *)W->mem.data;

	// 1. Vectorized sum of squares
	__m256 sum_sq_vec = _mm256_setzero_ps();
	int i = 0;
	for (; i <= size - 8; i += 8) {
		__m256 x_vec_f32 = load_bf16_as_f32(x + i);
		sum_sq_vec = _mm256_fmadd_ps(x_vec_f32, x_vec_f32, sum_sq_vec);
	}
	float ss = hsum_f32_avx2(sum_sq_vec);
	// Scalar tail for sum of squares
	for (; i < size; ++i) {
		float val = bf16_to_fp32(x[i]);
		ss += val * val;
	}

	// 2. Calculate inverse RMS
	ss /= size;
	ss += eps;
	const float inv_rms = 1.0f / sqrtf(ss);
	const __m256 inv_rms_vec = _mm256_set1_ps(inv_rms);

	// 3. Vectorized scaling and storing
	i = 0;
	for (; i <= size - 8; i += 8) {
		__m256 x_vec_f32 = load_bf16_as_f32(x + i);
		__m256 w_vec_f32 = _mm256_loadu_ps(w + i);

		__m256 result_f32 = _mm256_mul_ps(x_vec_f32, w_vec_f32);
		result_f32 = _mm256_mul_ps(result_f32, inv_rms_vec);

		store_f32_as_bf16_avx2(o + i, result_f32);
	}
	// Scalar tail for scaling
	for (; i < size; ++i) {
		float val = bf16_to_fp32(x[i]);
		o[i] = fp32_to_bf16(val * w[i] * inv_rms);
	}
}

__attribute__((target("avx2"))) void apply_residual_bf16_bf16_avx2(void *acc, const void *residual, int size)
{
	uint16_t *acc_bf16 = (uint16_t *)acc;
	const uint16_t *res_bf16 = (const uint16_t *)residual;
	int i = 0;
	for (; i <= size - 8; i += 8) {
		__m256 acc_vec_f32 = load_bf16_as_f32(acc_bf16 + i);
		__m256 res_vec_f32 = load_bf16_as_f32(res_bf16 + i);

		__m256 sum_vec_f32 = _mm256_add_ps(acc_vec_f32, res_vec_f32);

		store_f32_as_bf16_avx2(acc_bf16 + i, sum_vec_f32);
	}
	// Scalar tail loop
	for (; i < size; ++i) {
		float acc_f32 = bf16_to_fp32(acc_bf16[i]);
		float res_f32 = bf16_to_fp32(res_bf16[i]);
		acc_bf16[i] = fp32_to_bf16(acc_f32 + res_f32);
	}
}

__attribute__((target("avx2"))) void apply_rope_cache_bf16_avx2(RopeCacheType *rope_cache, void *X, int pos,
								int head_dim)
{
	int h_dim_half = head_dim / 2;
	uint16_t *x = (uint16_t *)X;

	const float *sin_vals = rope_cache->sin + pos * h_dim_half;
	const float *cos_vals = rope_cache->cos + pos * h_dim_half;

	int i = 0;
	for (; i <= h_dim_half - 8; i += 8) {
		// Load 8 BF16 values and convert to FP32 vectors
		__m256 x_r = load_bf16_as_f32(x + i);
		__m256 x_i = load_bf16_as_f32(x + i + h_dim_half);

		// Load 8 FP32 sin/cos values
		__m256 sin_v = _mm256_loadu_ps(sin_vals + i);
		__m256 cos_v = _mm256_loadu_ps(cos_vals + i);

		// Perform rotation using fused multiply-subtract and multiply-add
		// new_x_r = (x_r * cos_v) - (x_i * sin_v)
		__m256 new_x_r = _mm256_fmsub_ps(x_i, sin_v, _mm256_mul_ps(x_r, cos_v));
		// new_x_i = (x_r * sin_v) + (x_i * cos_v)
		__m256 new_x_i = _mm256_fmadd_ps(x_r, sin_v, _mm256_mul_ps(x_i, cos_v));

		// Convert results back to BF16 and store
		store_f32_as_bf16_avx2(x + i, new_x_r);
		store_f32_as_bf16_avx2(x + i + h_dim_half, new_x_i);
	}

	// Scalar tail loop for remaining elements
	for (; i < h_dim_half; ++i) {
		float x_real = bf16_to_fp32(x[i]);
		float x_imag = bf16_to_fp32(x[i + h_dim_half]);
		float sin_val = sin_vals[i];
		float cos_val = cos_vals[i];
		x[i] = fp32_to_bf16(x_real * cos_val - x_imag * sin_val);
		x[i + h_dim_half] = fp32_to_bf16(x_real * sin_val + x_imag * cos_val);
	}
}

__attribute__((target("avx2"))) void swiglu_activation_bf16_bf16_avx2(void *gate_void, const void *up_void, int size)
{
	uint16_t *gate = (uint16_t *)gate_void;
	const uint16_t *up = (const uint16_t *)up_void;

	const __m256 range = _mm256_set1_ps(SILU_X_MAX - SILU_X_MIN);
	const __m256 x_min = _mm256_set1_ps(SILU_X_MIN);
	const __m256 table_size_minus_1 = _mm256_set1_ps(SILU_TABLE_SIZE - 1);
	const __m256i zero_idx = _mm256_setzero_si256();
	const __m256i max_idx = _mm256_set1_epi32(SILU_TABLE_SIZE - 2);

	int i = 0;
	for (; i <= size - 8; i += 8) {
		// Load 8 BF16 values and convert to FP32
		__m256 gate_vec_f32 = load_bf16_as_f32(gate + i);
		__m256 up_vec_f32 = load_bf16_as_f32(up + i);

		// --- Vectorized SiLU Lookup (same logic as the FP32 version) ---
		__m256 pos_f =
			_mm256_div_ps(_mm256_mul_ps(_mm256_sub_ps(gate_vec_f32, x_min), table_size_minus_1), range);
		__m256i idx_i = _mm256_cvtps_epi32(pos_f);
		idx_i = _mm256_max_epi32(idx_i, zero_idx);
		idx_i = _mm256_min_epi32(idx_i, max_idx);
		__m256 frac_f = _mm256_sub_ps(pos_f, _mm256_cvtepi32_ps(idx_i));
		__m256 y1 = _mm256_i32gather_ps(silu_table, idx_i, sizeof(float));
		__m256 y2 =
			_mm256_i32gather_ps(silu_table, _mm256_add_epi32(idx_i, _mm256_set1_epi32(1)), sizeof(float));
		__m256 silu_val = _mm256_fmadd_ps(_mm256_sub_ps(y2, y1), frac_f, y1);

		// Final SwiGLU operation and store
		__m256 result_f32 = _mm256_mul_ps(silu_val, up_vec_f32);
		store_f32_as_bf16_avx2(gate + i, result_f32);
	}

	// Scalar tail loop
	for (; i < size; i++) {
		float gate_f32 = bf16_to_fp32(gate[i]);
		float up_f32 = bf16_to_fp32(up[i]);
		gate[i] = fp32_to_bf16(silu_lookup(gate_f32) * up_f32);
	}
}

__attribute__((target("avx2"))) void store_KV_cache_bf16_bf16_avx2(struct TIEContext *ctx, int layer_idx, int start_pos,
								   int batch_len)
{
	LayerKVCache *cache = &ctx->kv_cache[layer_idx];
	int kv_dim = ctx->model->num_kv_heads * ctx->model->head_dim;
	long long cache_offset = (long long)start_pos * kv_dim;
	long long batch_size_elements = (long long)batch_len * kv_dim;

	uint16_t *k_cache_data = (uint16_t *)cache->k.data;
	uint16_t *v_cache_data = (uint16_t *)cache->v.data;
	uint16_t *K_mem_data = (uint16_t *)ctx->mem.K.data;
	uint16_t *V_mem_data = (uint16_t *)ctx->mem.V.data;

	long long i = 0;
	// Process 16 bf16 values (256 bits) at a time
	for (; i <= batch_size_elements - 16; i += 16) {
		// Load 256 bits (16 bf16s)
		__m256i k_vec = _mm256_loadu_si256((__m256i *)(K_mem_data + i));
		__m256i v_vec = _mm256_loadu_si256((__m256i *)(V_mem_data + i));
		// Store 256 bits
		_mm256_storeu_si256((__m256i *)(k_cache_data + cache_offset + i), k_vec);
		_mm256_storeu_si256((__m256i *)(v_cache_data + cache_offset + i), v_vec);
	}
	// Scalar tail loop
	for (; i < batch_size_elements; i++) {
		k_cache_data[cache_offset + i] = K_mem_data[i];
		v_cache_data[cache_offset + i] = V_mem_data[i];
	}
}

__attribute__((target("avx2"))) float dot_product_bf16_q4k_avx2(const uint16_t *x_bf16,
                                                               const block_q4_k *block)
{
    const float d = fp16_to_fp32(block->d);
    const float dmin = fp16_to_fp32(block->dmin);
    const uint8_t *qs = block->qs;

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    // Pre-load all 8 scales + mins once (they are only 8×2 values)
    float scales[8], mins[8];
    for (int i = 0; i < 8; i += 2) {
        uint8_t s1,m1,s2,m2;
        get_scale_min_k4(i,   block->scales, &s1, &m1);
        get_scale_min_k4(i+1, block->scales, &s2, &m2);
        scales[i]   = d    * s1;  mins[i]   = dmin * m1;
        scales[i+1] = d    * s2;  mins[i+1] = dmin * m2;
    }

    // Unrolled over the two 32-byte chunks → removes branch + pointer math
    for (int chunk = 0; chunk < 2; ++chunk) {
        const int off = chunk * 32;
        const int scale_base = chunk * 4;

        #define PROCESS_8(offset) do { \
            __m128i q8 = _mm_loadu_si64(qs + off + offset); \
            __m128i lo = _mm_and_si128(q8, _mm_set1_epi8(0x0F)); \
            __m128i hi = _mm_and_si128(_mm_srli_epi16(q8, 4), _mm_set1_epi8(0x0F)); \
            __m256 q_lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)); \
            __m256 q_hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)); \
            __m256 x_lo = load_bf16_as_f32(x_bf16 + off + offset); \
            __m256 x_hi = load_bf16_as_f32(x_bf16 + off + 32 + offset); \
            __m256 w_lo = _mm256_sub_ps(_mm256_mul_ps(q_lo, _mm256_set1_ps(scales[scale_base + (offset/8)*2])), \
                                        _mm256_set1_ps(mins[scale_base + (offset/8)*2])); \
            __m256 w_hi = _mm256_sub_ps(_mm256_mul_ps(q_hi, _mm256_set1_ps(scales[scale_base + (offset/8)*2 + 1])), \
                                        _mm256_set1_ps(mins[scale_base + (offset/8)*2 + 1])); \
            acc0 = _mm256_fmadd_ps(x_lo, w_lo, acc0); \
            acc1 = _mm256_fmadd_ps(x_hi, w_hi, acc1); \
        } while(0)

        PROCESS_8(0);
        PROCESS_8(8);
        PROCESS_8(16);
        PROCESS_8(24);

        #undef PROCESS_8
    }

    return hsum_f32_avx2(_mm256_add_ps(acc0, acc1));
}

__attribute__((target("avx2"))) float dot_product_bf16_q6k_avx2(const uint16_t *x_bf16,
                                                               const block_q6_k *block)
{
    const float d = fp16_to_fp32(block->d);
    __m256 acc = _mm256_setzero_ps();
    __m256 dvec = _mm256_set1_ps(d);

    const uint8_t *ql = block->ql;
    const uint8_t *qh = block->qh;
    const int8_t  *sc = block->scales;

    // Fully unrolled 4×8 loop – removes all inner branches
    #define Q6K_BLOCK(l, xoff, qloff, scaleoff) do { \
        __m256 xv = load_bf16_as_f32(x_bf16 + (l) + (xoff)); \
        __m128i qlv = _mm_loadu_si128((const __m128i*)(ql + (qloff) + (l))); \
        __m128i qhv = _mm_loadu_si128((const __m128i*)(qh + (qloff)/2 + (l))); \
        __m128i qv = _mm_or_si128(_mm_and_si128(qlv, _mm_set1_epi8(0x0F)), \
                                  _mm_slli_epi16(_mm_and_si128(qhv, _mm_set1_epi8(3)), 4)); \
        if ((l) >= 16) qv = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(qlv,4), _mm_set1_epi8(0x0F)), \
                                         _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qhv, (scaleoff)*2), _mm_set1_epi8(3)), 4)); \
        __m256 qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_sub_epi8(qv, _mm_set1_epi8(32)))); \
        __m256 wf = _mm256_mul_ps(dvec, _mm256_mul_ps(_mm256_set1_ps((float)sc[(scaleoff)]), qf)); \
        acc = _mm256_fmadd_ps(wf, xv, acc); \
    } while(0)

    for (int base = 0; base < 128; base += 64) {  // two 128-element halves
        int qbase = base;
        int xbase = base;
        Q6K_BLOCK(0,  xbase,      qbase,       0);
        Q6K_BLOCK(8,  xbase,      qbase,       0);
        Q6K_BLOCK(32, xbase + 32, qbase + 32,   2);
        Q6K_BLOCK(40, xbase + 32, qbase + 32,   2);
        Q6K_BLOCK(64, xbase + 64, qbase,       4);
        Q6K_BLOCK(72, xbase + 64, qbase,       4);
        Q6K_BLOCK(96, xbase + 96, qbase + 32,   6);
        Q6K_BLOCK(104,xbase + 96, qbase + 32,   6);
    }
    #undef Q6K_BLOCK

    return hsum_f32_avx2(acc);
}


__attribute__((target("avx2"))) void mat_vec_row_bf16_q4k_f32_avx2(const void *X, const void *w_void, void *O,
								   int in_dim, int start_row, int end_row)
{
	const block_q4_k *w = (const block_q4_k *)w_void;
	const int nb = in_dim / QK_K;
	const uint16_t *x = (const uint16_t *)X;
	float *o = (float *)O;

	for (int i = start_row; i < end_row; i++) {
		const block_q4_k *w_row = w + (long long)i * nb;
		float sum = 0.0f;
		for (int j = 0; j < nb; j++) {
			sum += dot_product_bf16_q4k_avx2(x + j * QK_K, &w_row[j]);
		}
		o[i] = sum;
	}
}

__attribute__((target("avx2"))) void mat_vec_row_bf16_q4k_bf16_avx2(const void *X, const void *w_void, void *O,
								    int in_dim, int start_row, int end_row)
{
	const block_q4_k *w = (const block_q4_k *)w_void;
	const int nb = in_dim / QK_K;
	const uint16_t *x = (const uint16_t *)X;
	uint16_t *o = (uint16_t *)O;

	for (int i = start_row; i < end_row; i++) {
		const block_q4_k *w_row = w + (long long)i * nb;

		// Accumulate in FP32
		float sum = 0.0f;

		for (int j = 0; j < nb; j++) {
			sum += dot_product_bf16_q4k_avx2(x + j * QK_K, &w_row[j]);
		}

		// Convert final FP32 result to BF16 with RNE and store
		o[i] = fp32_to_bf16_rne(sum);
	}
}

__attribute__((target("avx2"))) void mat_vec_row_bf16_q6k_bf16_avx2(const void *X, const void *w_void, void *O,
								    int in_dim, int start_row, int end_row)
{
	const block_q6_k *w = (const block_q6_k *)w_void;
	const int nb = in_dim / QK_K;
	const uint16_t *x = (const uint16_t *)X;
	uint16_t *o = (uint16_t *)O;

	for (int i = start_row; i < end_row; i++) {
		const block_q6_k *w_row = w + (long long)i * nb;

		// Accumulate in FP32
		float sum = 0.0f;

		for (int j = 0; j < nb; j++) {
			sum += dot_product_bf16_q6k_avx2(x + j * QK_K, &w_row[j]);
		}

		// Convert final FP32 result to BF16 with RNE and store
		o[i] = fp32_to_bf16_rne(sum);
	}
}

__attribute__((target("avx2"))) void mat_vec_row_bf16_q6k_f32_avx2(const void *X, const void *w_void, void *O,
								   int in_dim, int start_row, int end_row)
{
	const block_q6_k *w = (const block_q6_k *)w_void;
	const int nb = in_dim / QK_K;
	const uint16_t *x = (const uint16_t *)X;
	float *o = (float *)O;

	for (int i = start_row; i < end_row; i++) {
		const block_q6_k *w_row = w + (long long)i * nb;

		// Accumulate in FP32
		float sum = 0.0f;

		for (int j = 0; j < nb; j++) {
			sum += dot_product_bf16_q6k_avx2(x + j * QK_K, &w_row[j]);
		}

		o[i] = sum;
	}
}

__attribute__((target("avx2"))) void accumulate_weighted_V_bf16_bf16_avx2(void *__restrict O, float weight,
									  const void *__restrict V, int size)
{
	uint16_t *out_bf16 = (uint16_t *)O;
	const uint16_t *v_bf16 = (const uint16_t *)V;

	const __m256 weight_ps = _mm256_set1_ps(weight);

	int i = 0;

	// Process 16 elements per full iteration (two 8-lane chunks)
	for (; i + 16 <= size; i += 16) {
		// --- First 8 ---
		__m256 v0_ps = load_bf16_as_f32(v_bf16 + i + 0);
		__m256 out0_ps = load_bf16_as_f32(out_bf16 + i + 0);
		__m256 res0_ps = _mm256_fmadd_ps(weight_ps, v0_ps, out0_ps);
		store_f32_as_bf16_avx2(out_bf16 + i + 0, res0_ps);

		// --- Second 8 ---
		__m256 v1_ps = load_bf16_as_f32(v_bf16 + i + 8);
		__m256 out1_ps = load_bf16_as_f32(out_bf16 + i + 8);
		__m256 res1_ps = _mm256_fmadd_ps(weight_ps, v1_ps, out1_ps);
		store_f32_as_bf16_avx2(out_bf16 + i + 8, res1_ps);
	}

	// Handle 8-element remainder (optional fast path)
	if (i + 8 <= size) {
		__m256 v_ps = load_bf16_as_f32(v_bf16 + i);
		__m256 out_ps = load_bf16_as_f32(out_bf16 + i);
		__m256 res_ps = _mm256_fmadd_ps(weight_ps, v_ps, out_ps);
		store_f32_as_bf16_avx2(out_bf16 + i, res_ps);
		i += 8;
	}

	// Scalar tail
	for (; i < size; ++i) {
		float v = bf16_to_fp32(v_bf16[i]);
		float o = bf16_to_fp32(out_bf16[i]);
		float r = fmaf(weight, v, o);
		out_bf16[i] = fp32_to_bf16_rne(r);
	}
}

__attribute__((target("avx2"))) static void dequantize_row_q6k_bf16_avx2(const void *__restrict__ q_void,
									 uint16_t *__restrict__ y_bf16, int k)
{
	const block_q6_k *x = (const block_q6_k *)q_void;
	const int64_t nb = k / QK_K;

	for (int i = 0; i < nb; i++) {
		const float d = fp16_to_fp32(x[i].d);
		const uint8_t *ql = x[i].ql;
		const uint8_t *qh = x[i].qh;
		const int8_t *sc = x[i].scales;

		const __m256 d_vec = _mm256_set1_ps(d);

		// process 128 elements per inner outer-loop iteration as in your FP32 code
		for (int n = 0; n < QK_K; n += 128) {
			for (int l = 0; l < 32; l += 8) {
				int is = l / 16;

				// --- q1: y[l + 0 .. l+7] ---
				{
					__m128i ql_vec = _mm_loadu_si128((const __m128i *)(ql + l));
					__m128i qh_vec = _mm_loadu_si128((const __m128i *)(qh + l));
					__m128i ql_low = _mm_and_si128(ql_vec, _mm_set1_epi8(0x0F));
					__m128i qh_01 = _mm_and_si128(qh_vec, _mm_set1_epi8(3));
					__m128i qh_shift = _mm_slli_epi16(qh_01, 4);
					__m128i q_vec = _mm_or_si128(ql_low, qh_shift);
					__m128i q_int = _mm_sub_epi8(q_vec, _mm_set1_epi8(32));
					__m256i q_32 = _mm256_cvtepi8_epi32(q_int);
					__m256 q1_vec = _mm256_cvtepi32_ps(q_32);
					__m256 scale_vec = _mm256_set1_ps((float)sc[is]);
					q1_vec = _mm256_mul_ps(q1_vec, _mm256_mul_ps(d_vec, scale_vec));
					// store 8 floats as 8 bf16 at y_bf16 + l
					store_f32_as_bf16_avx2(y_bf16 + l, q1_vec);
				}

				// --- q2: y[l + 32 .. l+39] ---
				{
					__m128i ql_vec = _mm_loadu_si128((const __m128i *)(ql + l + 32));
					__m128i qh_vec = _mm_loadu_si128((const __m128i *)(qh + l));
					__m128i ql_low = _mm_and_si128(ql_vec, _mm_set1_epi8(0x0F));
					__m128i qh_23 = _mm_and_si128(_mm_srli_epi16(qh_vec, 2), _mm_set1_epi8(3));
					__m128i qh_shift = _mm_slli_epi16(qh_23, 4);
					__m128i q_vec = _mm_or_si128(ql_low, qh_shift);
					__m128i q_int = _mm_sub_epi8(q_vec, _mm_set1_epi8(32));
					__m256i q_32 = _mm256_cvtepi8_epi32(q_int);
					__m256 q2_vec = _mm256_cvtepi32_ps(q_32);
					__m256 scale_vec = _mm256_set1_ps((float)sc[is + 2]);
					q2_vec = _mm256_mul_ps(q2_vec, _mm256_mul_ps(d_vec, scale_vec));
					store_f32_as_bf16_avx2(y_bf16 + l + 32, q2_vec);
				}

				// --- q3: y[l + 64 .. l+71] ---
				{
					__m128i ql_vec = _mm_loadu_si128((const __m128i *)(ql + l));
					__m128i qh_vec = _mm_loadu_si128((const __m128i *)(qh + l));
					__m128i ql_high = _mm_and_si128(_mm_srli_epi16(ql_vec, 4), _mm_set1_epi8(0x0F));
					__m128i qh_45 = _mm_and_si128(_mm_srli_epi16(qh_vec, 4), _mm_set1_epi8(3));
					__m128i qh_shift = _mm_slli_epi16(qh_45, 4);
					__m128i q_vec = _mm_or_si128(ql_high, qh_shift);
					__m128i q_int = _mm_sub_epi8(q_vec, _mm_set1_epi8(32));
					__m256i q_32 = _mm256_cvtepi8_epi32(q_int);
					__m256 q3_vec = _mm256_cvtepi32_ps(q_32);
					__m256 scale_vec = _mm256_set1_ps((float)sc[is + 4]);
					q3_vec = _mm256_mul_ps(q3_vec, _mm256_mul_ps(d_vec, scale_vec));
					store_f32_as_bf16_avx2(y_bf16 + l + 64, q3_vec);
				}

				// --- q4: y[l + 96 .. l+103] ---
				{
					__m128i ql_vec = _mm_loadu_si128((const __m128i *)(ql + l + 32));
					__m128i qh_vec = _mm_loadu_si128((const __m128i *)(qh + l));
					__m128i ql_high = _mm_and_si128(_mm_srli_epi16(ql_vec, 4), _mm_set1_epi8(0x0F));
					__m128i qh_67 = _mm_and_si128(_mm_srli_epi16(qh_vec, 6), _mm_set1_epi8(3));
					__m128i qh_shift = _mm_slli_epi16(qh_67, 4);
					__m128i q_vec = _mm_or_si128(ql_high, qh_shift);
					__m128i q_int = _mm_sub_epi8(q_vec, _mm_set1_epi8(32));
					__m256i q_32 = _mm256_cvtepi8_epi32(q_int);
					__m256 q4_vec = _mm256_cvtepi32_ps(q_32);
					__m256 scale_vec = _mm256_set1_ps((float)sc[is + 6]);
					q4_vec = _mm256_mul_ps(q4_vec, _mm256_mul_ps(d_vec, scale_vec));
					store_f32_as_bf16_avx2(y_bf16 + l + 96, q4_vec);
				}
			}

			// advance pointers exactly as FP32 version expects:
			y_bf16 += 128; // 128 elements produced (now BF16 elements)
			ql += 64;
			qh += 32;
			sc += 8;
		}
	}
}

__attribute__((target("avx2"))) void get_embedding_row_q6k_bf16_avx2(const Tensor *W, int row_index, void *dest,
								     int embed_dim)
{
	int blocks_per_row = embed_dim / QK_K;

	block_q6_k *src = (block_q6_k *)W->mem.data;
	long long row_block_offset = (long long)row_index * blocks_per_row;

	for (int block_idx = 0; block_idx < blocks_per_row; block_idx++) {
		dequantize_row_q6k_bf16_avx2(&src[row_block_offset + block_idx], (uint16_t *)dest + block_idx * QK_K,
					     QK_K);
	}
}

inline float gelu_fast(float x)
{
    const float a = 0.7978845608028654f;   // sqrt(2/pi)
    const float b = 0.044715f;
    float t = tanhf(a * x * (1.0f + b * x * x));
    return 0.5f * x * (1.0f + t);
}

void geglu_activation_f32_f32_avx2(void *gate, const void *up, int size)
{
	float *gate_fp32 = (float *)gate;
	const float *up_fp32 = (const float *)up;

	const int vec_stride = 8; // 8 floats per __m256
	int i = 0;

	// constants used in formula
	const float sqrt_2_over_pi_f = 0.7978845608028654f; // sqrt(2/pi)
	const float gelu_bias_f = 0.044715f;

	const __m256 v_half = _mm256_set1_ps(0.5f);
	const __m256 v_one = _mm256_set1_ps(1.0f);
	const __m256 v_s2opi = _mm256_set1_ps(sqrt_2_over_pi_f);
	const __m256 v_b = _mm256_set1_ps(gelu_bias_f);

	// constants for tanh rational approx: (t*(t^2 + 27))/(9*t^2 + 27)
	const __m256 v_27 = _mm256_set1_ps(27.0f);
	const __m256 v_9 = _mm256_set1_ps(9.0f);

	// clamp range for t to avoid extreme values (optional but stabilizes approx)
	const __m256 v_t_max = _mm256_set1_ps(4.0f);
	const __m256 v_t_min = _mm256_set1_ps(-4.0f);

	for (; i + vec_stride - 1 < size; i += vec_stride) {
		// load gate and up
		__m256 x = _mm256_loadu_ps(gate_fp32 + i); // gate
		__m256 u = _mm256_loadu_ps(up_fp32 + i);   // up

		// x3 = x * x * x
		__m256 x2 = _mm256_mul_ps(x, x);
		__m256 x3 = _mm256_mul_ps(x2, x);

		// inner = sqrt(2/pi) * (x + 0.044715 * x^3)
		__m256 inner = _mm256_fmadd_ps(v_b, x3, x); // x + b * x3
		inner = _mm256_mul_ps(v_s2opi, inner);	    // s2opi * ( ... )

		// clamp inner to [-10, 10] to stabilize rational tanh approx
		inner = _mm256_min_ps(v_t_max, _mm256_max_ps(v_t_min, inner));

		// tanh_approx = inner * (inner*inner + 27) / (9*inner*inner + 27)
		__m256 t2 = _mm256_mul_ps(inner, inner); // inner^2
		__m256 num = _mm256_add_ps(t2, v_27);	 // t^2 + 27
		num = _mm256_mul_ps(inner, num);	 // inner * (t^2 + 27)

		__m256 den = _mm256_mul_ps(v_9, t2); // 9 * t^2
		den = _mm256_add_ps(den, v_27);	     // 9*t^2 + 27

		__m256 tanh_approx = _mm256_div_ps(num, den); // approx tanh(inner)

		// gelu = 0.5 * x * (1 + tanh_approx)
		__m256 one_plus_t = _mm256_add_ps(v_one, tanh_approx);
		__m256 gelu = _mm256_mul_ps(v_half, _mm256_mul_ps(x, one_plus_t));

		// result = gelu * u
		__m256 res = _mm256_mul_ps(gelu, u);

		// store result back into gate buffer (matches scalar semantics)
		_mm256_storeu_ps(gate_fp32 + i, res);
	}

	// tail: process remaining elements with scalar fallback
	for (; i < size; ++i) {
		//		gate_fp32[i] = gelu_stable(gate_fp32[i]) * up_fp32[i];
		gate_fp32[i] = gelu_fast(gate_fp32[i]) * up_fp32[i];
	}
}

#endif
