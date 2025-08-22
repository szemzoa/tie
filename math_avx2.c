#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#include "engine.h"
#include "math_avx2.h"
#include "math_scalar.h"


#ifdef CONFIG_ENABLE_AVX2
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
	// Reinterpret the 256-bit float vector as a 256-bit integer vector
	__m256i val_i32 = _mm256_castps_si256(src_ps);

	// Right-shift each 32-bit integer by 16 bits, effectively dropping the mantissa
	// This is the core of the FP32 -> BF16 conversion
	__m256i shifted = _mm256_srli_epi32(val_i32, 16);

	// The result is now 8x 32-bit integers, where the lower 16 bits are the BF16 value.
	// We need to pack these into 8x 16-bit integers.
	// A permutation is used to re-order the elements for packing.
	__m256i perm_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
	__m256i permuted = _mm256_permutevar8x32_epi32(shifted, perm_mask);

	// Extract the lower 128 bits, which now contain all 8 packed BF16 values
	__m128i packed_bf16 = _mm256_extracti128_si256(permuted, 0);

	// Store the 8 packed BF16 values (128 bits) to the destination
	_mm_storeu_si128((__m128i *)dst, packed_bf16);
}

__attribute__((target("avx2"))) void rms_norm_f32_f32_avx2(void *__restrict O, const void *__restrict X,
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
	__m128 low = _mm256_castps256_ps128(sum);
	__m128 high = _mm256_extractf128_ps(sum, 1);
	__m128 sum128 = _mm_add_ps(low, high);
	sum128 = _mm_hadd_ps(sum128, sum128);
	sum128 = _mm_hadd_ps(sum128, sum128);
	float ss = _mm_cvtss_f32(sum128);

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

__attribute__((target("avx2"))) void rms_norm_bf16_f32_avx2(void *__restrict o_void, const void *__restrict x_void,
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
	__m128 sum_low = _mm256_castps256_ps128(sum_sq_vec);
	__m128 sum_high = _mm256_extractf128_ps(sum_sq_vec, 1);
	__m128 sum_4 = _mm_add_ps(sum_low, sum_high);
	__m128 sum_2 = _mm_hadd_ps(sum_4, sum_4);
	__m128 sum_1 = _mm_hadd_ps(sum_2, sum_2);
	float ss = _mm_cvtss_f32(sum_1);

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

__attribute__((target("avx2"))) void mat_vec_row_bf16_f32_avx2(const void *__restrict X, const void *__restrict w_void,
							       void *__restrict O, int in_dim, int start_row,
							       int end_row)
{
	float *x = (float *)X;
	uint16_t *w_bf16 = (uint16_t *)w_void;
	float *o = (float *)O;

	for (int i = start_row; i < end_row; i++) {
		const uint16_t *w_row = &w_bf16[i * in_dim];

		__m256 acc = _mm256_setzero_ps();
		int j = 0;
		for (; j <= in_dim - 8; j += 8) {
			__m256 w = load_bf16_as_f32(&w_row[j]);
			__m256 x_vec = _mm256_loadu_ps(&x[j]);
			acc = _mm256_fmadd_ps(x_vec, w, acc);
		}

		// Horizontally add 8 floats
		__m128 low = _mm256_castps256_ps128(acc);
		__m128 high = _mm256_extractf128_ps(acc, 1);
		__m128 sum128 = _mm_add_ps(low, high);
		sum128 = _mm_hadd_ps(sum128, sum128);
		sum128 = _mm_hadd_ps(sum128, sum128);
		float sum = _mm_cvtss_f32(sum128);

		// Tail
		for (; j < in_dim; j++) {
			uint32_t w_bits = ((uint32_t)w_row[j]) << 16;
			float w = *((float *)&w_bits);
			sum = fmaf(x[j], w, sum);
		}

		o[i] = sum;
	}
}

//__attribute__((target("avx2"))) float dot_product_f32_f32_avx2(const float *__restrict a, const float *__restrict b,
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
	__m128 sum_low = _mm256_castps256_ps128(sum_vec);    // Lower 4 lanes
	__m128 sum_high = _mm256_extractf128_ps(sum_vec, 1); // Upper 4 lanes
	__m128 sum_4 = _mm_add_ps(sum_low, sum_high);	     // Sum lower + upper
	__m128 sum_2 = _mm_hadd_ps(sum_4, sum_4);	     // Horizontal add: [s0+s1, s2+s3, ...]
	__m128 sum_1 = _mm_hadd_ps(sum_2, sum_2);	     // Final sum: [s0+s1+s2+s3, ...]
	float sum = _mm_cvtss_f32(sum_1);		     // Extract single float

	// Handle remaining elements (size % 8)
	for (; i < size; i++) {
		sum = fmaf(a[i], b[i], sum);
	}

	return sum;
}

//__attribute__((target("avx2"))) float dot_product_f32_bf16_avx2(const float *__restrict a, const uint16_t *__restrict
//b,
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

	__m128 sum_low = _mm256_castps256_ps128(acc);
	__m128 sum_high = _mm256_extractf128_ps(acc, 1);
	__m128 sum_4 = _mm_add_ps(sum_low, sum_high);
	__m128 sum_2 = _mm_hadd_ps(sum_4, sum_4);
	__m128 sum_1 = _mm_hadd_ps(sum_2, sum_2);
	float result = _mm_cvtss_f32(sum_1);

	// Tail loop
	for (; i < size; i++) {
		float b_val = bf16_to_fp32(b[i]);
		result = fmaf(a[i], b_val, result);
	}

	return result;
}

__attribute__((target("avx2"))) void mat_vec_row_f32_f32_avx2(const void *X, const void *w_void, void *O, int in_dim,
							      int start_row, int end_row)
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
		__m128 low = _mm256_castps256_ps128(acc);
		__m128 high = _mm256_extractf128_ps(acc, 1);
		__m128 sum128 = _mm_add_ps(low, high);
		sum128 = _mm_hadd_ps(sum128, sum128);
		sum128 = _mm_hadd_ps(sum128, sum128);
		float sum = _mm_cvtss_f32(sum128);

		// Tail
		for (; j < in_dim; j++) {
			sum = fmaf(x[j], w_row[j], sum);
		}

		o[i] = sum;
	}
}

__attribute__((target("avx2"))) void dequantize_row_q6_k_avx2(const void *__restrict__ q_void, float *__restrict__ y,
							      int k)
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

__attribute__((target("avx2"))) void dequantize_row_q4_k_avx2(const void *__restrict__ q_void, float *__restrict__ y,
							      int k)
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

__attribute__((target("avx2"))) float dot_product_q6_k_avx2(const float *x, const block_q6_k *block)
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
	__m128 sum_low = _mm256_castps256_ps128(sum_vec);
	__m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
	__m128 sum_4 = _mm_add_ps(sum_low, sum_high);
	__m128 sum_2 = _mm_hadd_ps(sum_4, sum_4);
	__m128 sum_1 = _mm_hadd_ps(sum_2, sum_2);
	float sum = _mm_cvtss_f32(sum_1);

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

__attribute__((target("avx2"))) float dot_product_q4_k_avx2(const float *x, const block_q4_k *blk)
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

			// OPTIMIZED UNPACKING
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

	// OPTIMIZED HORIZONTAL SUM
	// Reduce the 8-lane sum to a single float
	__m128 sum_low = _mm256_castps256_ps128(acc);
	__m128 sum_high = _mm256_extractf128_ps(acc, 1);
	__m128 sum_4 = _mm_add_ps(sum_low, sum_high);
	__m128 sum_2 = _mm_hadd_ps(sum_4, sum_4);
	__m128 sum_1 = _mm_hadd_ps(sum_2, sum_2);

	return _mm_cvtss_f32(sum_1);
}

__attribute__((target("avx2"))) void apply_rope_cache_f32_avx2(struct ctx_t *ctx, void *X, int pos, int head_dim)
{
	rope_cache_t *rope_cache = ctx->rope_cache;
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

__attribute__((target("avx2"))) void apply_residual_f32_avx2(void *acc_void, const void *residual_void, int size)
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

void mat_vec_row_q4_k_f32_avx2(const void *X, const void *w_void, void *O, int in_dim, int start_row, int end_row)
{
	const block_q4_k *w = (const block_q4_k *)w_void;
	const int nb = in_dim / QK_K;
	const float *x = (const float *)X;
	float *o = (float *)O;

	for (int i = start_row; i < end_row; i++) {
		const block_q4_k *w_row = w + (long long)i * nb;
		float sum = 0.0f;
		for (int j = 0; j < nb; j++) {
			sum += dot_product_q4_k_avx2(x + j * QK_K, &w_row[j]);
		}
		o[i] = sum;
	}
}

void mat_vec_row_q6_k_f32_avx2(const void *X, const void *w_void, void *O, int in_dim, int start_row, int end_row)
{
	const block_q6_k *w = (const block_q6_k *)w_void;
	const int nb = in_dim / QK_K;
	const float *x = (const float *)X;
	float *o = (float *)O;

	for (int i = start_row; i < end_row; i++) {
		const block_q6_k *w_row = w + (long long)i * nb;
		float sum = 0.0f;
		for (int j = 0; j < nb; j++) {
			sum += dot_product_q6_k_avx2(x + j * QK_K, &w_row[j]);
		}
		o[i] = sum;
	}
}

void get_embedding_row_q4_k_f32_avx2(const Tensor *W, int row_index, void *dest, int embed_dim)
{
	int blocks_per_row = embed_dim / QK_K;
	block_q4_k *src = (block_q4_k *)W->mem.data;
	long long row_block_offset = (long long)row_index * blocks_per_row;

	for (int block_idx = 0; block_idx < blocks_per_row; block_idx++)
		dequantize_row_q4_k_avx2(&src[row_block_offset + block_idx], (float *)dest + block_idx * QK_K, QK_K);
}

void get_embedding_row_q6_k_f32_avx2(const Tensor *W, int row_index, void *dest, int embed_dim)
{
	int blocks_per_row = embed_dim / QK_K;

	block_q6_k *src = (block_q6_k *)W->mem.data;
	long long row_block_offset = (long long)row_index * blocks_per_row;

	for (int block_idx = 0; block_idx < blocks_per_row; block_idx++) {
		dequantize_row_q6_k_avx2(&src[row_block_offset + block_idx], (float *)dest + block_idx * QK_K, QK_K);
	}
}

void convert_f32_bf16_avx2(const void *src, void *dest, int size)
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

#endif
