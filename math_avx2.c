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
	// Two accumulators to hide FMA latency
	__m256 acc0 = _mm256_setzero_ps();
	__m256 acc1 = _mm256_setzero_ps();

	const float d = fp16_to_fp32(block->d);
	__m256 d_vec = _mm256_set1_ps(d);

	const uint8_t *ql = block->ql;
	const uint8_t *qh = block->qh;
	const int8_t *sc = block->scales;

	// Loop over the two halves (0..127 and 128..255)
	for (int n = 0; n < 2; n++) {
		const float *x_ptr = x + n * 128;
		const uint8_t *ql_ptr = ql + n * 64;
		const uint8_t *qh_ptr = qh + n * 32;
		const int8_t *sc_ptr = sc + n * 8;

		// Process 32 elements (4 scales) per iteration
		// We unroll the inner logic slightly to group loads
		for (int l = 0; l < 32; l += 8) {
			int is = (l >= 16) ? 1 : 0; // Scale index offset

			// --- Load Scales ---
			// We need scales for 4 groups: [is+0], [is+2], [is+4], [is+6]
			__m256 sc0 = _mm256_set1_ps((float)sc_ptr[is + 0]);
			__m256 sc1 = _mm256_set1_ps((float)sc_ptr[is + 2]);
			__m256 sc2 = _mm256_set1_ps((float)sc_ptr[is + 4]);
			__m256 sc3 = _mm256_set1_ps((float)sc_ptr[is + 6]);

			__m256 w0 = _mm256_mul_ps(d_vec, sc0);
			__m256 w1 = _mm256_mul_ps(d_vec, sc1);
			__m256 w2 = _mm256_mul_ps(d_vec, sc2);
			__m256 w3 = _mm256_mul_ps(d_vec, sc3);

			// --- Load X ---
			__m256 x0 = _mm256_loadu_ps(x_ptr + l + 0);
			__m256 x1 = _mm256_loadu_ps(x_ptr + l + 32);
			__m256 x2 = _mm256_loadu_ps(x_ptr + l + 64);
			__m256 x3 = _mm256_loadu_ps(x_ptr + l + 96);

			// --- Load Quants ---
			__m128i ql_v = _mm_loadu_si128((const __m128i *)(ql_ptr + l));
			__m128i qh_v = _mm_loadu_si128((const __m128i *)(qh_ptr + l));
			__m128i ql_v32 = _mm_loadu_si128((const __m128i *)(ql_ptr + l + 32));

			// --- Decode q1 ---
			__m128i ql_low = _mm_and_si128(ql_v, _mm_set1_epi8(0x0F));
			__m128i qh_01 = _mm_and_si128(qh_v, _mm_set1_epi8(3));
			__m128i q1_i8 = _mm_sub_epi8(_mm_or_si128(ql_low, _mm_slli_epi16(qh_01, 4)), _mm_set1_epi8(32));
			__m256 q1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1_i8));

			// --- Decode q2 ---
			__m128i ql_low2 = _mm_and_si128(ql_v32, _mm_set1_epi8(0x0F));
			__m128i qh_23 = _mm_and_si128(_mm_srli_epi16(qh_v, 2), _mm_set1_epi8(3));
			__m128i q2_i8 =
				_mm_sub_epi8(_mm_or_si128(ql_low2, _mm_slli_epi16(qh_23, 4)), _mm_set1_epi8(32));
			__m256 q2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q2_i8));

			// --- Decode q3 ---
			__m128i ql_high = _mm_and_si128(_mm_srli_epi16(ql_v, 4), _mm_set1_epi8(0x0F));
			__m128i qh_45 = _mm_and_si128(_mm_srli_epi16(qh_v, 4), _mm_set1_epi8(3));
			__m128i q3_i8 =
				_mm_sub_epi8(_mm_or_si128(ql_high, _mm_slli_epi16(qh_45, 4)), _mm_set1_epi8(32));
			__m256 q3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q3_i8));

			// --- Decode q4 ---
			__m128i ql_high2 = _mm_and_si128(_mm_srli_epi16(ql_v32, 4), _mm_set1_epi8(0x0F));
			__m128i qh_67 = _mm_and_si128(_mm_srli_epi16(qh_v, 6), _mm_set1_epi8(3));
			__m128i q4_i8 =
				_mm_sub_epi8(_mm_or_si128(ql_high2, _mm_slli_epi16(qh_67, 4)), _mm_set1_epi8(32));
			__m256 q4 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q4_i8));

			// --- Accumulate (Pipelined) ---
			acc0 = _mm256_fmadd_ps(w0, _mm256_mul_ps(q1, x0), acc0);
			acc1 = _mm256_fmadd_ps(w1, _mm256_mul_ps(q2, x1), acc1);
			acc0 = _mm256_fmadd_ps(w2, _mm256_mul_ps(q3, x2), acc0);
			acc1 = _mm256_fmadd_ps(w3, _mm256_mul_ps(q4, x3), acc1);
		}
	}

	return hsum_f32_avx2(_mm256_add_ps(acc0, acc1));
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

	// Pre-calculate scales/mins
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

	// Two accumulators to break dependency chains
	__m256 acc0 = _mm256_setzero_ps();
	__m256 acc1 = _mm256_setzero_ps();

	const uint8_t *q = blk->qs;
	int is = 0;

	// Process 256 weights.
	// We will process 128 (2 superblocks) inside the loop to unroll.
	for (int j = 0; j < 256; j += 64) {
		__m256 d1_vec = _mm256_set1_ps(scales[is]);
		__m256 m1_vec = _mm256_set1_ps(mins[is]);
		__m256 d2_vec = _mm256_set1_ps(scales[is + 1]);
		__m256 m2_vec = _mm256_set1_ps(mins[is + 1]);

		// Inner loop: Process 32 bytes (64 weights)
		// We unroll this to do two 16-byte chunks per iteration
		for (int offset = 0; offset < 32; offset += 16) {
			// Chunk 1 (0..15 bytes)
			__m128i q_i8_0 = _mm_loadu_si64(q + offset);
			const __m128i mask_lo = _mm_set1_epi8(0x0F);

			__m256 qv_lo0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_and_si128(q_i8_0, mask_lo)));
			__m256 qv_hi0 = _mm256_cvtepi32_ps(
				_mm256_cvtepi8_epi32(_mm_and_si128(_mm_srli_epi16(q_i8_0, 4), mask_lo)));

			__m256 xv_lo0 = _mm256_loadu_ps(x + j + offset);
			__m256 xv_hi0 = _mm256_loadu_ps(x + j + 32 + offset);

			// Use FMA: (x * (d*q - m)) = x*d*q - x*m
			acc0 = _mm256_fmadd_ps(xv_lo0, _mm256_sub_ps(_mm256_mul_ps(qv_lo0, d1_vec), m1_vec), acc0);
			acc0 = _mm256_fmadd_ps(xv_hi0, _mm256_sub_ps(_mm256_mul_ps(qv_hi0, d2_vec), m2_vec), acc0);

			// Chunk 2 (16..31 bytes) - Pipelined with Chunk 1
			__m128i q_i8_1 = _mm_loadu_si64(q + offset + 8);

			__m256 qv_lo1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_and_si128(q_i8_1, mask_lo)));
			__m256 qv_hi1 = _mm256_cvtepi32_ps(
				_mm256_cvtepi8_epi32(_mm_and_si128(_mm_srli_epi16(q_i8_1, 4), mask_lo)));

			__m256 xv_lo1 = _mm256_loadu_ps(x + j + offset + 8);
			__m256 xv_hi1 = _mm256_loadu_ps(x + j + 32 + offset + 8);

			acc1 = _mm256_fmadd_ps(xv_lo1, _mm256_sub_ps(_mm256_mul_ps(qv_lo1, d1_vec), m1_vec), acc1);
			acc1 = _mm256_fmadd_ps(xv_hi1, _mm256_sub_ps(_mm256_mul_ps(qv_hi1, d2_vec), m2_vec), acc1);
		}
		q += 32;
		is += 2;
	}

	acc0 = _mm256_add_ps(acc0, acc1);
	return hsum_f32_avx2(acc0);
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

static inline __m256i float32_to_bf16_avx2(__m256 x0, __m256 x1)
{
	// Shift right by 16 to get the top 16 bits into the lower position
	//    Result logic: 0xYYYYZZZZ -> 0x0000YYYY
	__m256i i0 = _mm256_castps_si256(x0);
	__m256i i1 = _mm256_castps_si256(x1);

	i0 = _mm256_srli_epi32(i0, 16);
	i1 = _mm256_srli_epi32(i1, 16);

	// Pack 32-bit integers into 16-bit integers.
	//    _mm256_packus_epi32 packs lanes:
	//    Input:  [A0-A3][A4-A7] and [B0-B3][B4-B7]
	//    Output: [A0-A3, B0-B3] [A4-A7, B4-B7] (Notice the lane interleaving!)
	__m256i packed = _mm256_packus_epi32(i0, i1);

	// Fix the lane ordering.
	//    We need: [A0-A3, A4-A7] [B0-B3, B4-B7]
	//    Permute logic: 0xd8 = (11 01 10 00) -> Order: 3 1 2 0 ? No, check docs.
	//    We want indices 0, 2, 1, 3 from the 64-bit chunks.
	//    0xd8 = 11_01_10_00 in binary.
	return _mm256_permute4x64_epi64(packed, 0xd8);
}

void store_KV_cache_f32_bf16_avx2(struct TIEContext *ctx, int layer_idx, int start_pos, int batch_len, int sink_len)
{
	LayerKVCache *cache = &ctx->kv_cache[layer_idx];
	int kv_dim = ctx->model->num_kv_heads * ctx->model->head_dim;
	int ring_size = ctx->model->seq_length;
	int rolling_capacity = ring_size - sink_len;

	int tokens_written = 0;

	while (tokens_written < batch_len) {
		int current_logical_pos = start_pos + tokens_written;
		int current_physical_pos;

		// SINK + RING MAPPING
		if (current_logical_pos < sink_len) {
			current_physical_pos = current_logical_pos;
		} else {
			int offset = current_logical_pos - sink_len;
			current_physical_pos = sink_len + (offset % rolling_capacity);
		}

		// Calculate contiguous chunk size
		int chunk_len = 1;
		if (current_logical_pos >= sink_len) {
			int space_at_end = ring_size - current_physical_pos;
			int tokens_remaining = batch_len - tokens_written;
			chunk_len = (tokens_remaining < space_at_end) ? tokens_remaining : space_at_end;
		}

		// VECTORIZED COPY START
		long long dest_offset = (long long)current_physical_pos * kv_dim;
		long long src_offset = (long long)tokens_written * kv_dim;
		long long num_elements = (long long)chunk_len * kv_dim;

		uint16_t *k_cache = (uint16_t *)cache->k.data + dest_offset;
		uint16_t *v_cache = (uint16_t *)cache->v.data + dest_offset;
		float *K_src = (float *)ctx->mem.K.data + src_offset;
		float *V_src = (float *)ctx->mem.V.data + src_offset;

		long long i = 0;

		// Process 16 elements at a time (512 bits of F32 input -> 256 bits of BF16 output)
		for (; i <= num_elements - 16; i += 16) {
			// Load K
			__m256 k_f32_0 = _mm256_loadu_ps(K_src + i);
			__m256 k_f32_1 = _mm256_loadu_ps(K_src + i + 8);
			// Convert K
			__m256i k_bf16 = float32_to_bf16_avx2(k_f32_0, k_f32_1);
			// Store K
			_mm256_storeu_si256((__m256i *)(k_cache + i), k_bf16);

			// Load V
			__m256 v_f32_0 = _mm256_loadu_ps(V_src + i);
			__m256 v_f32_1 = _mm256_loadu_ps(V_src + i + 8);
			// Convert V
			__m256i v_bf16 = float32_to_bf16_avx2(v_f32_0, v_f32_1);
			// Store V
			_mm256_storeu_si256((__m256i *)(v_cache + i), v_bf16);
		}

		// Handle Scalar Remainder
		for (; i < num_elements; i++) {
			k_cache[i] = fp32_to_bf16(K_src[i]);
			v_cache[i] = fp32_to_bf16(V_src[i]);
		}

		tokens_written += chunk_len;
	}
}

__attribute__((target("avx2"))) float dot_product_bf16_q4k_avx2(const uint16_t *x_bf16, const block_q4_k *blk)
{
	const float d = fp16_to_fp32(blk->d);
	const float dmin = fp16_to_fp32(blk->dmin);
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

	// Use two accumulators for loop unrolling
	__m256 acc0 = _mm256_setzero_ps();
	__m256 acc1 = _mm256_setzero_ps();
	const uint8_t *q = blk->qs;
	int is = 0;

	for (int j = 0; j < QK_K; j += 64) {
		__m256 d1_vec = _mm256_set1_ps(scales[is]);
		__m256 m1_vec = _mm256_set1_ps(mins[is]);
		__m256 d2_vec = _mm256_set1_ps(scales[is + 1]);
		__m256 m2_vec = _mm256_set1_ps(mins[is + 1]);

		// Process 64 nibbles (32 bytes) per inner loop, unrolled
		for (int offset = 0; offset < 32; offset += 16) { // Process 16 bytes -> 32 nibbles
			// --- First 8 bytes ---
			__m128i q_i8_0 = _mm_loadu_si64(q + offset);
			const __m128i mask_lo = _mm_set1_epi8(0x0F);
			__m128i q_lo_i8_0 = _mm_and_si128(q_i8_0, mask_lo);
			__m128i q_hi_i8_0 = _mm_and_si128(_mm_srli_epi16(q_i8_0, 4), mask_lo);
			__m256 qv_lo0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_lo_i8_0));
			__m256 qv_hi0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi_i8_0));

			__m256 xv_lo0 = load_bf16_as_f32(x_bf16 + j + offset);
			__m256 xv_hi0 = load_bf16_as_f32(x_bf16 + j + 32 + offset);

			acc0 = _mm256_fmadd_ps(xv_lo0, _mm256_sub_ps(_mm256_mul_ps(qv_lo0, d1_vec), m1_vec), acc0);
			acc1 = _mm256_fmadd_ps(xv_hi0, _mm256_sub_ps(_mm256_mul_ps(qv_hi0, d2_vec), m2_vec), acc1);

			// --- Second 8 bytes ---
			__m128i q_i8_1 = _mm_loadu_si64(q + offset + 8);
			__m128i q_lo_i8_1 = _mm_and_si128(q_i8_1, mask_lo);
			__m128i q_hi_i8_1 = _mm_and_si128(_mm_srli_epi16(q_i8_1, 4), mask_lo);
			__m256 qv_lo1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_lo_i8_1));
			__m256 qv_hi1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi_i8_1));

			__m256 xv_lo1 = load_bf16_as_f32(x_bf16 + j + offset + 8);
			__m256 xv_hi1 = load_bf16_as_f32(x_bf16 + j + 32 + offset + 8);

			acc0 = _mm256_fmadd_ps(xv_lo1, _mm256_sub_ps(_mm256_mul_ps(qv_lo1, d1_vec), m1_vec), acc0);
			acc1 = _mm256_fmadd_ps(xv_hi1, _mm256_sub_ps(_mm256_mul_ps(qv_hi1, d2_vec), m2_vec), acc1);
		}
		q += 32;
		is += 2;
	}

	// Add the two accumulators together before the final horizontal sum
	acc0 = _mm256_add_ps(acc0, acc1);
	return hsum_f32_avx2(acc0);
}

__attribute__((target("avx2"))) float dot_product_bf16_q6k_avx2(const uint16_t *x_bf16, const block_q6_k *block)
{
	__m256 sum_vec = _mm256_setzero_ps();
	const float d = fp16_to_fp32(block->d);
	__m256 d_vec = _mm256_set1_ps(d);

	const uint8_t *ql = block->ql;
	const uint8_t *qh = block->qh;
	const int8_t *sc = block->scales;

	for (int n = 0; n < 2; n++) { // Two halves: 128 each
		const uint16_t *x_half = x_bf16 + n * 128;
		const uint8_t *ql_half = ql + n * 64;
		const uint8_t *qh_half = qh + n * 32;
		const int8_t *sc_half = sc + n * 8;

		for (int l = 0; l < 32; l += 8) {
			int is = (l >= 16) ? 1 : 0;

			// --- Load x as BF16 → FP32 ---
			__m256 x1 = load_bf16_as_f32(x_half + l);
			__m256 x2 = load_bf16_as_f32(x_half + l + 32);
			__m256 x3 = load_bf16_as_f32(x_half + l + 64);
			__m256 x4 = load_bf16_as_f32(x_half + l + 96);

			// --- Load quantization data ---
			__m128i ql_vec = _mm_loadu_si128((const __m128i *)(ql_half + l));
			__m128i qh_vec = _mm_loadu_si128((const __m128i *)(qh_half + l));
			__m128i ql_vec32 = _mm_loadu_si128((const __m128i *)(ql_half + l + 32));

			// q1
			__m128i ql_low = _mm_and_si128(ql_vec, _mm_set1_epi8(0x0F));
			__m128i qh_01 = _mm_and_si128(qh_vec, _mm_set1_epi8(3));
			__m128i qh_shift1 = _mm_slli_epi16(qh_01, 4);
			__m128i q_vec1 = _mm_or_si128(ql_low, qh_shift1);
			__m128i q_int1 = _mm_sub_epi8(q_vec1, _mm_set1_epi8(32));
			__m256 q1_vec = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_int1));
			__m256 scale_vec1 = _mm256_set1_ps((float)sc_half[is]);
			__m256 w1_vec = _mm256_mul_ps(d_vec, _mm256_mul_ps(scale_vec1, q1_vec));
			sum_vec = _mm256_fmadd_ps(w1_vec, x1, sum_vec);

			// q2
			__m128i ql_low2 = _mm_and_si128(ql_vec32, _mm_set1_epi8(0x0F));
			__m128i qh_23 = _mm_and_si128(_mm_srli_epi16(qh_vec, 2), _mm_set1_epi8(3));
			__m128i qh_shift2 = _mm_slli_epi16(qh_23, 4);
			__m128i q_vec2 = _mm_or_si128(ql_low2, qh_shift2);
			__m128i q_int2 = _mm_sub_epi8(q_vec2, _mm_set1_epi8(32));
			__m256 q2_vec = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_int2));
			__m256 scale_vec2 = _mm256_set1_ps((float)sc_half[is + 2]);
			__m256 w2_vec = _mm256_mul_ps(d_vec, _mm256_mul_ps(scale_vec2, q2_vec));
			sum_vec = _mm256_fmadd_ps(w2_vec, x2, sum_vec);

			// q3
			__m128i ql_high = _mm_and_si128(_mm_srli_epi16(ql_vec, 4), _mm_set1_epi8(0x0F));
			__m128i qh_45 = _mm_and_si128(_mm_srli_epi16(qh_vec, 4), _mm_set1_epi8(3));
			__m128i qh_shift3 = _mm_slli_epi16(qh_45, 4);
			__m128i q_vec3 = _mm_or_si128(ql_high, qh_shift3);
			__m128i q_int3 = _mm_sub_epi8(q_vec3, _mm_set1_epi8(32));
			__m256 q3_vec = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_int3));
			__m256 scale_vec3 = _mm256_set1_ps((float)sc_half[is + 4]);
			__m256 w3_vec = _mm256_mul_ps(d_vec, _mm256_mul_ps(scale_vec3, q3_vec));
			sum_vec = _mm256_fmadd_ps(w3_vec, x3, sum_vec);

			// q4
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

	// Horizontal reduce sum_vec → scalar
	__m128 sum_low = _mm256_castps256_ps128(sum_vec);
	__m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
	__m128 sum_4 = _mm_add_ps(sum_low, sum_high);
	__m128 sum_2 = _mm_hadd_ps(sum_4, sum_4);
	__m128 sum_1 = _mm_hadd_ps(sum_2, sum_2);
	float sum = _mm_cvtss_f32(sum_1);

	return sum;
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

inline float gelu_fast(float x)
{
	const float a = 0.7978845608028654f; // sqrt(2/pi)
	const float b = 0.044715f;
	float t = tanhf(a * x * (1.0f + b * x * x));
	return 0.5f * x * (1.0f + t);
}

__attribute__((target("avx2"))) void geglu_activation_f32_f32_avx2(void *gate, const void *up, int size)
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

__attribute__((target("avx2"))) void conv_2d_f32_f32_f32_f32_avx2(MemType *dest, const MemType *src_image,
								  const Tensor *kernel_tensor,
								  const Tensor *bias_tensor, int H_in, int W_in,
								  int stride, int padding)
{
	float *dest_data = (float *)dest->data;
	const float *src_data = (const float *)src_image->data;
	const float *kernel_data = (const float *)kernel_tensor->mem.data;
	const float *bias_data = bias_tensor ? (const float *)bias_tensor->mem.data : NULL;

	const int K_H = kernel_tensor->dimensions[0];
	const int K_W = kernel_tensor->dimensions[1];
	const int C_in = kernel_tensor->dimensions[2];
	const int C_out = kernel_tensor->dimensions[3]; // e.g. 1152

	const int H_out = (H_in + 2 * padding - K_H) / stride + 1;
	const int W_out = (W_in + 2 * padding - K_W) / stride + 1;
	const size_t dest_plane_size = (size_t)H_out * W_out;
	const size_t src_plane_size = (size_t)H_in * W_in;

	// Strides for kernel traversal [C_out, C_in, K_H, K_W]
	const size_t k_stride_C_out = (size_t)C_in * K_H * K_W;

	// Loop over Output Channels (Filters) in blocks of 8
	for (int co = 0; co < C_out; co += 8) {
		// Check boundary if C_out is not multiple of 8 (unlikely for 1152, but safe)
		int remain = C_out - co;
		if (remain < 8) {
			// Fallback to scalar for tail, or use masked AVX (omitted for brevity)
			// conv_2d_f32_scalar_partial(...)
			continue;
		}

		// Pre-load Bias
		__m256 bias_vec = bias_data ? _mm256_loadu_ps(bias_data + co) : _mm256_setzero_ps();

		// Iterate over Output Grid
		for (int oy = 0; oy < H_out; ++oy) {
			for (int ox = 0; ox < W_out; ++ox) {

				// Initialize accumulators with bias
				// We need 8 separate accumulators, one for each of the 8 filters we are processing
				// Actually, we are calculating ONE output pixel position (ox, oy) for 8 filters.
				// So we need a vector of 8 sums.
				__m256 sum_vec = bias_vec;

				// Convolution Loop
				for (int ci = 0; ci < C_in; ++ci) {
					const float *src_plane = src_data + (ci * src_plane_size);
					const float *k_ptr_base =
						kernel_data + (co * k_stride_C_out) + (ci * K_H * K_W);

					for (int ky = 0; ky < K_H; ++ky) {
						int iy = oy * stride + ky - padding;
						if (iy < 0 || iy >= H_in)
							continue;

						for (int kx = 0; kx < K_W; ++kx) {
							int ix = ox * stride + kx - padding;
							if (ix < 0 || ix >= W_in)
								continue;

							// Load 1 input pixel value and broadcast it
							float val = src_plane[iy * W_in + ix];
							__m256 input_val = _mm256_set1_ps(val);

							// Load 8 weights (one for each of the 8 filters at this kernel
							// position) Memory layout: Weights are [C_out, C_in, KH, KW] We
							// need weights[co...co+7][ci][ky][kx] These are NOT contiguous!
							// They are stride_C_out apart. We must gather them.

							// Standard layout [C_out, C_in, K_H, K_W] is bad for AVX2
							// output-channel parallelism. A [C_in, K_H, K_W, C_out] layout
							// would allow contiguous loads here. Since we can't change
							// layout easily on the fly:

							float w0 = k_ptr_base[0 * k_stride_C_out + ky * K_W + kx];
							float w1 = k_ptr_base[1 * k_stride_C_out + ky * K_W + kx];
							float w2 = k_ptr_base[2 * k_stride_C_out + ky * K_W + kx];
							float w3 = k_ptr_base[3 * k_stride_C_out + ky * K_W + kx];
							float w4 = k_ptr_base[4 * k_stride_C_out + ky * K_W + kx];
							float w5 = k_ptr_base[5 * k_stride_C_out + ky * K_W + kx];
							float w6 = k_ptr_base[6 * k_stride_C_out + ky * K_W + kx];
							float w7 = k_ptr_base[7 * k_stride_C_out + ky * K_W + kx];

							__m256 weights = _mm256_setr_ps(w0, w1, w2, w3, w4, w5, w6, w7);

							// FMA: sum += input * weights
							sum_vec = _mm256_fmadd_ps(input_val, weights, sum_vec);
						}
					}
				}

				// Store 8 results to the 8 output planes
				float res[8];
				_mm256_storeu_ps(res, sum_vec);

				for (int k = 0; k < 8; ++k) {
					float *dest_ptr = dest_data + ((co + k) * dest_plane_size);
					dest_ptr[oy * W_out + ox] = res[k];
				}
			}
		}
	}
}

__attribute__((target("avx2"))) void layer_norm_f32_f32_f32_f32_avx2(MemType *dest, const MemType *src,
								     const Tensor *weight, const Tensor *bias, int size,
								     float eps)
{
	float *x = (float *)src->data;
	float *d = (float *)dest->data;
	const float *w = (const float *)weight->mem.data;
	const float *b = (const float *)bias->mem.data;

	// Calculate Mean
	__m256 sum_vec = _mm256_setzero_ps();
	int i = 0;
	for (; i <= size - 8; i += 8) {
		sum_vec = _mm256_add_ps(sum_vec, _mm256_loadu_ps(x + i));
	}

	// Horizontal sum
	float temp[8];
	_mm256_storeu_ps(temp, sum_vec);
	float sum = 0.0f;
	for (int k = 0; k < 8; k++)
		sum += temp[k];
	for (; i < size; i++)
		sum += x[i]; // Tail

	float mean = sum / size;
	__m256 mean_vec = _mm256_set1_ps(mean);

	// Calculate Variance
	__m256 sq_sum_vec = _mm256_setzero_ps();
	i = 0;
	for (; i <= size - 8; i += 8) {
		__m256 val = _mm256_sub_ps(_mm256_loadu_ps(x + i), mean_vec);
		sq_sum_vec = _mm256_fmadd_ps(val, val, sq_sum_vec);
	}

	_mm256_storeu_ps(temp, sq_sum_vec);
	float sq_sum = 0.0f;
	for (int k = 0; k < 8; k++)
		sq_sum += temp[k];
	for (; i < size; i++) {
		float diff = x[i] - mean;
		sq_sum += diff * diff;
	}

	float inv_std = 1.0f / sqrtf(sq_sum / size + eps);
	__m256 inv_std_vec = _mm256_set1_ps(inv_std);

	// Normalize and Scale
	i = 0;
	for (; i <= size - 8; i += 8) {
		__m256 val = _mm256_loadu_ps(x + i);
		__m256 w_vec = _mm256_loadu_ps(w + i);
		__m256 b_vec = _mm256_loadu_ps(b + i);

		// (val - mean) * inv_std * weight + bias
		__m256 norm = _mm256_mul_ps(_mm256_sub_ps(val, mean_vec), inv_std_vec);
		__m256 res = _mm256_fmadd_ps(norm, w_vec, b_vec);

		_mm256_storeu_ps(d + i, res);
	}

	// Tail
	for (; i < size; i++) {
		d[i] = ((x[i] - mean) * inv_std) * w[i] + b[i];
	}
}

__attribute__((target("avx2"))) void transpose_f32_avx2(MemType *dest, const MemType *src, int rows, int cols)
{
	const float *src_data = (const float *)src->data;
	float *dest_data = (float *)dest->data;

	// Block size 8x8 for AVX2 (ymm registers hold 8 floats)
	const int block_sz = 8;

	// Process 8x8 blocks
	for (int r = 0; r < rows; r += block_sz) {
		for (int c = 0; c < cols; c += block_sz) {

			// Check bounds (handle edges if dimensions aren't multiples of 8)
			int r_limit = (r + block_sz <= rows) ? block_sz : (rows - r);
			int c_limit = (c + block_sz <= cols) ? block_sz : (cols - c);

			if (r_limit == 8 && c_limit == 8) {
				// Fast path: Full 8x8 block using Intrinsics
				__m256 row0 = _mm256_loadu_ps(&src_data[(r + 0) * cols + c]);
				__m256 row1 = _mm256_loadu_ps(&src_data[(r + 1) * cols + c]);
				__m256 row2 = _mm256_loadu_ps(&src_data[(r + 2) * cols + c]);
				__m256 row3 = _mm256_loadu_ps(&src_data[(r + 3) * cols + c]);
				__m256 row4 = _mm256_loadu_ps(&src_data[(r + 4) * cols + c]);
				__m256 row5 = _mm256_loadu_ps(&src_data[(r + 5) * cols + c]);
				__m256 row6 = _mm256_loadu_ps(&src_data[(r + 6) * cols + c]);
				__m256 row7 = _mm256_loadu_ps(&src_data[(r + 7) * cols + c]);

				// Unpack 32-bit elements
				__m256 t0 = _mm256_unpacklo_ps(row0, row1);
				__m256 t1 = _mm256_unpackhi_ps(row0, row1);
				__m256 t2 = _mm256_unpacklo_ps(row2, row3);
				__m256 t3 = _mm256_unpackhi_ps(row2, row3);
				__m256 t4 = _mm256_unpacklo_ps(row4, row5);
				__m256 t5 = _mm256_unpackhi_ps(row4, row5);
				__m256 t6 = _mm256_unpacklo_ps(row6, row7);
				__m256 t7 = _mm256_unpackhi_ps(row6, row7);

				// Unpack 64-bit blocks
				__m256 tt0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
				__m256 tt1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
				__m256 tt2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
				__m256 tt3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
				__m256 tt4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
				__m256 tt5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
				__m256 tt6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
				__m256 tt7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

				// Unpack 128-bit blocks (swapping between lanes)
				row0 = _mm256_permute2f128_ps(tt0, tt4, 0x20);
				row1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
				row2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
				row3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
				row4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
				row5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
				row6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
				row7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);

				// Store
				_mm256_storeu_ps(&dest_data[(c + 0) * rows + r], row0);
				_mm256_storeu_ps(&dest_data[(c + 1) * rows + r], row1);
				_mm256_storeu_ps(&dest_data[(c + 2) * rows + r], row2);
				_mm256_storeu_ps(&dest_data[(c + 3) * rows + r], row3);
				_mm256_storeu_ps(&dest_data[(c + 4) * rows + r], row4);
				_mm256_storeu_ps(&dest_data[(c + 5) * rows + r], row5);
				_mm256_storeu_ps(&dest_data[(c + 6) * rows + r], row6);
				_mm256_storeu_ps(&dest_data[(c + 7) * rows + r], row7);

			} else {
				// Slow path: edge cases
				for (int rr = 0; rr < r_limit; ++rr) {
					for (int cc = 0; cc < c_limit; ++cc) {
						dest_data[(c + cc) * rows + (r + rr)] =
							src_data[(r + rr) * cols + (c + cc)];
					}
				}
			}
		}
	}
}

#if 0
__attribute__((target("avx2"))) void quantize_row_q8_0_avx2(const float *__restrict x, void *__restrict y, int k)
{
	const int nb = k / QK8_0; // Q8_0 block size is 32
	block_q8_0 *blocks = (block_q8_0 *)y;

	for (int i = 0; i < nb; i++) {
		// Load 32 floats
		__m256 v0 = _mm256_loadu_ps(x + i * 32 + 0);
		__m256 v1 = _mm256_loadu_ps(x + i * 32 + 8);
		__m256 v2 = _mm256_loadu_ps(x + i * 32 + 16);
		__m256 v3 = _mm256_loadu_ps(x + i * 32 + 24);

		// Find Absolute Max to calculate scale
		__m256 abs0 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v0);
		__m256 abs1 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v1);
		__m256 abs2 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v2);
		__m256 abs3 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v3);

		__m256 max_v = _mm256_max_ps(_mm256_max_ps(abs0, abs1), _mm256_max_ps(abs2, abs3));

		// Horizontal Max Reduction
		float amax = 0.0f;
		// (Simplest reduction for clarity, full AVX reduction is faster)
		float temp[8];
		_mm256_storeu_ps(temp, max_v);
		for (int j = 0; j < 8; j++)
			if (temp[j] > amax)
				amax = temp[j];

		// Calculate scale: we want max value to map to 127
		float d = amax / 127.0f;
		if (!d)
			d = 1.0f;
		blocks[i].d = fp32_to_fp16(d); // Store as FP16
		float id = 1.0f / d;

		// Quantize: x * (1/d)
		__m256 mul = _mm256_set1_ps(id);

		// Convert to 32-bit int, then pack to 16-bit, then 8-bit
		__m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, mul));
		__m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, mul));
		__m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, mul));
		__m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, mul));

		// Pack 32 -> 16
		__m256i p0 = _mm256_packs_epi32(i0, i1);
		__m256i p1 = _mm256_packs_epi32(i2, i3);

		// Pack 16 -> 8
		__m256i p_final = _mm256_packs_epi16(p0, p1);

		// Fix permutation (AVX2 pack instruction acts on 128-bit lanes)
		p_final = _mm256_permute4x64_epi64(p_final, 0xD8);

		// Store
		_mm256_storeu_si256((__m256i *)blocks[i].qs, p_final);
	}
}
#endif

#endif
