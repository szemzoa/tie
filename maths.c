#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#include "config.h"
#include "maths.h"
#include "threadpool.h"
#include "main.h"


void dequantize_row_q6_k_avx2(const void *__restrict__ q_void, float *__restrict__ y, int k);
void dequantize_row_q4_k_avx2(const void *__restrict__ q_void, float *__restrict__ y, int k);

inline float bf16_to_fp32(uint16_t bf16)
{
	union {
		uint32_t u;
		float f;
	} converter;
	converter.u = ((uint32_t)bf16) << 16;
	return converter.f;
}

uint16_t fp32_to_bf16(float f)
{
	union {
		float f;
		uint32_t u;
	} u = {.f = f};
	return (uint16_t)(u.u >> 16);
}

float *convert_bf16_to_f32(void *bf16_ptr, size_t count)
{
	if (!bf16_ptr)
		return NULL;
	uint16_t *bf16 = (uint16_t *)bf16_ptr;
	float *f32 = malloc(count * sizeof(float));
	if (!f32) {
		fprintf(stderr, "Failed to allocate memory for FP32 conversion.\n");
		return NULL;
	}
	for (size_t i = 0; i < count; i++) {
		f32[i] = bf16_to_fp32(bf16[i]);
	}
	return f32;
}

static inline float fp16_to_fp32(uint16_t h)
{
	// Simple FP16 to FP32 conversion (handles normalized numbers)
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

void dequantize_row_q6_k(const void *__restrict__ q_void, float *__restrict__ y, int k)
{
#ifdef CONFIG_ENABLE_AVX2
	if (__builtin_cpu_supports("avx2")) {
		return dequantize_row_q6_k_avx2(q_void, y, k);
	}
#endif
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

void get_scale_min_k4(int j, const uint8_t *scales, uint8_t *scale, uint8_t *min)
{
	if (j < 4) {
		*scale = scales[j] & 0x3F;
		*min = scales[j + 4] & 0x3F;
	} else {
		*scale = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
		*min = (scales[j + 4] >> 4) | ((scales[j - 0] >> 6) << 4);
	}
}

void dequantize_row_q4_k(const void *__restrict__ q_void, float *__restrict__ y, int k)
{
#ifdef CONFIG_ENABLE_AVX2
	if (__builtin_cpu_supports("avx2")) {
		return dequantize_row_q4_k_avx2(q_void, y, k);
	}
#endif
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

#ifdef CONFIG_ENABLE_AVX2
__attribute__((target("avx2"))) void rms_norm_avx2(float *__restrict o, const float *__restrict x,
						   const float *__restrict weight, int size, float eps)
{
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

inline __m256 load_bf16_as_f32(const uint16_t *src)
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

__attribute__((target("avx2"))) void mat_vec_bf16_avx2_row(const float *__restrict x, const uint16_t *__restrict w_bf16,
							   float *__restrict o, int in_dim, int start_row, int end_row)
{
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

__attribute__((target("avx2"))) float dot_product_f32_avx2(const float *__restrict a, const float *__restrict b,
							   int size)
{
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

__attribute__((target("avx2"))) float dot_product_f32_bf16_avx2(const float *__restrict a, const uint16_t *__restrict b,
								int size)
{
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

	// Horizontal add
	float sum[8];
	_mm256_storeu_ps(sum, acc);

	float result = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];

	// Tail loop
	for (; i < size; i++) {
		float b_val = bf16_to_fp32(b[i]);
		result = fmaf(a[i], b_val, result);
	}

	return result;
}

__attribute__((target("avx2"))) void mat_vec_avx2_row(const float *x, const float *w, float *o, int in_dim,
						      int start_row, int end_row)
{
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

__attribute__((target("avx2"))) void accumulate_weighted_fp32_bf16_avx2(float *__restrict out, float weight,
									const uint16_t *__restrict v_bf16, int size)
{
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

__attribute__((target("avx2"))) void apply_rope_cache_avx2(struct ctx_t *ctx, float *x, int pos, int head_dim)
{
    rope_cache_t *rope_cache = ctx->rope_cache;
    int h_dim_half = head_dim / 2;

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
#endif

void rms_norm(float *__restrict o, const float *__restrict x, const float *__restrict weight, int size, float eps)
{
#ifdef CONFIG_ENABLE_AVX2
	if (__builtin_cpu_supports("avx2")) {
		return rms_norm_avx2(o, x, weight, size, eps);
	}
#endif
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

void mat_vec_bf16_row(const float *__restrict x, const uint16_t *__restrict w_bf16, float *__restrict o, int in_dim,
		      int start_row, int end_row)
{
#ifdef CONFIG_ENABLE_AVX2
	if (__builtin_cpu_supports("avx2")) {
		return mat_vec_bf16_avx2_row(x, w_bf16, o, in_dim, start_row, end_row);
	}
#endif
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

void mat_vec_fp32_row(const float *__restrict x, const float *__restrict w, float *__restrict o, int in_dim,
		      int start_row, int end_row)
{
#ifdef CONFIG_ENABLE_AVX2
	if (__builtin_cpu_supports("avx2")) {
		return mat_vec_avx2_row(x, w, o, in_dim, start_row, end_row);
	}
#endif
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

float dot_product_f32(const float *__restrict a, const float *__restrict b, int size)
{
#ifdef CONFIG_ENABLE_AVX2
	if (__builtin_cpu_supports("avx2")) {
		return dot_product_f32_avx2(a, b, size);
	}
#endif
	float sum = 0.0f;
	for (int j = 0; j < size; j++)
		sum = fmaf(a[j], b[j], sum);

	return sum;
}

float dot_product_f32_bf16(const float *__restrict a, const uint16_t *__restrict b, int size)
{
#ifdef CONFIG_ENABLE_AVX2
	if (__builtin_cpu_supports("avx2")) {
		return dot_product_f32_bf16_avx2(a, b, size);
	}
#endif

	float sum = 0.0f;
	for (int j = 0; j < size; j++)
		sum = fmaf(a[j], bf16_to_fp32(b[j]), sum);

	return sum;
}

void accumulate_weighted_fp32_bf16(float *out, float weight, const uint16_t *v, int size)
{
#ifdef CONFIG_ENABLE_AVX2
	if (__builtin_cpu_supports("avx2")) {
		return accumulate_weighted_fp32_bf16_avx2(out, weight, v, size);
	}
#endif

	for (int i = 0; i < size; i++) {
		out[i] = fmaf(weight, bf16_to_fp32(v[i]), out[i]);
	}
}

static void mat_vec_task_fp32(void *arg)
{
	mat_vec_task_fp32_t *task = (mat_vec_task_fp32_t *)arg;
	mat_vec_fp32_row(task->x, task->w, task->o, task->in_dim, task->start_row, task->end_row);
	free(task); // The worker is responsible for freeing its arguments.
}

void parallel_mat_vec_fp32(const float *x, const float *w, float *o, int in_dim, int out_dim, bool use_threads)
{
	if (use_threads == 0) {
		// Fallback to the sequential version if the threadpool is disabled.
		mat_vec_fp32_row(x, w, o, in_dim, 0, out_dim);
		return;
	}

	int num_threads = thread_pool->num_threads;
	// Calculate rows per thread, ensuring the last thread handles any remainder.
	int rows_per_thread = (out_dim + num_threads - 1) / num_threads;

	for (int t = 0; t < num_threads; t++) {
		int start_row = t * rows_per_thread;
		int end_row = start_row + rows_per_thread;

		if (start_row >= out_dim) {
			break; // No more rows to process.
		}
		if (end_row > out_dim) {
			end_row = out_dim; // Clamp to the actual output dimension.
		}

		// Use heap allocation for task arguments to ensure thread safety.
		mat_vec_task_fp32_t *task = malloc(sizeof(mat_vec_task_fp32_t));
		if (!task) {
			fprintf(stderr, "ERROR: Failed to allocate memory for mat_vec_task_fp32\n");
			continue;
		}

		*task = (mat_vec_task_fp32_t){
			.x = x, .w = w, .o = o, .in_dim = in_dim, .start_row = start_row, .end_row = end_row};

		thread_pool_submit(thread_pool, mat_vec_task_fp32, task);
	}
	// Wait for this specific batch of matrix-vector tasks to complete.
	thread_pool_wait(thread_pool);
}

float dot_product_q6_k(const float *x, const block_q6_k *block)
{
#ifdef CONFIG_ENABLE_AVX2
	if (__builtin_cpu_supports("avx2")) {
		return dot_product_q6_k_avx2(x, block);
	}
#endif
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

float dot_product_q4_k(const float *x, const block_q4_k *blk)
{
#ifdef CONFIG_ENABLE_AVX2
	if (__builtin_cpu_supports("avx2")) {
		return dot_product_q4_k_avx2(x, blk);
	}
#endif
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

// Performs mat-vec multiplication with Q6_K weights.
// Dequantizes blocks on the fly.
void mat_vec_q6_k_row(const float *x, const void *w_void, float *o, int in_dim, int start_row, int end_row)
{
	const block_q6_k *w = (const block_q6_k *)w_void;
	const int K = QK_K;
	const int nb = (in_dim + K - 1) / K; // Ceiling division

	for (int i = start_row; i < end_row; i++) {
		const block_q6_k *w_row = w + i * nb;
		float sum = 0.0f;
		for (int j = 0; j < nb; j++) {
			sum += dot_product_q6_k(x + j * QK_K, &w_row[j]);
		}
		o[i] = sum;
	}
}

void mat_vec_q4_k_row(const float *x, const void *w_void, float *o, int in_dim, int start_row, int end_row)
{
	const block_q4_k *w = (const block_q4_k *)w_void;
	const int K = QK_K; // 256
	const int nb = (in_dim + K - 1) / K;

	for (int i = start_row; i < end_row; i++) {
		const block_q4_k *w_row = w + i * nb;
		float sum = 0.0f;

		for (int j = 0; j < nb; j++) {
			sum += dot_product_q4_k(x + j * QK_K, &w_row[j]);
		}

		o[i] = sum;
	}
}

static void mat_vec_task(void *arg)
{
	mat_vec_task_t *task = (mat_vec_task_t *)arg;
	const Tensor *tensor = task->W;

	switch (tensor->type) {
	case GGML_TYPE_Q4_K:
		mat_vec_q4_k_row(task->x, tensor->data, task->o, task->in_dim, task->start_row, task->end_row);
		break;

	case GGML_TYPE_Q6_K:
		mat_vec_q6_k_row(task->x, tensor->data, task->o, task->in_dim, task->start_row, task->end_row);
		break;

	case GGML_TYPE_BF16:
		mat_vec_bf16_row(task->x, (uint16_t *)tensor->data, task->o, task->in_dim, task->start_row,
				 task->end_row);
		break;

	default:
		fprintf(stderr, "Error: Unsupported tensor type %d in parallel_mat_vec_unified\n", tensor->type);
		break;
	}

	free(task); // The worker is responsible for freeing its arguments.
}

void parallel_mat_vec_unified(const float *x, const Tensor *w_tensor, float *o, int in_dim, int out_dim,
			      bool use_threads)
{
	// Fallback to the sequential version if the threadpool is disabled.
	if (use_threads == 0) {
		switch (w_tensor->type) {
		case GGML_TYPE_Q4_K:
			mat_vec_q4_k_row(x, w_tensor->data, o, in_dim, 0, out_dim);
			break;

		case GGML_TYPE_Q6_K:
			mat_vec_q6_k_row(x, w_tensor->data, o, in_dim, 0, out_dim);
			break;

		case GGML_TYPE_BF16:
			mat_vec_bf16_row(x, (uint16_t *)w_tensor->data, o, in_dim, 0, out_dim);
			break;

		default:
			fprintf(stderr, "Error: Unsupported tensor type %d in parallel_mat_vec_unified\n",
				w_tensor->type);
			break;
		}
		return;
	}

	int num_threads = thread_pool->num_threads;
	// Calculate rows per thread, ensuring the last thread handles any remainder.
	int rows_per_thread = (out_dim + num_threads - 1) / num_threads;

	for (int t = 0; t < num_threads; t++) {
		int start_row = t * rows_per_thread;
		int end_row = start_row + rows_per_thread;

		if (start_row >= out_dim) {
			break; // No more rows to process.
		}
		if (end_row > out_dim) {
			end_row = out_dim; // Clamp to the actual output dimension.
		}

		// Use heap allocation for task arguments to ensure thread safety.
		mat_vec_task_t *task = malloc(sizeof(mat_vec_task_t));
		if (!task) {
			fprintf(stderr, "ERROR: Failed to allocate memory for mat_vec_task\n");
			continue;
		}

		*task = (mat_vec_task_t){
			.x = x, .W = w_tensor, .o = o, .in_dim = in_dim, .start_row = start_row, .end_row = end_row};

		thread_pool_submit(thread_pool, mat_vec_task, task);
	}
	// Wait for this specific batch of matrix-vector tasks to complete.
	thread_pool_wait(thread_pool);
}

static void mat_mat_task(void *arg)
{
	mat_mat_task_t *task = (mat_mat_task_t *)arg;
	const Tensor *tensor = task->W;

	// We call the non parallel version,
	// because the parallelism is now handled at this higher level.
	switch (tensor->type) {

	case GGML_TYPE_Q4_K:
		for (int i = task->start_row; i < task->end_row; ++i) {
			const float *x_row = task->X + (long long)i * task->in_dim;
			float *o_row = task->O + (long long)i * task->out_dim;

			mat_vec_q4_k_row(x_row, tensor->data, o_row, task->in_dim, 0, task->out_dim);
		}
		break;

	case GGML_TYPE_Q6_K:
		for (int i = task->start_row; i < task->end_row; ++i) {
			const float *x_row = task->X + (long long)i * task->in_dim;
			float *o_row = task->O + (long long)i * task->out_dim;

			mat_vec_q6_k_row(x_row, tensor->data, o_row, task->in_dim, 0, task->out_dim);
		}
		break;

	case GGML_TYPE_BF16:
		for (int i = task->start_row; i < task->end_row; ++i) {
			const float *x_row = task->X + (long long)i * task->in_dim;
			float *o_row = task->O + (long long)i * task->out_dim;

			mat_vec_bf16_row(x_row, (uint16_t *)tensor->data, o_row, task->in_dim, 0, task->out_dim);
		}
		break;

	default:
		break;
	}
	free(task);
}

void parallel_mat_mat_unified(const float *X, const Tensor *W_tensor, float *O, int prompt_len, int in_dim, int out_dim,
			      bool use_threads)
{
	// If batch size is 1, call the specialized and faster mat_vec dispatcher.
	if (prompt_len == 1) {
		parallel_mat_vec_unified(X, W_tensor, O, in_dim, out_dim, use_threads);
		return;
	}

	// --- Batched Prompt Processing (prompt_len > 1) ---
	int num_threads = thread_pool->num_threads;
	int rows_per_thread = (prompt_len + num_threads - 1) / num_threads;

	for (int t = 0; t < num_threads; t++) {
		int start_row = t * rows_per_thread;
		int end_row = start_row + rows_per_thread;
		if (end_row > prompt_len)
			end_row = prompt_len;
		if (start_row >= end_row)
			break;

		mat_mat_task_t *task = malloc(sizeof(mat_mat_task_t));
		*task = (mat_mat_task_t){
			.X = X,
			.W = W_tensor,
			.O = O,
			.in_dim = in_dim,
			.out_dim = out_dim,
			.start_row = start_row,
			.end_row = end_row,
		};
		thread_pool_submit(thread_pool, mat_mat_task, task);
	}
	thread_pool_wait(thread_pool);
}
