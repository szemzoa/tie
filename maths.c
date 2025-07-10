#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <immintrin.h>

#include "config.h"
#include "maths.h"
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

#ifdef CONFIG_ENABLE_AVX2
void rms_norm_avx2(float *__restrict o, const float *__restrict x, const float *__restrict weight, int size, float eps)
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

void mat_vec_bf16_avx2_row(const float *__restrict x, const uint16_t *__restrict w_bf16, float *__restrict o,
			   int in_dim, int start_row, int end_row)
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

void mat_vec_bf16_avx2(const float *__restrict x, const uint16_t *__restrict w_bf16, float *__restrict o, int in_dim,
		       int out_dim)
{
	for (int i = 0; i < out_dim; i++) {
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

float dot_product_f32_avx2(const float *__restrict a, const float *__restrict b, int size)
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

void mat_vec_avx2(const float *x, const float *w, float *o, int in_dim, int out_dim)
{
	for (int i = 0; i < out_dim; i++) {
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

void mat_vec_avx2_row(const float *x, const float *w, float *o, int in_dim, int start_row, int end_row)
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
#endif

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

void mat_vec_bf16(const float *__restrict x, const uint16_t *__restrict w_bf16, float *__restrict o, int in_dim,
		  int out_dim)
{
#ifdef CONFIG_ENABLE_AVX2
	if (__builtin_cpu_supports("avx2")) {
		return mat_vec_bf16_avx2(x, w_bf16, o, in_dim, out_dim);
	}
#endif
	for (int i = 0; i < out_dim; i++) {
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

void mat_vec_row(const float *__restrict x, const float *__restrict w, float *__restrict o, int in_dim, int start_row,
		 int end_row)
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

void mat_vec(const float *__restrict x, const float *__restrict w, float *__restrict o, int in_dim, int out_dim)
{
#ifdef CONFIG_ENABLE_AVX2
	if (__builtin_cpu_supports("avx2")) {
		return mat_vec_avx2(x, w, o, in_dim, out_dim);
	}
#endif
	for (int i = 0; i < out_dim; i++) {
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

// Ensure process_mat_vec_task frees its argument
static void process_mat_vec_task_bf16(void *arg)
{
	mat_vec_task_bf16_t *task = (mat_vec_task_bf16_t *)arg;
	mat_vec_bf16_row(task->x, task->w_bf16, task->o, task->in_dim, task->start_row, task->end_row);
	free(task); // The worker is responsible for freeing its arguments.
}

void parallel_mat_vec_bf16(const float *x, const uint16_t *w_bf16, float *o, int in_dim, int out_dim, bool use_threads)
{
	if (use_threads == 0) {
		// Fallback to the sequential version if the threadpool is disabled.
		mat_vec_bf16(x, w_bf16, o, in_dim, out_dim);
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
		mat_vec_task_bf16_t *task = malloc(sizeof(mat_vec_task_bf16_t));
		if (!task) {
			fprintf(stderr, "ERROR: Failed to allocate memory for mat_vec_task\n");
			continue;
		}

		*task = (mat_vec_task_bf16_t){
			.x = x, .w_bf16 = w_bf16, .o = o, .in_dim = in_dim, .start_row = start_row, .end_row = end_row};

		thread_pool_submit(thread_pool, process_mat_vec_task_bf16, task);
	}
	// Wait for this specific batch of matrix-vector tasks to complete.
	thread_pool_wait(thread_pool);
}

static void process_mat_vec_task(void *arg)
{
	mat_vec_task_t *task = (mat_vec_task_t *)arg;
	mat_vec_row(task->x, task->w, task->o, task->in_dim, task->start_row, task->end_row);
	free(task); // The worker is responsible for freeing its arguments.
}

void parallel_mat_vec(const float *x, const float *w, float *o, int in_dim, int out_dim, bool use_threads)
{
	if (use_threads == 0) {
		// Fallback to the sequential version if the threadpool is disabled.
		mat_vec(x, w, o, in_dim, out_dim);
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
			.x = x, .w = w, .o = o, .in_dim = in_dim, .start_row = start_row, .end_row = end_row};

		thread_pool_submit(thread_pool, process_mat_vec_task, task);
	}
	// Wait for this specific batch of matrix-vector tasks to complete.
	thread_pool_wait(thread_pool);
}

static void process_mat_mat_task(void *arg)
{
	mat_mat_task_t *task = (mat_mat_task_t *)arg;
	for (int i = task->start_row; i < task->end_row; ++i) {
		const float *x_row = task->X + (long long)i * task->in_dim;
		float *o_row = task->O + (long long)i * task->out_dim;
		// We call the original mat_vec, not the parallel version,
		// because the parallelism is now handled at this higher level.
		mat_vec_bf16(x_row, task->W, o_row, task->in_dim, task->out_dim);
	}
	free(task);
}

void parallel_mat_mat_bf16(const float *X, const uint16_t *W, float *O, int prompt_len, int in_dim, int out_dim,
			   int use_threads)
{
	if (prompt_len > 1) {
		if (use_threads == 0) {
			// Fallback to a simple sequential loop if threads are disabled
			for (int i = 0; i < prompt_len; ++i) {
				mat_vec_bf16(X + (long long)i * in_dim, W, O + (long long)i * out_dim, in_dim, out_dim);
			}
			return;
		}

		int num_threads = thread_pool->num_threads;
		int rows_per_thread = (prompt_len + num_threads - 1) / num_threads;

		for (int t = 0; t < num_threads; t++) {
			int start_row = t * rows_per_thread;
			int end_row = start_row + rows_per_thread;

			if (start_row >= prompt_len) {
				break;
			}
			if (end_row > prompt_len) {
				end_row = prompt_len;
			}

			mat_mat_task_t *task = malloc(sizeof(mat_mat_task_t));
			if (!task) {
				fprintf(stderr, "ERROR: Failed to allocate memory for mat_mat_task\n");
				continue;
			}

			*task = (mat_mat_task_t){
				.X = X,
				.W = W,
				.O = O,
				.in_dim = in_dim,
				.out_dim = out_dim,
				.start_row = start_row,
				.end_row = end_row,
				.use_threads = 0 // The sub-task uses a sequential mat-vec
			};

			thread_pool_submit(thread_pool, process_mat_mat_task, task);
		}
		thread_pool_wait(thread_pool);
		return;
	}

	parallel_mat_vec_bf16(X, W, O, in_dim, out_dim, use_threads);
}
