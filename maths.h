#ifndef __MATHS_H__
#define __MATHS_H__

#include "config.h"

typedef struct {
	const float *x;
	const float *w;
	float *o;
	int in_dim;
	int start_row; // The starting output row for this thread
	int end_row;   // The ending output row for this thread
} mat_vec_task_t;

typedef struct {
	const float *X;
	const uint16_t *W;
	float *O;
	int in_dim;
	int out_dim;
	int start_row;
	int end_row;
	int use_threads; // To pass down to the mat_vec function
} mat_mat_task_t;

typedef struct {
	const float *x;
	const uint16_t *w_bf16;
	float *o;
	int in_dim;
	int start_row; // The starting output row for this thread
	int end_row;   // The ending output row for this thread
} mat_vec_task_bf16_t;

extern inline float bf16_to_fp32(uint16_t bf16);
extern float *convert_bf16_to_f32(void *bf16_ptr, size_t count);

#ifdef CONFIG_ENABLE_AVX2
#include <immintrin.h>
extern void rms_norm_avx2(float *__restrict o, const float *__restrict x, const float *__restrict weight, int size,
			  float eps);
extern inline __m256 load_bf16_as_f32(const uint16_t *src);
extern void mat_vec_bf16_avx2_row(const float *__restrict x, const uint16_t *__restrict w_bf16, float *__restrict o,
				  int in_dim, int start_row, int end_row);
extern void mat_vec_bf16_avx2(const float *__restrict x, const uint16_t *__restrict w_bf16, float *__restrict o,
			      int in_dim, int out_dim);
extern float dot_product_f32_avx2(const float *__restrict a, const float *__restrict b, int size);
extern void mat_vec_avx2(const float *x, const float *w, float *o, int in_dim, int out_dim);
extern void mat_vec_avx2_row(const float *x, const float *w, float *o, int in_dim, int start_row, int end_row);
#endif

extern void mat_vec_bf16_row(const float *__restrict x, const uint16_t *__restrict w_bf16, float *__restrict o,
			     int in_dim, int start_row, int end_row);
extern void mat_vec_bf16(const float *__restrict x, const uint16_t *__restrict w_bf16, float *__restrict o, int in_dim,
			 int out_dim);
extern void mat_vec_row(const float *__restrict x, const float *__restrict w, float *__restrict o, int in_dim,
			int start_row, int end_row);
extern void mat_vec(const float *__restrict x, const float *__restrict w, float *__restrict o, int in_dim, int out_dim);
extern float dot_product_f32(const float *__restrict a, const float *__restrict b, int size);

extern void parallel_mat_vec_bf16(const float *x, const uint16_t *w_bf16, float *o, int in_dim, int out_dim,
				  bool use_threads);
extern void parallel_mat_vec(const float *x, const float *w, float *o, int in_dim, int out_dim, bool use_threads);
extern void parallel_mat_mat_bf16(const float *X, const uint16_t *W, float *O, int prompt_len, int in_dim, int out_dim,
				  int use_threads);

#endif
