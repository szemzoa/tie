#ifndef __MATHS_H__
#define __MATHS_H__

#include "config.h"
#include "gguf.h"
#include "model.h"

typedef struct {
	const float *x;
	const float *w;
	float *o;
	int in_dim;
	int start_row; // The starting output row for this thread
	int end_row;   // The ending output row for this thread
} mat_vec_task_fp32_t;

typedef struct {
	const float *X;
	const Tensor *W;
	float *O;
	int in_dim;
	int out_dim;
	int start_row;
	int end_row;
	int use_threads; // To pass down to the mat_vec function
} mat_mat_task_t;

typedef struct {
	const float *x;
	const Tensor *W;
	float *o;
	int in_dim;
	int start_row; // The starting output row for this thread
	int end_row;   // The ending output row for this thread
} mat_vec_task_t;


extern float bf16_to_fp32(uint16_t bf16);
extern uint16_t fp32_to_bf16(const float src);

extern float *convert_bf16_to_f32(void *bf16_ptr, size_t count);
extern void dequantize_row_q6_k(const void *__restrict__ q_void, float *__restrict__ y, int k);

extern void rms_norm(float *__restrict o, const float *__restrict x, const float *__restrict weight, int size,
					 float eps);
extern float dot_product_f32(const float *__restrict a, const float *__restrict b, int size);
extern float dot_product_f32_bf16(const float *__restrict a, const uint16_t *__restrict b, int size);
extern void accumulate_weighted_fp32_bf16(float *out, float weight, const uint16_t *v, int size);

extern void parallel_mat_vec_fp32(const float *x, const float *w, float *o, int in_dim, int out_dim, bool use_threads);
extern void parallel_mat_vec_unified(const float *x, const Tensor *w_tensor, float *o, int in_dim, int out_dim,
				     bool use_threads);
extern void parallel_mat_mat_unified(const float *X, const Tensor *W_tensor, float *O, int prompt_len, int in_dim,
				     int out_dim, bool use_threads);

#endif
