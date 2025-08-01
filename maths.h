#ifndef __MATHS_H__
#define __MATHS_H__

#include "config.h"
#include "gguf.h"
#include "model.h"

typedef struct {
	const float *x;
	const Tensor *W;
	float *o;
	int in_dim;
	int start_row; // The starting output row for this thread
	int end_row;   // The ending output row for this thread
} mat_vec_task_t;

typedef struct {
	const float *X;
	const Tensor *W;
	float *O;
	int in_dim;
	int out_dim;
	int start_row;
	int end_row;
	int use_threads;
} mat_mat_task_t;

typedef struct {
    struct ctx_t* ctx;
    int layer_idx;
    int thread_id;
    int expert_idx;
    float* normed_input;
    float* output_buffer;
} expert_task_t;

extern float bf16_to_fp32(uint16_t bf16);
extern uint16_t fp32_to_bf16(const float src);

extern void dequantize_row_q6_k(const void *__restrict__ q_void, float *__restrict__ y, int k);
extern void dequantize_row_q4_k(const void *__restrict__ q_void, float *__restrict__ y, int k);

extern void rms_norm(float *__restrict o, const float *__restrict x, const float *__restrict weight, int size,
		     float eps);

#ifdef CONFIG_ENABLE_AVX2
extern void apply_rope_cache_avx2(struct ctx_t *ctx, float *x, int pos, int head_dim);
#endif

extern float dot_product_f32(const float *__restrict a, const float *__restrict b, int size);
extern float dot_product_f32_bf16(const float *__restrict a, const uint16_t *__restrict b, int size);

extern void accumulate_weighted_fp32_bf16(float *out, float weight, const uint16_t *v, int size);

extern void parallel_mat_vec_unified(const float *x, const Tensor *w_tensor, float *o, int in_dim, int out_dim,
				     bool use_threads);
extern void parallel_mat_mat_unified(const float *X, const Tensor *W_tensor, float *O, int prompt_len, int in_dim,
				     int out_dim, bool use_threads);

#endif
