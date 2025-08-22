#ifndef __MATH_AVX2_H__
#define __MATH_AVX2_H__

#include "main.h"

extern void rms_norm_f32_f32_avx2(void *__restrict O, const void *__restrict X, const Tensor *__restrict W, int size,
				  float eps);
extern void rms_norm_bf16_f32_avx2(void *__restrict o_void, const void *__restrict x_void, const Tensor *__restrict W,
				   int size, float eps);

extern void apply_rope_cache_f32_avx2(struct ctx_t *ctx, void *X, int pos, int head_dim);

extern void apply_residual_f32_avx2(void *acc_void, const void *residual_void, int size);

extern void mat_vec_row_q4_k_f32_avx2(const void *X, const void *w_void, void *O, int in_dim, int start_row,
				      int end_row);
extern void mat_vec_row_q6_k_f32_avx2(const void *X, const void *w_void, void *O, int in_dim, int start_row,
				      int end_row);

extern void mat_vec_row_bf16_f32_avx2(const void *__restrict X, const void *__restrict w_void, void *__restrict O,
				      int in_dim, int start_row, int end_row);
extern void mat_vec_row_f32_f32_avx2(const void *X, const void *w_void, void *O, int in_dim, int start_row,
				     int end_row);

extern void dequantize_row_q4_k_avx2(const void *__restrict__ q_void, float *__restrict__ y, int k);
extern void dequantize_row_q6_k_avx2(const void *__restrict__ q_void, float *__restrict__ y, int k);

extern void accumulate_weighted_V_f32_bf16_avx2(void *__restrict O, float weight, const void *__restrict V, int size);

extern float dot_product_q4_k_avx2(const float *x, const block_q4_k *blk);
extern float dot_product_q6_k_avx2(const float *x, const block_q6_k *block);
extern float dot_product_f32_bf16_avx2(const void *__restrict a, const void *__restrict b, int size);
extern float dot_product_f32_f32_avx2(const void *__restrict a, const void *__restrict b, int size);

extern void get_embedding_row_q4_k_f32_avx2(const Tensor *W, int row_index, void *dest, int embed_dim);
extern void get_embedding_row_q6_k_f32_avx2(const Tensor *W, int row_index, void *dest, int embed_dim);

extern void store_KV_cache_f32_bf16_avx2(struct ctx_t *ctx, int layer_idx, int start_pos, int batch_len);
extern void convert_f32_bf16_avx2(const void *src, void *dest, int size);

extern void swiglu_activation_f32_f32_avx2(void *gate_void, const void *up_void, int size);

#endif
