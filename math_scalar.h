#ifndef __MATH_SCALAR_H__
#define __MATH_SCALAR_H__

#include "main.h"

extern float bf16_to_fp32(uint16_t bf16);
extern uint16_t fp32_to_bf16(float f);
extern float fp16_to_fp32(uint16_t h);
extern void get_scale_min_k4(int j, const uint8_t *scales, uint8_t *scale, uint8_t *min);

extern void rms_norm_f32_f32_scalar(void *__restrict O, const void *__restrict X, const Tensor *__restrict W, int size,
				    float eps);
extern float dot_product_f32_f32_scalar(const void *__restrict a, const void *__restrict b, int size);
extern float dot_product_f32_bf16_scalar(const void *__restrict a, const void *__restrict b, int size);

extern void get_embedding_row_q4_k_f32_scalar(const Tensor *W, int row_index, void *dest, int embed_dim);
extern void get_embedding_row_q6_k_f32_scalar(const Tensor *W, int row_index, void *dest, int embed_dim);
extern void get_embedding_row_f32_f32_scalar(const Tensor *W, int row_index, void *dest, int embed_dim);
extern void get_embedding_row_bf16_f32_scalar(const Tensor *W, int row_index, void *dest, int embed_dim);

extern void apply_rope_cache_f32_scalar(struct ctx_t *ctx, void *X, int pos, int head_dim);
extern void apply_residual_f32_scalar(void *acc_void, const void *residual_void, int size);

extern void accumulate_weighted_V_f32_bf16_scalar(void *O, float weight, const void *V, int size);
extern void store_KV_cache_f32_bf16_scalar(struct ctx_t *ctx, int layer_idx, int start_pos, int batch_len);

extern void mat_vec_row_q4_k_f32_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row,
					int end_row);
extern void mat_vec_row_q6_k_f32_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row,
					int end_row);
extern void mat_vec_row_bf16_f32_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row,
					int end_row);
extern void mat_vec_row_f32_f32_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row,
				       int end_row);

extern void swiglu_activation_f32_f32_scalar(void *gate, const void *up, int size);

extern void convert_f32_bf16_scalar(const void *src, void *dest, int size);
extern void convert_bf16_f32_scalar(const void *src, void *dest, int size);
extern void convert_bf16_bf16_scalar(const void *src, void *dest, int size);
extern void convert_f32_f32_scalar(const void *src, void *dest, int size);

#endif
