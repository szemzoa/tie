#ifndef __MATH_SCALAR_H__
#define __MATH_SCALAR_H__

#include "main.h"

extern float bf16_to_fp32(uint16_t bf16);
extern uint16_t fp32_to_bf16(float f);
extern uint16_t fp32_to_bf16_rne(float f);
extern float fp16_to_fp32(uint16_t h);
extern void get_scale_min_k4(int j, const uint8_t *scales, uint8_t *scale, uint8_t *min);

extern void convert_bf16_f32_scalar(const void *src, void *dest, int size);
extern void convert_bf16_bf16_scalar(const void *src, void *dest, int size);
extern void convert_f32_bf16_scalar(const void *src, void *dest, int size);
extern void convert_f32_f32_scalar(const void *src, void *dest, int size);

extern float dot_product_f32_f32_scalar(const void *__restrict a, const void *__restrict b, int size);
extern float dot_product_f32_bf16_scalar(const void *__restrict a, const void *__restrict b, int size);

extern void get_embedding_row_q4k_f32_scalar(const Tensor *W, int row_index, void *dest, int embed_dim);
extern void get_embedding_row_q6k_f32_scalar(const Tensor *W, int row_index, void *dest, int embed_dim);
extern void get_embedding_row_bf16_f32_scalar(const Tensor *W, int row_index, void *dest, int embed_dim);
extern void get_embedding_row_f32_f32_scalar(const Tensor *W, int row_index, void *dest, int embed_dim);
extern void get_embedding_row_q4k_bf16_scalar(const Tensor *W, int row_index, void *dest, int embed_dim);
extern void get_embedding_row_q6k_bf16_scalar(const Tensor *W, int row_index, void *dest, int embed_dim);
extern void get_embedding_row_bf16_bf16_scalar(const Tensor *W, int row_index, void *O, int embed_dim);

extern void apply_residual_f32_f32_scalar(void *acc_void, const void *residual_void, int size);
extern void apply_residual_bf16_bf16_scalar(void *acc_void, const void *residual_void, int size);
extern void apply_residual_f32_bf16_scalar(void *acc_void, const void *residual_void, int size);

extern void rms_norm_f32_f32_f32_scalar(void *__restrict O, const void *__restrict X, const Tensor *__restrict W,
					int size, float eps);
extern void rms_norm_bf16_f32_f32_scalar(void *O, const void *X, const Tensor *W, int size, float eps);
extern void rms_norm_bf16_f32_bf16_scalar(void *O, const void *X, const Tensor *W, int size, float eps);

extern void mat_vec_row_bf16_q4k_f32_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row,
					    int end_row);
extern void mat_vec_row_bf16_q6k_f32_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row,
					    int end_row);
extern void mat_vec_row_bf16_bf16_f32_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row,
					     int end_row);
extern void mat_vec_row_bf16_f32_f32_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row,
					    int end_row);
extern void mat_vec_row_f32_q4k_f32_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row,
					   int end_row);
extern void mat_vec_row_f32_q6k_f32_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row,
					   int end_row);
extern void mat_vec_row_f32_bf16_f32_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row,
					    int end_row);
extern void mat_vec_row_f32_f32_f32_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row,
					   int end_row);
extern void mat_vec_row_bf16_q4k_bf16_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row,
					     int end_row);
extern void mat_vec_row_bf16_q6k_bf16_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row,
					     int end_row);
extern void mat_vec_row_bf16_bf16_bf16_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row,
					      int end_row);
extern void mat_vec_row_f32_q4k_bf16_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row,
					    int end_row);
extern void mat_vec_row_f32_q6k_bf16_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row,
					    int end_row);
extern void mat_vec_row_f32_f16_f32_scalar(const void *X, const void *w_void, void *O, int in_dim, int start_row,
					   int end_row);

extern void accumulate_weighted_V_f32_bf16_scalar(void *O, float weight, const void *V, int size);
extern void accumulate_weighted_V_f32_f32_scalar(void *O, float weight, const void *V, int size);
extern void accumulate_weighted_V_bf16_bf16_scalar(void *O, float weight, const void *V, int size);

extern void swiglu_activation_f32_f32_scalar(void *gate, const void *up, int size);
extern void swiglu_activation_bf16_bf16_scalar(void *gate, const void *up, int size);

extern void store_KV_cache_f32_bf16_scalar(struct TIEContext *ctx, int layer_idx, int start_pos, int batch_len);
extern void store_KV_cache_f32_f32_scalar(struct TIEContext *ctx, int layer_idx, int start_pos, int batch_len);
extern void store_KV_cache_bf16_bf16_scalar(struct TIEContext *ctx, int layer_idx, int start_pos, int batch_len);

extern void apply_rope_cache_f32_scalar(RopeCacheType *rope_cache, void *X, int pos, int head_dim);
extern void apply_rope_cache_bf16_scalar(RopeCacheType *rope_cache, void *X, int pos, int head_dim);
extern void apply_mrope_cache_f32_scalar(RopeCacheType *rope_cache, void *X, int pos, int head_dim);

extern void geglu_activation_f32_f32_scalar(void *gate, const void *up, int size);

extern void dispatch_conv_2d_scalar(MemType *dest, const MemType *src_image, const Tensor *kernel_tensor,
				    const Tensor *bias_tensor, int H_in, int W_in, int stride, int padding);
#endif
