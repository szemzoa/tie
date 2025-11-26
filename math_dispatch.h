#ifndef __MATH_DISPATCH_H__
#define __MATH_DISPATCH_H__

#include "main.h"
#include "gguf.h"
#include "threadpool.h"

typedef void (*embedding_row_fn)(const Tensor *W, int row_index, void *O, int embed_dim);
typedef void (*rms_norm_fn)(void *O, const void *X, const Tensor *W, int size, float eps);

typedef void (*mat_vec_fn)(const void *X, const void *W, void *O, int in_dim, int out_dim, int use_threads);
typedef void (*mat_mat_fn)(const void *X, const void *W, void *O, int prompt_len, int in_dim, int out_dim,
			   int use_threads);
typedef void (*apply_rope_cache_fn)(RopeCacheType *rope_cache, void *X, int pos, int head_dim);
typedef void (*apply_mrope_cache_fn)(RopeCacheType *rope_cache, void *X, int pos, int head_dim);
typedef void (*accumulate_weighted_V_fn)(void *O, float W, const void *V, int size);
typedef void (*store_KV_cache_fn)(struct TIEContext *ctx, int layer_idx, int start_pos, int batch_len, int sink_len);
typedef void (*apply_residual_fn)(void *acc, const void *residual, int size);
typedef void (*swiglu_fn)(void *gate, const void *up, int size);
typedef void (*geglu_fn)(void *gate, const void *up, int size);
typedef void (*convert_fn)(const void *src, void *dest, int size);
typedef float (*dot_product_fn)(const void *vec_a, const void *vec_b, int size);
typedef void (*layer_norm_fn)(MemType *dest, const MemType *src, const Tensor *weight, const Tensor *bias, int size,
			      float eps);
typedef void (*conv_2d_fn)(MemType *dest, const MemType *src_image, const Tensor *kernel_tensor,
			   const Tensor *bias_tensor, int H_in, int W_in, int stride, int padding);
typedef void (*transpose_fn)(MemType *dest, const MemType *src, int rows, int cols);

typedef struct {
	GGMLType input_type;
	GGMLType output_type;
	embedding_row_fn func;
	int accel;
} embedding_row_dispatch_t;

typedef struct {
	GGMLType input_type;
	GGMLType tensor_type;
	GGMLType output_type;
	rms_norm_fn func;
	int accel;
} rms_norm_dispatch_t;

typedef struct {
	GGMLType input_type;
	GGMLType output_type;
	GGMLType tensor_type;
	GGMLType bias_type;
	layer_norm_fn func;
	int accel;
} layer_norm_dispatch_t;

typedef struct {
	const void *X;
	const Tensor *W;
	void *O;
	int in_dim;
	int start_row; // The starting output row for this thread
	int end_row;   // The ending output row for this thread
	mat_vec_fn mat_vec;
	int accel;
} mat_vec_task_t;

typedef struct {
	struct TIEContext *ctx;
	const void *X;
	const Tensor *W;
	void *O;
	int in_dim;
	int out_dim;
	int start_row;
	int end_row;
	int use_threads;
	mat_vec_fn mat_vec;
	GGMLType X_type; // The data type of the input buffer
	GGMLType O_type; // The data type of the output buffer
	int accel;
} mat_mat_task_t;

typedef struct {
	GGMLType input_type;
	GGMLType tensor_type;
	GGMLType output_type;
	mat_vec_fn mat_vec;
	int accel;
} mat_vec_dispatch_t;

typedef struct {
	GGMLType input_type;
	GGMLType tensor_type;
	GGMLType output_type;
	mat_vec_fn mat_vec;
	int accel;
} mat_mat_dispatch_t;

typedef struct {
	GGMLType input_type;
	apply_rope_cache_fn func;
	int accel;
} apply_rope_cache_dispatch_t;

typedef struct {
	GGMLType input_type;
	apply_mrope_cache_fn func;
	int accel;
} apply_mrope_cache_dispatch_t;

typedef struct {
	GGMLType output_type;
	GGMLType value_type;
	accumulate_weighted_V_fn func;
	int accel;
} accumulate_weighted_V_dispatch_t;

typedef struct {
	GGMLType input_type;
	GGMLType output_type;
	store_KV_cache_fn func;
	int accel;
} store_KV_cache_dispatch_t;

typedef struct {
	GGMLType input_type;
	GGMLType output_type;
	apply_residual_fn func;
	int accel;
} apply_residual_dispatch_t;

typedef struct {
	GGMLType gate_type;
	GGMLType up_type;
	swiglu_fn func;
	int accel;
} swiglu_activation_dispatch_t;

typedef struct {
	GGMLType gate_type;
	GGMLType up_type;
	geglu_fn func;
	int accel;
} geglu_activation_dispatch_t;

typedef struct {
	GGMLType input_type;
	GGMLType output_type;
	convert_fn func;
	int accel;
} convert_dispatch_t;

typedef struct {
	GGMLType type_a;
	GGMLType type_b;
	dot_product_fn func;
	int accel;
} dot_product_dispatch_t;

typedef struct {
	GGMLType output_type;
	GGMLType input_type;
	GGMLType kernel_type;
	GGMLType bias_type;
	conv_2d_fn func;
	int accel;
} conv_2d_dispatch_t;

typedef struct {
	GGMLType type;
	transpose_fn func;
	int accel;
} transpose_dispatch_t;


extern void dispatch_embedding_row(const Tensor *W, int row_index, MemType *O_slice, int embed_dim);
extern void dispatch_rms_norm(const MemType *X_slice, const Tensor *W, MemType *O_slice, int size, float eps);
extern void dispatch_mat_vec(struct TIEContext *ctx, const MemType *X, const Tensor *W, MemType *O, int in_dim,
			     int out_dim, int use_threads);
extern void dispatch_mat_mat(struct TIEContext *ctx, const MemType *X, const Tensor *W, MemType *O, int batch_len,
			     int in_dim, int out_dim, int use_threads);
extern void dispatch_apply_rope_cache(RopeCacheType *rope_cache, MemType *X_slice, int pos, int head_dim);
extern void dispatch_apply_mrope_cache(RopeCacheType *rope_cache, MemType *X_slice, int pos, int head_dim);
extern void dispatch_accumulate_weighted_V(const MemType *V_slice, MemType *O_slice, float weight, int size);
extern void dispatch_store_KV_cache(struct TIEContext *ctx, int layer_idx, int start_pos, int batch_len, int sink_len);
extern void dispatch_apply_residual(MemType *acc, const MemType *residual, int size);
extern void dispatch_swiglu_activation(MemType *gate, MemType *up, int size);
extern void dispatch_geglu_activation(MemType *gate, MemType *up, int size);
extern void dispatch_convert(const MemType *src, MemType *dest, int size);
extern float dispatch_dot_product(const MemType *vec_a, const MemType *vec_b, int size);
extern void dispatch_layer_norm(MemType *dest, const MemType *src, const Tensor *weight, const Tensor *bias, int size,
				float eps);
extern void dispatch_conv_2d(MemType *dest, const MemType *src_image, const Tensor *kernel_tensor,
			     const Tensor *bias_tensor, int H_in, int W_in, int stride, int padding);
extern void dispatch_transpose(MemType *dest, const MemType *src, int rows, int cols);
extern void math_dispatch_init(void);

#endif
