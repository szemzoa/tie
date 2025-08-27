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
typedef void (*apply_rope_cache_fn)(struct ctx_t *ctx, void *X, int pos, int head_dim);
typedef void (*accumulate_weighted_V_fn)(void *O, float W, const void *V, int size);
typedef void (*store_KV_cache_fn)(struct ctx_t *ctx, int layer_idx, int start_pos, int batch_len);
typedef void (*apply_residual_fn)(void *acc, const void *residual, int size);
typedef void (*swiglu_fn)(void *gate, const void *up, int size);
typedef void (*convert_fn)(const void *src, void *dest, int size);
typedef float (*dot_product_fn)(const void *vec_a, const void *vec_b, int size);


typedef struct {
	ggml_type input_type;
	ggml_type output_type;
	embedding_row_fn func;
	int accel;
} embedding_row_dispatch_t;

typedef struct {
	ggml_type input_type;
	ggml_type tensor_type;
	ggml_type output_type;
	rms_norm_fn func;
	int accel;
} rms_norm_dispatch_t;

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
	const void *X;
	const Tensor *W;
	void *O;
	int in_dim;
	int out_dim;
	int start_row;
	int end_row;
	int use_threads;
	mat_vec_fn mat_vec;
	ggml_type X_type; // The data type of the input buffer
	ggml_type O_type; // The data type of the output buffer
	int accel;
} mat_mat_task_t;

typedef struct {
	ggml_type input_type;
	ggml_type tensor_type;
	ggml_type output_type;
	mat_vec_fn mat_vec;
	int accel;
} mat_vec_dispatch_t;

typedef struct {
	ggml_type input_type;
	ggml_type tensor_type;
	ggml_type output_type;
	mat_vec_fn mat_vec;
	int accel;
} mat_mat_dispatch_t;

typedef struct {
	ggml_type input_type;
	apply_rope_cache_fn func;
	int accel;
} apply_rope_cache_dispatch_t;

typedef struct {
	ggml_type output_type;
	ggml_type value_type;
	accumulate_weighted_V_fn func;
	int accel;
} accumulate_weighted_V_dispatch_t;

typedef struct {
	ggml_type input_type;
	ggml_type output_type;
	store_KV_cache_fn func;
	int accel;
} store_KV_cache_dispatch_t;

typedef struct {
	ggml_type input_type;
	ggml_type output_type;
	apply_residual_fn func;
	int accel;
} apply_residual_dispatch_t;

typedef struct {
	ggml_type gate_type;
	ggml_type up_type;
	swiglu_fn func;
	int accel;
} swiglu_activation_dispatch_t;

typedef struct {
	ggml_type input_type;
	ggml_type output_type;
	convert_fn func;
	int accel;
} convert_dispatch_t;

typedef struct {
	ggml_type type_a;
	ggml_type type_b;
	dot_product_fn func;
	int accel;
} dot_product_dispatch_t;

extern void dispatch_embedding_row(const Tensor *W, int row_index, MemType *O_slice, int embed_dim);
extern void dispatch_rms_norm(const MemType *X_slice, const Tensor *W, MemType *O_slice, int size, float eps);
extern void dispatch_mat_vec(const MemType *X, const Tensor *W, MemType *O, int in_dim, int out_dim, int use_threads);
extern void dispatch_mat_mat(const MemType *X, const Tensor *W, MemType *O, int prompt_len, int in_dim, int out_dim,
			     int use_threads);
extern void dispatch_apply_rope_cache(struct ctx_t *ctx, MemType *X_slice, int pos, int head_dim);
extern void dispatch_accumulate_weighted_V(const MemType *V_slice, MemType *O_slice, float weight, int size);
extern void dispatch_store_KV_cache(struct ctx_t *ctx, int layer_idx, int start_pos, int batch_len);
extern void dispatch_apply_residual(MemType *acc, const MemType *residual, int size);
extern void dispatch_swiglu_activation(MemType *gate, MemType *up, int size);
extern void dispatch_convert(const MemType *src, MemType *dest, int size);
extern float dispatch_dot_product(const MemType *vec_a, const MemType *vec_b, int size);

#endif
