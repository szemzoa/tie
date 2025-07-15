#ifndef __MODEL_H__
#define __MODEL_H__

#include "gguf.h"

typedef struct {
	enum ggml_type type; // The data type (Q6_K, BF16, etc.)
	void *data;	      // Can point to mmap'd region or malloc'd buffer
	bool is_mmaped;	      // Flag to know if we need to free() it
	size_t size_in_bytes; // Total size of the tensor data
} Tensor;

typedef struct {
	Tensor attn_norm;
	Tensor attn_q;
	Tensor attn_k;
	Tensor attn_v;
	Tensor attn_q_norm;
	Tensor attn_k_norm;
	Tensor attn_out;
	Tensor ffn_norm;
	Tensor ffn_gate;
	Tensor ffn_up;
	Tensor ffn_down;
} layer_weights;

typedef struct {
	int embed_dim;
	int num_layers;
	int num_heads;
	int num_kv_heads;
	int head_dim;
	int ffn_dim;
	float rope_freq_base;
	float norm_eps;
	uint64_t vocab_size;
	uint64_t merges_size;
	int seq_length;
	int eos_token;

	float yarn_scale_factor;
	float repetition_penalty;
	float attn_scale;

	Tensor token_embd;
	Tensor output_norm;
	layer_weights *layers;

} Qwen3Model;

extern int model_create(struct ctx_t *ctx, int use_mmap);
extern int compare_weights(char *filename, int file_size, int py_offset, int size, float *c_weights);
extern void model_cleanup(struct ctx_t *ctx, int use_mmap);
extern int model_init(struct ctx_t *ctx, float yarn_scale_factor, float repetiton_penality);

#endif
