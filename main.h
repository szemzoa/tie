#ifndef __MAIN_H__
#define __MAIN_H__

#include <stddef.h>
#include "config.h"
#include "tokenize.h"
#include "gguf.h"

#define ARRAY_SIZE(x) (sizeof(x) / sizeof(x[0]))

enum {
	TOOL_CALL_STATE_UNINITIALIZED = 0,
	TOOL_CALL_STATE_IDLE,
	TOOL_CALL_STATE_PROCESSING,
	TOOL_CALL_STATE_END,
};

struct tool_call_t {
	int state;
	int token_start;
	int token_end;
	char buffer[TOOL_CALL_BUFFER_SIZE];
	int len;
	char *result;
};

typedef char *(*tool_func_t)(const char *);

struct tool_entry_t {
	const char *name;
	tool_func_t func;
};

typedef struct {
	float *sin; // [max_pos][head_dim]
	float *cos;
	int max_pos;
	int head_dim;
} rope_cache_t;

typedef struct {
	ggml_type type;
	void *data;
} MemType;

typedef struct {
	MemType	mem;
	bool is_mmaped;	      // Flag to know if we need to free() it
	size_t size_in_bytes; // Total size of the tensor data
} Tensor;

struct context_memory_t {
	MemType hidden_state;
	MemType normed_qkv_input;
	MemType Q;
	MemType K;
	MemType V;
	MemType attn_output;
	MemType attn_proj_output;
	MemType normed_ffn_input;
	MemType gate_proj_output;
	MemType up_proj_output;
	MemType ffn_down_output;
	MemType logits;
	MemType *q_head_fp32_scratch;

	float *attn_scores_buffer[MAX_THREADS];

	// MoE
	MemType expert_scores; // buffer to hold the output of the router.
	MemType *ffn_hidden1_scratch;
	MemType *ffn_hidden2_scratch;
	MemType *expert_outputs;
	MemType expert_out_fp32;
};

typedef struct {
	MemType k;
	MemType v;
} LayerKVCache;

typedef struct {
	Tensor attn_norm;
	Tensor attn_q;
	Tensor attn_k;
	Tensor attn_v;
	Tensor attn_q_norm;
	Tensor attn_k_norm;
	Tensor attn_out;
	Tensor ffn_norm;

	/* dense */
	Tensor ffn_gate;
	Tensor ffn_up;
	Tensor ffn_down;

	/* MoE */
	Tensor ffn_gate_inp;
	Tensor ffn_gate_exps;
	Tensor ffn_up_exps;
	Tensor ffn_down_exps;
} layer_weights;

typedef struct {
	char *arch_name;
	int is_moe;
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

	int expert_count;
	int expert_used_count;
	int expert_ffn_dim;

	Tensor token_embd;
	Tensor output_norm;
	layer_weights *layers;
	Tensor output;
} Qwen3Model;

struct ctx_t {
	int fd;
	size_t file_size;
	void *mapped_data;
	uint8_t *fptr;

	uint64_t tensor_count;
	uint64_t tensor_loaded;
	uint64_t tensor_data_offset;
	uint64_t metadata_kv_count;
	struct gguf_metadata_kv_t *metadata;
	gguf_tensor *tensors;

	struct context_memory_t mem;
	LayerKVCache *kv_cache;
	rope_cache_t *rope_cache;

	Qwen3Model *model;

	uint32_t kv_pos;

	TrieNode *root;
	StringPool *pool;

	unsigned char **token_table; // Points to each token string in pool->data
	int *token_lens;	     // Length of each token
	int token_count;	     // Total number of tokens

	unsigned int utf8_state;
	unsigned int utf8_codepoint;
};

#endif
