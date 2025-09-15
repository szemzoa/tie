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
	MemType mem;
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

	// Gemma3
	MemType residual_stratch;

	// MoE
	MemType expert_scores; // buffer to hold the output of the router.
	MemType *ffn_hidden1_scratch;
	MemType *ffn_hidden2_scratch;
	MemType *expert_outputs;
	MemType expert_out_fp32;
};

typedef enum { ATTN_TYPE_GLOBAL, ATTN_TYPE_LOCAL } AttentionType;

typedef struct {
	MemType k;
	MemType v;
} LayerKVCache;

typedef enum { LAYOUT_ROW_MAJOR, LAYOUT_COL_MAJOR } WeightLayout;

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

	/* gemma3 */
	Tensor post_attn_norm;
	Tensor post_ffw_norm;
} layer_weights;

enum {
	ARCH_UNKNOWN = 0,
	ARCH_QWEN3,
	ARCH_GEMMA3,
};

typedef struct model_interface_t {
	int *(*tokenize_prompt)(struct ctx_t *ctx, const char *input_buf, size_t *num_tokens);
	void (*token_out)(struct ctx_t *ctx, const char *p, int len);
	void (*embedding_scale)(struct ctx_t *ctx, MemType *hidden_state_slice);
	void (*activation)(MemType *gate, MemType *up, int size);
} model_interface_t;

typedef struct {
	char *arch_name;

	int arch;
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

	int sot_token;		/* start of turn */
	int eot_token;		/* end of turn */
	int eos_token;		/* end of seq */
	int bos_token;		/* begin of seq */
	int unk_token;
	int pad_token;
	int add_bos_token;
	int add_eos_token;
	int bos_token_sent;

	int role_user_token;
	int role_model_token;
	int newline_token;


	float yarn_scale_factor;

	float rope_scale_factor;      // GEMMA3
	uint32_t attn_sliding_window; // GEMMA3

	float repetition_penalty;
	float attn_scale;

	int expert_count;
	int expert_used_count;
	int expert_ffn_dim;

	Tensor token_embd;
	Tensor output_norm;
	layer_weights *layers;
	Tensor output;

	WeightLayout weight_layout;
} Model;

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

	Model *model;
	model_interface_t interface;

	struct context_memory_t mem;

	uint32_t kv_pos;
	LayerKVCache *kv_cache;

	rope_cache_t *rope_cache_local;
	rope_cache_t *rope_cache_global;

	unsigned int utf8_state;
	unsigned int utf8_codepoint;

	Tokenizer tokenizer;
};

extern void token_out_qwen3(struct ctx_t *ctx, const char *p, int len);
extern void token_out_gemma3(struct ctx_t *ctx, const char *p, int len);

#endif
