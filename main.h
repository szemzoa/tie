#ifndef __MAIN_H__
#define __MAIN_H__

#include "config.h"
#include "tokenize.h"
#include "gguf.h"
#include "model.h"

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

struct tfmem_t {
	float *hidden_state;
	float *normed_qkv_input;
	float *Q;
	float *K;
	float *V;
	float *attn_scores_buffer[MAX_THREADS];
	float *attn_output;
	float *attn_proj_output;
	float *normed_ffn_input;
	float *gate_proj_output;
	float *up_proj_output;
	float *ffn_down_output;
	float *logits;
};

typedef struct {
	uint16_t *k;
	uint16_t *v;
	int size;
} LayerKVCache;

typedef struct {
	float *sin; // [max_pos][head_dim]
	float *cos;
	int max_pos;
	int head_dim;
} rope_cache_t;

struct ctx_t {
	int fd;
	size_t file_size;
	void *mapped_data;
	uint8_t *fptr;

	uint64_t tensor_count;
	uint64_t tensor_data_offset;
	uint64_t metadata_kv_count;

	struct gguf_metadata_kv_t *metadata;
	gguf_tensor *tensors;

	struct tfmem_t mem;
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
