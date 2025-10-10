#ifndef __MODEL_H__
#define __MODEL_H__

#include <stdint.h>
#include <stdbool.h>
#include "gguf.h"
#include "engine.h"

enum {
	ARCH_UNKNOWN = 0,
	ARCH_QWEN3,
	ARCH_GEMMA3,
	ARCH_GEMMA3N,
};

typedef enum {
	FLAG_NONE = 0,
	FLAG_OPTIONAL = 1 << 0,
	FLAG_DENSE_ONLY = 1 << 1,
	FLAG_MOE_ONLY = 1 << 2,
} TensorLoadFlags;

typedef enum {
	SIZE_EMBED_DIM,
	SIZE_FFN_DIM,
	SIZE_Q_DIM,
	SIZE_KV_DIM,
	SIZE_VOCAB_SIZE,
	//	SIZE_PLI_DIM,
	//	SIZE_HEAD_DIM,
	SIZE_NUM_LAYERS_X_PLI_DIM, // For the full per_layer_inputs buffer
} BufferSizeType;

typedef struct {
	const char *name_fmt;
	size_t offset;
	TensorLoadFlags flags;
} TensorDef;

typedef struct {
	const char *key_fmt;
	enum gguf_metadata_value_type type;
	size_t offset;
	bool is_optional;
} MetadataDef;

typedef struct {
	size_t offset;
	BufferSizeType size_type;
	ggml_type type;
	TensorLoadFlags flags;
} BufferDef;

typedef struct {
	int sot_token_id;
	int eot_token_id;
	int eos_token_id;
	int newline_token_id;
	int role_user_token_id;
	int role_model_token_id;
	int shared_kv_layers;
	int ffn_dim;
	float final_logit_softcap;
} ModelParams;

typedef struct {
	int *(*tokenize_prompt)(struct ctx_t *ctx, const char *prompt, size_t *num_tokens);
	void (*process_prompt)(struct ctx_t *ctx, int *prompt_tokens, size_t prompt_len);
	void (*token_out)(struct ctx_t *ctx, const char *p, int len);
	void (*prepare_next_token)(struct ctx_t *ctx, int next_token);
	void (*embedding_scale)(struct ctx_t *ctx, MemType *hidden_state_slice);
	int (*transformer_layer)(struct ctx_t *ctx, int layer_idx, int batch_len);
} ModelInterface;

typedef struct {
	char *arch_name;
	int arch;
	char *name;
	int is_moe;

	ModelInterface interface;

	int embed_dim;
	int num_layers;
	int num_heads;
	int num_kv_heads;
	int shared_kv_layers;
	int head_dim;
	int ffn_dim;
	float rope_freq_base;
	float norm_eps;
	uint64_t vocab_size;
	uint64_t merges_size;
	int seq_length;

	int sot_token_id; /* start of turn */
	int eot_token_id; /* end of turn */
	int eos_token_id; /* end of seq */
	int bos_token_id; /* begin of seq */
	int unk_token_id;
	int pad_token_id;
	int role_user_token_id;
	int role_model_token_id;
	int newline_token_id;
	int add_bos_token;
	int add_eos_token;
	int bos_token_sent;

	float yarn_scale_factor;
	float rope_scale_factor;
	uint32_t attn_sliding_window;

	float repetition_penalty;
	float attn_scale;

	int expert_count;
	int expert_used_count;
	int expert_ffn_dim;

	Tensor token_embd;
	Tensor output_norm;
	Tensor output;
	layer_weights *layers;
	WeightLayout weight_layout;

	/* gemma3n */
	int pli_dim;
	uint32_t altup_num_inputs;
	Tensor altup_proj;
	Tensor altup_unembd_proj;
	Tensor per_layer_model_proj;
	Tensor per_layer_proj_norm;
	Tensor per_layer_token_embd;
	float final_logit_softcap;
} Model;

typedef struct {
	uint8_t arch;
	char *name;

	ModelParams params;
	ModelInterface interface;

	const MetadataDef *metadata_defs;
	size_t num_metadata_defs;

	const TensorDef *global_tensors;
	size_t num_global_tensors;

	const TensorDef *layer_tensors;
	size_t num_layer_tensors;

	const BufferDef *buffer_defs;
	size_t num_buffer_defs;
} ModelDef;

extern int model_load(struct ctx_t *ctx, int use_mmap, int context_length);
extern int model_init(struct ctx_t *ctx, float yarn_scale_factor, float repetiton_penality);
extern void model_cleanup(struct ctx_t *ctx, int use_mmap);

#endif
