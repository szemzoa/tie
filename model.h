#ifndef __MODEL_H__
#define __MODEL_H__

#include <stdint.h>
#include <stdbool.h>
#include "gguf.h"
#include "engine.h"

enum {
	ARCH_UNKNOWN = 0,
	ARCH_QWEN3,
	ARCH_QWEN3_MOE,
	ARCH_QWEN3VL,
	ARCH_GEMMA3,
	ARCH_GEMMA3N,
	ARCH_CLIP_VISION,
};

enum {
	VISION_PROJECTOR_GEMMA3,
	VISION_PROJECTOR_QWEN3VL,
};

struct arch_t {
	const char *name;
	int id;
};

typedef enum {
	FLAG_NONE = 0,
	FLAG_OPTIONAL = 1 << 0,
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

	/* vision */
	SIZE_VISION_IMAGE_RAW,
	SIZE_VISION_PATCH_EMBEDS,
	SIZE_VISION_SEQ_LEN_X_EMBED_DIM,
	SIZE_VISION_SEQ_LEN_X_FFN_DIM,
	SIZE_VISION_SEQ_LEN_X_QKV_DIM_X3,
	SIZE_VISION_POOLED_EMBEDS,
	SIZE_VISION_PROJ_EMBEDS,
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
	bool is_array;
	bool is_optional;
} MetadataDef;

typedef struct {
    void *data;
    uint64_t size;
    uint32_t type;
} ModelArray;

typedef struct {
	size_t offset;
	BufferSizeType size_type;
	GGMLType type;
	TensorLoadFlags flags;
} BufferDef;

typedef struct {
	uint8_t is_moe;
	int sot_token_id;
	int eot_token_id;
	int eos_token_id;
	int newline_token_id;
	int role_user_token_id;
	int role_model_token_id;
	int shared_kv_layers;
	int ffn_dim;
	float final_logit_softcap;

	/* vision */
	int proj_scale_factor;
} ModelParams;

typedef struct {
	int *(*tokenize_prompt)(struct TIEContext *ctx, const char *prompt, size_t *num_tokens);
	void (*process_prompt)(struct TIEContext *ctx, int *prompt_tokens, size_t prompt_len);
//	void (*token_out)(struct TIEContext *ctx, const char *p, int len);
	void (*token_out)(struct TIEContext *ctx, int token_id);
	void (*prepare_next_token)(struct TIEContext *ctx, int next_token);
	void (*embedding_scale)(struct TIEContext *ctx, MemType *hidden_state_slice);
	int (*transformer_layer)(struct TIEContext *ctx, int layer_idx, int batch_len);
} ModelInterface;

typedef struct {
	int token_detect_specials;
	int token_load_merges;
	int token_load_scores;
} TokenizeDef;


typedef struct {
	int arch;
	char *name;

	int projector;

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

	const TokenizeDef *tokenize_defs;

	size_t struct_size;
	size_t layers_offset;
	size_t layer_struct_size;
	size_t num_layers_offset;
} ModelDef;

typedef struct {
	ModelDef *def;

	int is_moe;
	ModelInterface interface;

	int embed_dim;
	int num_heads;
	int num_kv_heads;
	int num_layers;
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
	LayerWeights *layers;
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
	int image_size;	     // e.g. 896
	float image_mean[3]; // e.g. {0.5, 0.5, 0.5}
	float image_std[3];  // e.g. {0.5, 0.5, 0.5}
} ClipVisionMeta;

typedef struct {
	ModelDef *def;

	ModelInterface interface;
	int has_vision_encoder;

	int projection_dim;
	int image_size;
	int patch_size;

	int embed_dim;
	int ffn_dim;
	int num_layers;
	int num_heads;
	float norm_eps;

	int proj_scale_factor;		/* GEMMA3 */
	int spatial_merge_size;		/* QWEN3VL */

	int soi_token_id;
	int eoi_token_id;
	int image_soft_token_id;

	ClipVisionMeta meta;

	ModelArray image_mean;
	ModelArray image_std;

	ModelArray is_deepstack_layers;		/* QWEN3VL */

	Tensor input_projection;
	Tensor soft_embd_norm;
	Tensor patch_embd_bias;
	Tensor patch_embd;
	Tensor position_embd;
	Tensor post_ln_bias;
	Tensor post_ln;

	VisionLayerWeights *layers;
	WeightLayout weight_layout;
} VisionModel;

extern ModelDef *find_model_def(struct GGUFModel *gguf_model);

extern int detect_architecture(const char *model_name);
extern int detect_projector(const char *model_name);

extern int model_load(struct TIEContext *ctx, struct GGUFModel *gguf, void **model, const ModelDef *def, int use_mmap);
extern int model_language_init(struct TIEContext *ctx, Model *model, const ModelDef *def, float yarn_scale_factor,
			       float repetiton_penality);
extern int model_vision_init(struct TIEContext *ctx, VisionModel *model_vision, const ModelDef *def);

extern void model_language_cleanup(struct TIEContext *ctx, struct GGUFModel *model, ModelDef *def, int use_mmap);
extern void model_vision_cleanup(struct TIEContext *ctx, struct GGUFModel *model, ModelDef *def, int use_mmap);
extern void language_model_info(struct TIEContext *ctx);
extern void vision_model_info(struct TIEContext *ctx);

#endif
