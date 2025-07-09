#ifndef __MODEL_H__
#define __MODEL_H__

typedef struct {
	uint16_t *attn_q;
	uint16_t *attn_k;
	uint16_t *attn_v;
	uint16_t *attn_out;
	uint16_t *ffn_gate;
	uint16_t *ffn_up;
	uint16_t *ffn_down;
	float	 *attn_norm;
	float	 *attn_q_norm;
	float	 *attn_k_norm;
	float	 *ffn_norm;
} LayerWeights;

typedef struct {
	int		 embed_dim;
	int		 num_layers;
	int		 num_heads;
	int		 num_kv_heads;
	int		 head_dim;
	int		 ffn_dim;
	float	 rope_freq_base;
	float	 norm_eps;
	uint64_t vocab_size;
	uint64_t merges_size;
	int		 seq_length;
	int		 eos_token;

	float yarn_scale_factor;
	float repetition_penalty;

	float *token_embd;
	float *output_norm;

	LayerWeights *layers;

	float attn_scale;
} Qwen3Model;

extern int	model_create(struct ctx_t *ctx);
extern int	compare_weights(char *filename, int file_size, int py_offset, int size, float *c_weights);
extern void model_cleanup(struct ctx_t *ctx);
extern int	model_init(struct ctx_t *ctx, float yarn_scale_factor, float repetiton_penality);

#endif
