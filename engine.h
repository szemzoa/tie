#ifndef __ENGINE_H__
#define __ENGINE_H__

#include <stdbool.h>
#include "gguf.h"
#include "config.h"

#define ARRAY_SIZE(x) (sizeof(x) / sizeof(x[0]))

#define SILU_TABLE_SIZE (8 * 1024)
#define SILU_X_MIN -10.0f
#define SILU_X_MAX 10.0f

typedef struct {
	int index;
	float score;
} ExpertChoice;

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
	bool is_mmaped;
	size_t size_in_bytes;
} Tensor;

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

	/* gemma3n */
	Tensor altup_correct_coef;
	Tensor altup_correct_scale;
	Tensor altup_predict_coef;
	Tensor altup_router;
	Tensor altup_router_norm;
	Tensor inp_gate;
	Tensor laurel_l;
	Tensor laurel_post_norm;
	Tensor laurel_r;
	Tensor proj;
	Tensor post_norm;
} layer_weights;

typedef struct {
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
	MemType residual_stratch;

	// MoE
	MemType expert_scores;
	MemType *ffn_hidden1_scratch;
	MemType *ffn_hidden2_scratch;
	MemType *expert_outputs;
	MemType expert_out_fp32;

	MemType *altup_hidden_states;
	MemType *altup_predicted_states;
	MemType per_layer_inputs;
} MemLayout;

typedef struct {
	struct ctx_t *ctx;
	int layer_idx;
	int kv_source_layer_idx;
	int batch_len;
	int batch_start_pos;
	AttentionType attn_type;
	int thread_id;

	int token_start_idx;
	int token_end_idx;
	int head_start;
	int head_end;
} attention_worker_task_t;

typedef struct {
	struct ctx_t *ctx;
	int layer_idx;
	int thread_id;
	int expert_idx;
	MemType normed_input;
} expert_task_t;

typedef void (*attention_fn)(void *args);

extern float silu_table[SILU_TABLE_SIZE];

extern void alloc_memtype(MemType *m, ggml_type t, size_t nelems);
extern void free_memtype(MemType *m);
extern void debug_memtype_f32(MemType *mem, char *name, int layer_idx);
extern MemType mem_slice(MemType *buffer, size_t offset_elements);

extern void silu_table_init(void);
extern float silu_lookup(float x);

extern void kv_cache_reset(struct ctx_t *ctx);

extern void rope_cache_init(struct ctx_t *ctx, rope_cache_t *rope_cache, int max_pos, int head_dim, float base, float scale);

extern void attention_worker(void *arg);
extern void attention_worker_gemma3n(void *arg);

extern int transformer_layer_qwen3(struct ctx_t *ctx, int layer_idx, int batch_len);
extern int transformer_layer_gemma3(struct ctx_t *ctx, int layer_idx, int batch_len);
extern int transformer_layer_gemma3n(struct ctx_t *ctx, int layer_idx, int batch_len);

extern void embedding_scale_gemma3(struct ctx_t *ctx, MemType *hidden_state_slice);
extern void dispatch_softcap_logits(MemType *logits, int size, float cap);

extern void prepare_next_token_standard(struct ctx_t *ctx, int next_token);
extern void prepare_next_token_gemma3n(struct ctx_t *ctx, int next_token);

extern void process_prompt_standard(struct ctx_t *ctx, int *prompt_tokens, size_t prompt_len);
extern void process_prompt_gemma3n(struct ctx_t *ctx, int *prompt_tokens, size_t prompt_len);

extern void calculate_per_token_magnitude(float *magnitudes, const MemType *state, size_t num_tokens, int dim);
extern void create_altup_parallel_states(struct ctx_t *ctx, MemType *base_state, size_t prompt_len,
					 Tensor *altup_proj_tensor, MemType *destination);
extern void dispatch_apply_residual_to_buffer(const MemType *src1, const MemType *src2, MemType *dest, int size);
extern void calculate_and_deinterleave_pli_raw(struct ctx_t *ctx, int *prompt_tokens, size_t prompt_len,
					       MemType *dest_buffer);
extern void post_process_altup_states(struct ctx_t *ctx, MemType *final_hidden_state, MemType *final_altup_states,
				      size_t n_tokens);

#endif
