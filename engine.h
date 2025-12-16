#ifndef __ENGINE_H__
#define __ENGINE_H__

#include <stdbool.h>
#include "gguf.h"
#include "config.h"

#define ARRAY_SIZE(x) (sizeof(x) / sizeof(x[0]))

#define SILU_TABLE_SIZE (8 * 1024)
#define SILU_X_MIN -10.0f
#define SILU_X_MAX 10.0f

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

typedef struct {
	int index;
	float score;
} ExpertChoice;

typedef struct {
	float *sin; // [max_pos][head_dim]
	float *cos;
	int max_pos;
	int head_dim;
	int rope_dim;
} RopeCacheType;

typedef struct {
	float *cos_table;    // [seq_len * head_dim] (e.g., 2304 * 64)
	float *sin_table;    // [seq_len * head_dim] (e.g., 2304 * 64)
	size_t num_elements; // e.g., 2304 * 64
} MRopeCacheType;

typedef struct {
	GGMLType type;
	size_t n_bytes;
	void *data;
} MemType;

typedef struct {
	MemType mem;
	bool is_mmaped;
	size_t size_in_bytes;
	uint64_t dimensions[4];
} Tensor;

typedef enum { ATTN_TYPE_GLOBAL, ATTN_TYPE_LOCAL } AttentionType;

typedef struct {
	MemType k;
	MemType v;
} LayerKVCache;

typedef enum { LAYOUT_ROW_MAJOR, LAYOUT_COL_MAJOR } WeightLayout;

enum {
	LAYER_TYPE_ATTENTION = 0,
	LAYER_TYPE_MAMBA2,
	LAYER_TYPE_SHORTCONV,
};

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

	/* MoE */
	Tensor ffn_gate_inp;
	Tensor ffn_gate_exps;
	Tensor ffn_up_exps;
	Tensor ffn_down_exps;

	/* Gemma3 */
	Tensor post_attn_norm;
	Tensor post_ffw_norm;

	/* Gemma-3n */
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

	/* Granite-Hybrid */
	Tensor ssm_a;
	Tensor ssm_conv1d_bias;
	Tensor ssm_conv1d_weight;
	Tensor ssm_d;
	Tensor ssm_dt_bias;
	Tensor ssm_in;
	Tensor ssm_norm;
	Tensor ssm_out;
	Tensor ffn_gate_shexps;
	Tensor ffn_up_shexps;
	Tensor ffn_down_shexps;

	/* LFM2 */
	Tensor sconv_conv;
	Tensor sconv_bias;
	Tensor sconv_in_proj;
	Tensor sconv_out_proj;
	Tensor exp_probs_b_bias;

	/* GLM4 */
	Tensor attn_k_bias;
	Tensor attn_q_bias;
	Tensor attn_v_bias;

} LayerWeights;

typedef struct {
	Tensor ln1_bias;
	Tensor ln1;
	Tensor ln2_bias;
	Tensor ln2;
	Tensor ffn_up_bias;
	Tensor ffn_up;
	Tensor ffn_down_bias;
	Tensor ffn_down;
	Tensor attn_k_bias;
	Tensor attn_k;
	Tensor attn_q_bias;
	Tensor attn_q;
	Tensor attn_v_bias;
	Tensor attn_v;
	Tensor attn_out_bias;
	Tensor attn_out;

	/* Qwen3-VL */
	Tensor attn_qkv_bias;
	Tensor attn_qkv;
	Tensor ds_fc1_bias;
	Tensor ds_fc1_weight;
	Tensor ds_fc2_bias;
	Tensor ds_fc2_weight;
	Tensor ds_norm_bias;
	Tensor ds_norm_weight;
} VisionLayerWeights;

// Mamba-2 State Definition
// Conv State: [conv_kernel, dim_inner] ring buffer
// SSM State:  [dim_inner, state_size] or [n_heads, head_dim, state_size]
typedef struct {
	float *conv_state;
	float *ssm_state;
	int conv_pos;		// ring buffer
} LayerMambaState;

typedef struct {
	MemType shared_expert_output;
} LayerSharedExpertState;

typedef struct {
	float *buffer; // ring buffer data
	int pos;       // write position
} LayerConvState;

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

	/* Gemma-3n */
	MemType *altup_hidden_states;
	MemType *altup_predicted_states;
	MemType per_layer_inputs;

	/* Qwen3-VL */
	MemType pos_ids;

	/* Granite-Hybrid */
	LayerMambaState *mamba_states;	  // Array of structs, one per layer
	MemType mamba_conv_states_memory; // The raw memory blocks
	MemType mamba_ssm_states_memory;
	MemType shared_expert_output;
	MemType mamba_in_proj_output; 

	/* LFM2 */
	MemType sconv_in_proj_output;
	MemType sconv_conv_output;
	LayerConvState *lfm2_conv_states;
} MemLayout;

typedef struct {
	MemType q_head;
	MemType k_head;
	MemType v_head;
	MemType v_head_t;
	MemType scores;
	MemType output_head;
} VisionAttnScratch;

typedef struct {
	MemType image_raw;
	MemType patch_embeds;
	MemType hidden_state;
	MemType QKV_fused;
	MemType Q;
	MemType K;
	MemType V;
	MemType residual_scratch;
	MemType normed_input;
	MemType qkv_proj_output;
	float *attn_scores_buffer[MAX_THREADS];
	MemType attn_output;
	MemType attn_proj_output;
	MemType ffn_up_output;
	MemType ffn_down_output;
	MemType pooled_embeddings;
	MemType projected_embeddings;
	VisionAttnScratch *attn_scratch;

	int image_raw_width;
	int image_raw_height;

	/* Qwen3-VL */
	MemType merger_norm_buf;
	MemType merger_fc1_buf;
	MemType *deepstack_features;
} MemLayoutVision;

typedef struct {
	struct TIEContext *ctx;
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
	int sink_len;
} attention_worker_task_t;

typedef struct {
	struct TIEContext *ctx;
	int layer_idx;
	int thread_id;
	int expert_idx;
	MemType normed_input;
} expert_task_t;

typedef void (*attention_fn)(void *args);

extern float silu_table[SILU_TABLE_SIZE];


extern void engine_alloc(struct TIEContext *ctx, int num_threads);
extern void engine_release(struct TIEContext *ctx);

extern int compare_tensor_with_file(const char *ref_filename, const float *your_c_tensor, long num_elements,
				    float tolerance, int debug_offset);
extern void debug_memtype_f32(MemType *mem, char *name, int layer_idx, int debug_offset);
extern int load_tensor_from_file(const char *ref_filename, float *dest_tensor, long num_elements);

extern void *xaligned_alloc(size_t alignment, size_t size_bytes);
extern void alloc_memtype(MemType *m, GGMLType t, size_t nelems);
extern void free_memtype(MemType *m);
extern MemType mem_slice(MemType *buffer, size_t offset_elements);

extern void silu_table_init(void);
extern float silu_lookup(float x);

extern void kv_cache_reset(struct TIEContext *ctx);

extern void rope_cache_init(struct TIEContext *ctx, RopeCacheType *rope_cache, int max_pos, int head_dim, int rope_dim,
			    float base, float scale);
extern void build_mrope_position_ids(struct TIEContext *ctx, const int *prompt_tokens, size_t prompt_len,
				     bool has_image, int start_pos, int h_patches_in, int w_patches_in);
extern void text_rope_cache_init(struct TIEContext *ctx, int seq_len, int start_pos);

extern void dispatch_gelu_inplace(MemType *tensor, int size);
extern void dispatch_softcap_logits(MemType *logits, int size, float cap);

extern void prepare_next_token_standard(struct TIEContext *ctx, int next_token);
extern void process_embeddings(struct TIEContext *ctx, MemType *embeddings, size_t n_tokens);

extern void softmax(float *x, int size);
extern void attention(struct TIEContext *ctx, int batch_len, int layer_idx, int kv_source_layer_idx, int start_pos,
		      AttentionType attn_type, attention_fn worker, int sink_len);
extern void attention_worker(void *arg);

extern void process_expert_task(void *arg);
extern void find_top_k(const float *router_logits, int expert_count, int k, ExpertChoice *top_k);

extern void process_prompt_gemma3n(struct TIEContext *ctx, int *prompt_tokens, size_t prompt_len);
extern void prepare_next_token_gemma3n(struct TIEContext *ctx, int next_token);
extern void embedding_scale_granite(struct TIEContext *ctx, MemType *hidden_state_slice);
extern void embedding_scale_gemma3(struct TIEContext *ctx, MemType *hidden_state_slice);
extern void calculate_per_token_magnitude(float *magnitudes, const MemType *state, size_t num_tokens, int dim);
extern void create_altup_parallel_states(struct TIEContext *ctx, MemType *base_state, size_t prompt_len,
					 Tensor *altup_proj_tensor, MemType *destination);
extern void dispatch_apply_residual_to_buffer(const MemType *src1, const MemType *src2, MemType *dest, int size);
extern void calculate_and_deinterleave_pli_raw(struct TIEContext *ctx, int *prompt_tokens, size_t prompt_len,
					       MemType *dest_buffer);
extern void post_process_altup_states(struct TIEContext *ctx, MemType *final_hidden_state, MemType *final_altup_states,
				      size_t n_tokens);

#endif
