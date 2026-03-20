#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "model.h"
#include "model_defs.h"
#include "engine.h"
#include "vision.h"
#include "tokenize.h"
#include "tools.h"
#include "math_dispatch.h"
#include "main.h"

int transformer_layer_qwen35(struct TIEContext *ctx, int layer_idx, int batch_len);
int build_prompt_qwen35(struct TIEContext *ctx, int *prompt_tokens, int *user_text_tokens, int user_text_token_count,
			 bool has_image);
char *build_system_prompt_qwen35(struct TIEContext *ctx);


static const TensorDef QWEN35_GLOBAL_TENSORS[] = {
	DECLARE_GLOBAL_TENSORS_BASE_DEFS,
};

static const TensorDef QWEN35_LAYER_TENSORS[] = {
	{"blk.%u.attn_norm.weight", offsetof(LayerWeights, attn_norm), FLAG_NONE},
	{"blk.%u.post_attention_norm.weight", offsetof(LayerWeights, post_attn_norm), FLAG_NONE},
	{"blk.%u.ffn_down.weight", offsetof(LayerWeights, ffn_down), FLAG_NONE},
	{"blk.%u.ffn_gate.weight", offsetof(LayerWeights, ffn_gate), FLAG_NONE},
	{"blk.%u.ffn_up.weight", offsetof(LayerWeights, ffn_up), FLAG_NONE},
	{"blk.%u.attn_k.weight", offsetof(LayerWeights, attn_k), FLAG_OPTIONAL},
	{"blk.%u.attn_k_norm.weight", offsetof(LayerWeights, attn_k_norm), FLAG_OPTIONAL},
	{"blk.%u.attn_q.weight", offsetof(LayerWeights, attn_q), FLAG_OPTIONAL},
	{"blk.%u.attn_q_norm.weight", offsetof(LayerWeights, attn_q_norm), FLAG_OPTIONAL},
	{"blk.%u.attn_v.weight", offsetof(LayerWeights, attn_v), FLAG_OPTIONAL},
	{"blk.%u.attn_output.weight", offsetof(LayerWeights, attn_out), FLAG_OPTIONAL},
	{"blk.%u.attn_gate.weight", offsetof(LayerWeights, attn_gate), FLAG_OPTIONAL}, // used for wqkv_gate in linear
	{"blk.%u.attn_qkv.weight", offsetof(LayerWeights, attn_qkv), FLAG_OPTIONAL},
	{"blk.%u.ssm_dt.bias", offsetof(LayerWeights, ssm_dt_bias), FLAG_OPTIONAL},
	{"blk.%u.ssm_norm.weight", offsetof(LayerWeights, ssm_norm), FLAG_OPTIONAL},
	{"blk.%u.ssm_a", offsetof(LayerWeights, ssm_a), FLAG_OPTIONAL},
	{"blk.%u.ssm_out.weight", offsetof(LayerWeights, ssm_out), FLAG_OPTIONAL},
	{"blk.%u.ssm_conv1d.weight", offsetof(LayerWeights, ssm_conv1d_weight), FLAG_OPTIONAL},
	{"blk.%u.ssm_alpha.weight", offsetof(LayerWeights, ssm_alpha), FLAG_OPTIONAL},
	{"blk.%u.ssm_beta.weight", offsetof(LayerWeights, ssm_beta), FLAG_OPTIONAL},
/*
	{"blk.%u.ffn_gate_inp.weight", offsetof(LayerWeights, ffn_gate_inp), FLAG_OPTIONAL},
	{"blk.%u.ffn_gate_inp_shexps.weight", offsetof(LayerWeights, ffn_gate_exps), FLAG_OPTIONAL},
	{"blk.%u.ffn_gate_exps.weight", offsetof(LayerWeights, ffn_gate_exps), FLAG_OPTIONAL},
	{"blk.%u.ffn_gate_shexp.weight", offsetof(LayerWeights, ffn_gate_inp), FLAG_OPTIONAL},
	{"blk.%u.ffn_down_exps.weight", offsetof(LayerWeights, ffn_down_exps), FLAG_OPTIONAL},
	{"blk.%u.ffn_down_shexps.weight", offsetof(LayerWeights, ffn_down_exps), FLAG_OPTIONAL},
	{"blk.%u.ffn_up_exps.weight", offsetof(LayerWeights, ffn_up_exps), FLAG_OPTIONAL},
	{"blk.%u.ffn_up_shexps.weight", offsetof(LayerWeights, ffn_up_exps), FLAG_OPTIONAL},
*/
};

static const MetadataDef QWEN35_METADATA_DEFS[] = {
	DECLARE_LLAMA_BASE_METADATA_DEFS,
	{"%s.rope.dimension_sections", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, mrope_sections), true, true},
	{"%s.rope.dimension_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, rope_dimension_count), false,
	 false},
	{"%s.attention.value_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, attention_value_length), false,
	 false},
	{"%s.full_attention_interval", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, full_attention_interval), false,
	 false},
	{"%s.ssm.conv_kernel", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, ssm_conv_kernel), false, false},
	{"%s.ssm.state_size", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, ssm_state_size), false, false},
	{"%s.ssm.group_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, ssm_group_count), false, false},
	{"%s.ssm.time_step_rank", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, ssm_time_step_rank), false, false},
	{"%s.ssm.inner_size", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, ssm_inner_size), false, false},
	// (MoE)
	{"%s.expert_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, expert_count), false, true},
	{"%s.expert_used_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, expert_used_count), false, true},
	{"%s.expert_feed_forward_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, expert_ffn_dim), false, true},
	{"%s.expert_shared_feed_forward_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, expert_shared_ffn_dim), false, true},
	DECLARE_TOKENIZER_BASE_METADATA_DEFS,
};

static const BufferDef QWEN35_BUFFERS[] = {
	{offsetof(MemLayout, hidden_state), SIZE_EMBED_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayout, normed_qkv_input), SIZE_EMBED_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayout, Q), SIZE_QWEN35_Q_GATE, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayout, K), SIZE_KV_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayout, V), SIZE_KV_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayout, attn_output), SIZE_QWEN35_SSM_QKV, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayout, attn_proj_output), SIZE_EMBED_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayout, normed_ffn_input), SIZE_EMBED_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayout, gate_proj_output), SIZE_FFN_DIM_X_2, INTERNAL_MEMORY_TYPE, FLAG_NONE}, // Q+Gate or Z
	{offsetof(MemLayout, up_proj_output), SIZE_FFN_DIM, INTERNAL_MEMORY_TYPE,
	 FLAG_NONE}, // Temp for Gate de-interleave
	{offsetof(MemLayout, ffn_down_output), SIZE_EMBED_DIM, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayout, qkv_buf), SIZE_QWEN35_SSM_QKV, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayout, ssm_dt_buf), SIZE_QWEN35_SSM_DT_BETA, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayout, ssm_beta_buf), SIZE_QWEN35_SSM_DT_BETA, INTERNAL_MEMORY_TYPE, FLAG_NONE},
	{offsetof(MemLayout, logits), SIZE_VOCAB_SIZE, GGML_TYPE_F32, FLAG_NONE},
	{offsetof(MemLayout, pos_ids), SIZE_POS_IDS, GGML_TYPE_I32, FLAG_NONE},
};

static const TokenizeDef QWEN35_TOKENIZE_DEF = {
	.token_detect_specials = 1,
	.token_load_merges = 1,
	.token_load_scores = 0,
};

ModelDef QWEN35_DEF = {.arch = ARCH_QWEN35,
			.name = "Qwen35",
			.params =
				{
					.sot_token_id = 248045,
					.eot_token_id = 248046,
					.eos_token_id = 248046,
					.newline_token_id = 198,
					.role_user_token_id = 846,
					.role_model_token_id = 74455,
					.vision_start_token_id = -1,
					.vision_end_token_id = -1,
					.vision_embed_token_id = -1,

				},
			.interface =
				{
					//.build_system_prompt = build_system_prompt_qwen35,
					.tokenize_encode = tokenize_bpe,
					.tokenize_decode = decode_token_bpe,
					.build_prompt = build_prompt_qwen35,
					.transformer_layer = transformer_layer_qwen35,
					.build_rope_cache = build_rope_cache_dynamic,
				},
			DECLARE_LANGUAGE_MODEL_DEF(QWEN35, QWEN35)};

ModelDef QWEN35_MOE_DEF = {.arch = ARCH_QWEN35_MOE,
			.name = "Qwen35_MoE",
			.params =
				{
					.is_moe = 1,
					.sot_token_id = 248045,
					.eot_token_id = 248046,
					.eos_token_id = 248046,
					.newline_token_id = 198,
					.role_user_token_id = 846,
					.role_model_token_id = 74455,
					.vision_start_token_id = -1,
					.vision_end_token_id = -1,
					.vision_embed_token_id = -1,

				},
			.interface =
				{
					//.build_system_prompt = build_system_prompt_qwen35,
					.tokenize_encode = tokenize_bpe,
					.tokenize_decode = decode_token_bpe,
					.build_prompt = build_prompt_qwen35,
					.transformer_layer = transformer_layer_qwen35,
					.build_rope_cache = build_rope_cache_dynamic,
				},
			DECLARE_LANGUAGE_MODEL_DEF(QWEN35, QWEN35)};

/*
 * -------------------------------------------------------------------------
 * QWEN35 SPECIFIC IMPLEMENTATION
 * -------------------------------------------------------------------------
 */

#include "math_scalar.h"
void sconv1d_f32_scalar(struct TIEContext *ctx, int layer_idx, float *qkv_buf, int channels, int batch_len,
			int kernel_size)
{
	LayerMambaState *state = &ctx->mem.mamba_states[layer_idx];
	LayerWeights *w = &ctx->model->layers[layer_idx];

	float *conv_state = state->conv_state;
	const void *weights_data = w->ssm_conv1d_weight.mem.data;
	GGMLType weights_type = w->ssm_conv1d_weight.mem.type;

	for (int t = 0; t < batch_len; t++) {
		float *batch_qkv = qkv_buf + (size_t)t * channels;

		for (int c = 0; c < channels; c++) {
			float *c_state = conv_state + (size_t)c * kernel_size;

			// Shift history and add current token value
			for (int k = 0; k < kernel_size - 1; k++) {
				c_state[k] = c_state[k + 1];
			}
			c_state[kernel_size - 1] = batch_qkv[c];

			// Compute convolution (Depthwise)
			float sum = 0.0f;

			// GGUF [4, 6144] layout is c * 4 + k
			for (int k = 0; k < kernel_size; k++) {
				float weight;
				if (weights_type == GGML_TYPE_F32) {
					weight = ((const float *)weights_data)[c * kernel_size + k];
				} else if (weights_type == GGML_TYPE_BF16) {
					weight = bf16_to_fp32(((const uint16_t *)weights_data)[c * kernel_size + k]);
				} else {
					printf("%s unsupported weight\n", __FUNCTION__);
					exit(EXIT_FAILURE);
				}
				sum += c_state[k] * weight;
			}

			// Overwrite with exact SiLU(conv), match Python exactly
			batch_qkv[c] = sum / (1.0f + expf(-sum));
		}
	}
}

static void l2_norm_f32(float *x, int dim, float eps)
{
	float sum = 0.0f;

	for (int i = 0; i < dim; i++)
		sum = fmaf(x[i], x[i], sum);

	float inv_norm = 1.0f / sqrtf(sum + eps);

	for (int i = 0; i < dim; i++)
		x[i] *= inv_norm;
}

static inline float silu_exact(float x)
{
	return x / (1.0f + expf(-x)); // Use standard expf, not fast
}

void gated_delta_net_f32_scalar(struct TIEContext *ctx, int layer_idx, float *q_buf, float *k_buf, float *v_buf,
				float *alpha_buf, float *beta_buf, float *gate_buf, float *output, int batch_len,
				int n_heads, int n_kv_heads, int d_head, int qkv_stride)
{
	LayerMambaState *state_ptr = &ctx->mem.mamba_states[layer_idx];
	LayerWeights *l = &ctx->model->layers[layer_idx];
	float *state_base = state_ptr->ssm_state;

	float *dt_bias_f32 = (float *)l->ssm_dt_bias.mem.data;
	float *ssm_a_f32 = (float *)l->ssm_a.mem.data;

	int repeat_ratio = n_heads / n_kv_heads;

	// Qwen3.5 Gated Delta Net dimensions
	const float q_scale = 1.0f / sqrtf((float)d_head);

	// QK dimension per head group
	const int qk_dim_per_head = ctx->model->ssm_state_size;

	int ssm_inner_size = n_heads * d_head;

	static int debug_printed = 0;
	if (!debug_printed) {
		debug_printed = 1;
	}

	for (int t = 0; t < batch_len; t++) {
		// q_buf, k_buf, v_buf pointers are often slices of a single qkv_buf with qkv_stride
		float *T_q = q_buf + (size_t)t * qkv_stride;
		float *T_k = k_buf + (size_t)t * qkv_stride;
		float *T_v = v_buf + (size_t)t * qkv_stride;
		float *T_alpha = alpha_buf + (size_t)t * n_heads; // back to n_heads
		float *T_beta = beta_buf + (size_t)t * n_heads;	  // back to n_heads
		float *T_y = output + (size_t)t * ssm_inner_size;
		float *T_z = gate_buf + (size_t)t * ssm_inner_size;

		// Apply L2 norm and scale to Q/K ONCE (per head group)
		for (int hk = 0; hk < n_kv_heads; hk++) {
			float *qh_k = T_q + (size_t)hk * qk_dim_per_head;
			float *kh_k = T_k + (size_t)hk * qk_dim_per_head;
			l2_norm_f32(qh_k, qk_dim_per_head, ctx->model->norm_eps);
			l2_norm_f32(kh_k, qk_dim_per_head, ctx->model->norm_eps);

//			MemType qh_k = {};
//			MemType kh_k = {};
//			dispatch_l2_norm(qh_k, qk_dim_per_head, ctx->model->norm_eps);
//			dispatch_l2_norm(kh_k, qk_dim_per_head, ctx->model->norm_eps);

			for (int i = 0; i < qk_dim_per_head; i++) {
				qh_k[i] *= q_scale;
			}
		}

		// For Qwen3.5 4B: repeat Q/K to match V head count
		float *qh_repeated = NULL;
		float *kh_repeated = NULL;

		if (repeat_ratio > 1) {
			qh_repeated = (float *)malloc(n_heads * qk_dim_per_head * sizeof(float));
			kh_repeated = (float *)malloc(n_heads * qk_dim_per_head * sizeof(float));
			for (int h = 0; h < n_heads; h++) {
				int k_head_idx = h / repeat_ratio;
				memcpy(qh_repeated + h * qk_dim_per_head, T_q + k_head_idx * qk_dim_per_head,
				       qk_dim_per_head * sizeof(float));
				memcpy(kh_repeated + h * qk_dim_per_head, T_k + k_head_idx * qk_dim_per_head,
				       qk_dim_per_head * sizeof(float));
			}
		}

		for (int h = 0; h < n_heads; h++) {
			// State layout: [n_heads, d_head, qk_dim_per_head]
			float *h_state = state_base + (size_t)h * d_head * qk_dim_per_head;

			// For V heads, direct mapping
			float *vh = T_v + h * d_head;
			float *yh = T_y + h * d_head;

			// For Q/K, use repeated values if needed
			float *qh =
				(repeat_ratio > 1) ? (qh_repeated + h * qk_dim_per_head) : (T_q + h * qk_dim_per_head);
			float *kh =
				(repeat_ratio > 1) ? (kh_repeated + h * qk_dim_per_head) : (T_k + h * qk_dim_per_head);


			int weight_idx = h;
			if (repeat_ratio > 1) {
				int g = h / repeat_ratio;
				int r = h % repeat_ratio;
				// Map Sequential (G0H1 is idx 1) to Grouped (G0H1 is idx 16 in 4B)
				weight_idx = r * n_kv_heads + g;
			}

			// Preparation
			double alpha_val = (double)T_alpha[h] + (double)dt_bias_f32[weight_idx];
			float softplus_dt = (alpha_val > 20.0) ? alpha_val : log(1.0 + exp(alpha_val));
			double decay = exp((double)ssm_a_f32[weight_idx] * softplus_dt);
			float beta = 1.0 / (1.0 + exp(-(float)T_beta[h]));


			// Recurrence
			for (int i = 0; i < d_head; i++) {
				float v_pred = 0.0;
				for (int j = 0; j < qk_dim_per_head; j++) {
					// Decay previous state
					float decayed_state = (float)h_state[i * qk_dim_per_head + j] * (float)decay;
					// Use decayed state for prediction
					v_pred += decayed_state * (float)kh[j];
					// Save decayed state
					h_state[i * qk_dim_per_head + j] = (float)decayed_state;
				}

				float delta = (float)beta * ((float)vh[i] - v_pred);
				for (int j = 0; j < qk_dim_per_head; j++) {
					// Update state with error
					h_state[i * qk_dim_per_head + j] += (float)((float)delta * (float)kh[j]);
				}
			}

			// Output Summation
			for (int j = 0; j < d_head; j++) {
				float sum = 0.0;
				for (int i = 0; i < qk_dim_per_head; i++) {
					sum += (float)h_state[j * qk_dim_per_head + i] * (float)qh[i];
				}
				yh[j] = (float)sum;
			}
		}

		free(qh_repeated);
		free(kh_repeated);

		// Gated RMSNorm
		// Qwen 3.5 Mamba applies Norm and Gating per recurrent head
		for (int h = 0; h < n_heads; h++) {
			float *yh = T_y + h * d_head;
			float *zh = T_z + h * d_head;
			rms_norm_f32_f32_f32_scalar(yh, yh, &l->ssm_norm, d_head, ctx->model->norm_eps);

			for (int i = 0; i < d_head; i++) {
				float z = zh[i];
				float silu_z = z / (1.0f + expf(-z));
				yh[i] *= silu_z;
			}
		}
	}
}

void dispatch_sconv1d(struct TIEContext *ctx, int layer_idx, MemType *qkv_buf, int channels, int batch_len,
		      int kernel_size)
{
	sconv1d_f32_scalar(ctx, layer_idx, (float *)qkv_buf->data, channels, batch_len, kernel_size);
}

void dispatch_gated_delta_net(struct TIEContext *ctx, int layer_idx, MemType *q_buf, MemType *k_buf, MemType *v_buf,
			      MemType *alpha_buf, MemType *beta_buf, MemType *gate_buf, MemType *output, int batch_len,
			      int n_heads, int n_kv_heads, int d_head, int qkv_stride)
{
	gated_delta_net_f32_scalar(ctx, layer_idx, (float *)q_buf->data, (float *)k_buf->data, (float *)v_buf->data,
				   (float *)alpha_buf->data, (float *)beta_buf->data, (float *)gate_buf->data,
				   (float *)output->data, batch_len, n_heads, n_kv_heads, d_head, qkv_stride);
}

void activation_f32_f32_scalar(void *gate, const void *up, int size)
{
	float *gate_fp32 = (float *)gate;
	float *up_fp32 = (float *)up;

	for (int i = 0; i < size; i++) {
		gate_fp32[i] = silu_exact(gate_fp32[i]) * up_fp32[i];
	}
}

int transformer_layer_qwen35(struct TIEContext *ctx, int layer_idx, int batch_len)
{
	LayerWeights *l = &ctx->model->layers[layer_idx];
	int kv_dim = ctx->model->num_kv_heads * ctx->model->head_dim;
	int q_dim = ctx->model->num_heads * ctx->model->head_dim;
	RopeCacheType *active_rope_cache = ctx->model->rope_cache_global;
	int layer_type = ctx->model->layer_types[layer_idx];
	int sink_len = 4;
	int start_pos = ctx->kv_pos;


	// RMSNorm
	dispatch_rms_norm_batched(&ctx->mem.hidden_state, &l->attn_norm, &ctx->mem.normed_qkv_input,
				  ctx->model->embed_dim, ctx->model->norm_eps, batch_len, ctx->model->embed_dim);

	if (layer_type == LAYER_TYPE_MAMBA2) {
		int n_heads_ssm = ctx->model->ssm_time_step_rank;
		int s_inner = ctx->model->ssm_inner_size;
		int s_group = ctx->model->ssm_group_count;
		int s_state = ctx->model->ssm_state_size;
		int n_heads_v = ctx->model->ssm_inner_size / s_state; // 32 for 4B
		int n_groups = s_group;
		int repeat_ratio = n_heads_v / n_groups; // 2 for 4B
		int d_head_ssm = s_inner / n_heads_ssm;

		int qk_dim_ssm = s_group * s_state;	       // 2048
		int ssm_qkv_stride = 2 * qk_dim_ssm + s_inner; // 8192

		// Project Q, K, V
		dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_qkv, &ctx->mem.qkv_buf, batch_len,
				 ctx->model->embed_dim, ssm_qkv_stride, true);

		// Project Gate (Z)
		dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_gate, &ctx->mem.gate_proj_output, batch_len,
				 ctx->model->embed_dim, s_inner, true);

		// Convolution (must happen BEFORE de-shuffle because filters are also grouped)
		dispatch_sconv1d(ctx, layer_idx, &ctx->mem.qkv_buf, ssm_qkv_stride, batch_len,
				 ctx->model->ssm_conv_kernel);

		// De-shuffle V and Z for 4B+ models
		if (repeat_ratio > 1) {
			float *v_tmp = (float *)malloc(s_inner * sizeof(float));
			float *z_tmp = (float *)malloc(s_inner * sizeof(float));

			for (int t = 0; t < batch_len; t++) {
				// De-shuffle V (located at offset 2 * qk_dim_ssm in qkv_buf)
				float *v_base =
					(float *)ctx->mem.qkv_buf.data + (size_t)t * ssm_qkv_stride + 2 * qk_dim_ssm;
				// De-shuffle Z (the entire z_buf)
				float *z_base = (float *)ctx->mem.gate_proj_output.data + (size_t)t * s_inner;

				for (int g = 0; g < n_groups; g++) {
					for (int r = 0; r < repeat_ratio; r++) {
						int src_head_idx = r * n_groups + g;
						int dst_head_idx = g * repeat_ratio + r;

						// Copy V head
						memcpy(v_tmp + (size_t)dst_head_idx * s_state,
						       v_base + (size_t)src_head_idx * s_state,
						       s_state * sizeof(float));

						// Copy Z head
						memcpy(z_tmp + (size_t)dst_head_idx * s_state,
						       z_base + (size_t)src_head_idx * s_state,
						       s_state * sizeof(float));
					}
				}
				memcpy(v_base, v_tmp, s_inner * sizeof(float));
				memcpy(z_base, z_tmp, s_inner * sizeof(float));
			}
			free(v_tmp);
			free(z_tmp);
		}

		// Alpha (DT) Projection
		dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->ssm_alpha, &ctx->mem.ssm_dt_buf, batch_len,
				 ctx->model->embed_dim, n_heads_ssm, true);

		// Beta Projection
		dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->ssm_beta, &ctx->mem.ssm_beta_buf, batch_len,
				 ctx->model->embed_dim, n_heads_ssm, true);

		// De-shuffle
		if (repeat_ratio > 1) {
			int rank_per_group = ctx->model->ssm_time_step_rank / s_group;
			float *a_tmp = (float *)malloc(ctx->model->ssm_time_step_rank * sizeof(float));
			float *b_tmp = (float *)malloc(ctx->model->ssm_time_step_rank * sizeof(float));

			for (int t = 0; t < batch_len; t++) {
				float *a_base = (float *)ctx->mem.ssm_dt_buf.data + t * ctx->model->ssm_time_step_rank;
				float *b_base =
					(float *)ctx->mem.ssm_beta_buf.data + t * ctx->model->ssm_time_step_rank;
				for (int g = 0; g < s_group; g++) {
					for (int r = 0; r < rank_per_group; r++) {
						int src = r * s_group + g;
						int dst = g * rank_per_group + r;
						a_tmp[dst] = a_base[src];
						b_tmp[dst] = b_base[src];
					}
				}
				memcpy(a_base, a_tmp, ctx->model->ssm_time_step_rank * sizeof(float));
				memcpy(b_base, b_tmp, ctx->model->ssm_time_step_rank * sizeof(float));
			}
			free(a_tmp);
			free(b_tmp);
		}

		// Split Q, K, V for Recurrence
		MemType q_buf = mem_slice(&ctx->mem.qkv_buf, 0);
		MemType k_buf = mem_slice(&ctx->mem.qkv_buf, qk_dim_ssm);
		MemType v_buf = mem_slice(&ctx->mem.qkv_buf, 2 * qk_dim_ssm);

		// The Recurrence Call
		dispatch_gated_delta_net(ctx, layer_idx, &q_buf, &k_buf, &v_buf, &ctx->mem.ssm_dt_buf,
					 &ctx->mem.ssm_beta_buf, &ctx->mem.gate_proj_output, &ctx->mem.attn_output,
					 batch_len, n_heads_ssm, s_group, d_head_ssm, ssm_qkv_stride);

		if (repeat_ratio > 1) {
			// We must de-shuffle the attn_output back to 'Grouped' layout
			// so it matches the ssm_out weight matrix layout.
			float *grouped_attn_out = (float *)malloc(s_inner * sizeof(float));
			for (int t = 0; t < batch_len; t++) {
				float *seq_base = (float *)ctx->mem.attn_output.data + (size_t)t * s_inner;

				for (int g = 0; g < n_groups; g++) {
					for (int r = 0; r < repeat_ratio; r++) {
						int seq_head_idx = g * repeat_ratio + r; // [G0H0, G0H1...]
						int grp_head_idx = r * n_groups + g;	 // [G0H0, G1H0...]

						memcpy(grouped_attn_out + (size_t)grp_head_idx * s_state,
						       seq_base + (size_t)seq_head_idx * s_state,
						       s_state * sizeof(float));
					}
				}
				memcpy(seq_base, grouped_attn_out, s_inner * sizeof(float));
			}
			free(grouped_attn_out);
		}

		dispatch_mat_mat(ctx, &ctx->mem.attn_output, &l->ssm_out, &ctx->mem.attn_proj_output, batch_len,
				 s_inner, ctx->model->embed_dim, true);

		// Residual Add
		dispatch_apply_residual(&ctx->mem.hidden_state, &ctx->mem.attn_proj_output,
					batch_len * ctx->model->embed_dim);

	} else {

		// Full Attention
		// Compute Q+Gate, K, and V Matrices
		// attn_q in Qwen 3.5 Full layers outputs [Q, Gate] concatenated.
		// output 2 * q_dim to ensure capture the gating signal.
		dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_q, &ctx->mem.Q, batch_len,
				 ctx->model->embed_dim, 2 * q_dim, true);

		dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_k, &ctx->mem.K, batch_len,
				 ctx->model->embed_dim, kv_dim, true);

		dispatch_mat_mat(ctx, &ctx->mem.normed_qkv_input, &l->attn_v, &ctx->mem.V, batch_len,
				 ctx->model->embed_dim, kv_dim, true);

		// Separate Q and Gate into contiguous arrays
		// allocate a buffer for Q_clean and Gate_clean
		float *Q_split = (float *)malloc(batch_len * 2 * q_dim * sizeof(float));
		float *Q_clean = Q_split;
		float *Gate_clean = Q_split + batch_len * q_dim;

		for (int t = 0; t < batch_len; t++) {
			float *q_src = (float *)ctx->mem.Q.data + t * 2 * q_dim;
			float *q_dst = Q_clean + t * q_dim;
			float *g_dst = Gate_clean + t * q_dim;

			for (int h = 0; h < ctx->model->num_heads; h++) {
				memcpy(q_dst + h * ctx->model->head_dim, q_src + h * (2 * ctx->model->head_dim),
				       ctx->model->head_dim * sizeof(float));
				memcpy(g_dst + h * ctx->model->head_dim,
				       q_src + h * (2 * ctx->model->head_dim) + ctx->model->head_dim,
				       ctx->model->head_dim * sizeof(float));
			}
		}

		void *original_Q_ptr = ctx->mem.Q.data;
		ctx->mem.Q.data = Q_clean;

		// Apply Q/K Normalization and Apply RoPE
		static int attn_debug = 0;
		for (int h = 0; h < ctx->model->num_heads; h++) {
			MemType Q_slice = mem_slice(&ctx->mem.Q, (size_t)h * ctx->model->head_dim);
			dispatch_rms_norm_batched(&Q_slice, &l->attn_q_norm, &Q_slice, ctx->model->head_dim,
						  ctx->model->norm_eps, batch_len, q_dim);

			dispatch_apply_mrope_cache_batched(active_rope_cache, &Q_slice, start_pos,
							   ctx->model->rope_dimension_count, batch_len, q_dim);
		}

		for (int h = 0; h < ctx->model->num_kv_heads; h++) {
			MemType K_slice = mem_slice(&ctx->mem.K, (size_t)h * ctx->model->head_dim);
			dispatch_rms_norm_batched(&K_slice, &l->attn_k_norm, &K_slice, ctx->model->head_dim,
						  ctx->model->norm_eps, batch_len, kv_dim);

			dispatch_apply_mrope_cache_batched(active_rope_cache, &K_slice, start_pos,
							   ctx->model->rope_dimension_count, batch_len, kv_dim);
		}

		// Store K/V to cache
		dispatch_store_KV_cache(ctx, layer_idx, start_pos, batch_len, sink_len);

		// Multi-Head Attention Calculation
		attention(ctx, batch_len, layer_idx, layer_idx, start_pos, ATTN_TYPE_GLOBAL, attention_worker, sink_len,
			  256);

		// Apply Sigmoid Gating to Attention Output
		// Formula: y = y * sigmoid(gate).
		for (int t = 0; t < batch_len; t++) {
			float *out_t = (float *)ctx->mem.attn_output.data + (size_t)t * q_dim;
			float *gate_t = Gate_clean + (size_t)t * q_dim;

			for (int i = 0; i < q_dim; i++) {
				float g = gate_t[i];
				float sig_g = 1.0f / (1.0f + expf(-g)); // exact sigmoid
				out_t[i] *= sig_g;
			}
		}

		ctx->mem.Q.data = original_Q_ptr;
		free(Q_split);

		// Output Projection (O-Proj)
		dispatch_mat_mat(ctx, &ctx->mem.attn_output, &l->attn_out, &ctx->mem.attn_proj_output, batch_len, q_dim,
				 ctx->model->embed_dim, true);

		// Residual Add
		dispatch_apply_residual(&ctx->mem.hidden_state, &ctx->mem.attn_proj_output,
					batch_len * ctx->model->embed_dim);
	}

	// FFN Norm
	dispatch_rms_norm_batched(&ctx->mem.hidden_state, &l->post_attn_norm, &ctx->mem.normed_ffn_input,
				  ctx->model->embed_dim, ctx->model->norm_eps, batch_len, ctx->model->embed_dim);

	// FFN
	dispatch_mat_mat(ctx, &ctx->mem.normed_ffn_input, &l->ffn_gate, &ctx->mem.gate_proj_output, batch_len,
			 ctx->model->embed_dim, ctx->model->ffn_dim, true);

	dispatch_mat_mat(ctx, &ctx->mem.normed_ffn_input, &l->ffn_up, &ctx->mem.up_proj_output, batch_len,
			 ctx->model->embed_dim, ctx->model->ffn_dim, true);

	activation_f32_f32_scalar(ctx->mem.gate_proj_output.data, ctx->mem.up_proj_output.data,
				  batch_len * ctx->model->ffn_dim);

	dispatch_mat_mat(ctx, &ctx->mem.gate_proj_output, &l->ffn_down, &ctx->mem.ffn_down_output, batch_len,
			 ctx->model->ffn_dim, ctx->model->embed_dim, true);

	// Residual Add 2
	dispatch_apply_residual(&ctx->mem.hidden_state, &ctx->mem.ffn_down_output, batch_len * ctx->model->embed_dim);

	return 0;
}

int build_prompt_qwen35(struct TIEContext *ctx, int *prompt_tokens, int *user_text_tokens, int user_text_token_count,
			 bool has_image)
{
	VisionModel *vm = ctx->model_vision;
	ModelDef *def = ctx->model->def;
	int prompt_len = 0;

	if (ctx->model->add_bos_token == 1 && ctx->model->bos_token_sent == 0) {
		prompt_tokens[prompt_len++] = ctx->model->bos_token_id;
		ctx->model->bos_token_sent = 1;
	}

	if (def->params.sot_token_id != -1)
		prompt_tokens[prompt_len++] = def->params.sot_token_id;

	prompt_tokens[prompt_len++] = def->params.role_user_token_id;

	if (def->params.newline_token_id != -1)
		prompt_tokens[prompt_len++] = def->params.newline_token_id;

	memcpy(&prompt_tokens[prompt_len], user_text_tokens, user_text_token_count * sizeof(int));
	prompt_len += user_text_token_count;

	if (has_image) {
		int raw_w = ctx->vision_mem.image_raw_width;
		int raw_h = ctx->vision_mem.image_raw_height;
		int w_patches = raw_w / vm->patch_size;
		int h_patches = raw_h / vm->patch_size;
		int merged_w = w_patches / vm->spatial_merge_size;
		int merged_h = h_patches / vm->spatial_merge_size;
		int num_patches = merged_w * merged_h;

		prompt_tokens[prompt_len++] = def->params.vision_start_token_id;
		int patch_token = def->params.vision_embed_token_id;
		for (int i = 0; i < num_patches; i++) {
			prompt_tokens[prompt_len++] = patch_token;
		}
		prompt_tokens[prompt_len++] = def->params.vision_end_token_id;
		prompt_tokens[prompt_len++] = def->params.newline_token_id;
	}

	if (def->params.eot_token_id != -1)
		prompt_tokens[prompt_len++] = def->params.eot_token_id;

	if (def->params.newline_token_id != -1)
		prompt_tokens[prompt_len++] = def->params.newline_token_id;

	if (def->params.sot_token_id != -1)
		prompt_tokens[prompt_len++] = def->params.sot_token_id;

	prompt_tokens[prompt_len++] = def->params.role_model_token_id;

	if (def->params.newline_token_id != -1)
		prompt_tokens[prompt_len++] = def->params.newline_token_id;

	return prompt_len;
}

char *build_system_prompt_qwen35(struct TIEContext *ctx)
{
	return strdup(
		"<|im_start|>system\n"
		"You are a helpful and harmless assistant.<|im_end|>\n");
}
