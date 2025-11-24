#include <assert.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <locale.h>

#include "main.h"
#include "threadpool.h"
#include "model.h"
#include "engine.h"
#include "predict.h"
#include "tokenize.h"
#include "version.h"
#include "vision.h"
#include "math_dispatch.h"
#include "tools.h"


// Architecture Helpers
static void handle_model_specific_post_step(struct TIEContext *ctx, int step, size_t prompt_len)
{
	if (ctx->gguf_text->arch == ARCH_GEMMA3N) {
		// For the first generation step, process the LAST token of the prompt.
		// For subsequent steps, buffer contains 1 token.
		size_t n_tokens_in_buffer = (step == 0) ? prompt_len : 1;
		post_process_altup_states(ctx, &ctx->mem.hidden_state, ctx->mem.altup_hidden_states,
					  n_tokens_in_buffer);
	}
}

int64_t elapsed_time_us(const struct timespec after, const struct timespec before)
{
	return ((int64_t)after.tv_sec - (int64_t)before.tv_sec) * (int64_t)1000000
	       + ((int64_t)after.tv_nsec - (int64_t)before.tv_nsec) / 1000;
}

void generate_output(struct TIEContext *ctx, int current_token)
{
	ModelDef *def = ctx->model->def;

	if (current_token == def->params.eos_token_id)
		return;

	char piece[256]; // Sufficient for one token

	int len = ctx->model->interface.decode_token(ctx, current_token, piece, sizeof(piece));

	if (len > 0) {
		fwrite(piece, 1, len, stdout);
		fflush(stdout);
	}
}

int *build_multimodal_turn_tokens(struct TIEContext *ctx, const char *input_buf, bool has_image, size_t *num_tokens)
{
	ModelDef *def = ctx->model->def;
	size_t user_text_token_count = 0;

	// Tokenize the actual user input text
	int *user_text_tokens = ctx->model->interface.tokenize_prompt(ctx, input_buf, &user_text_token_count);
	if (!user_text_tokens)
		return NULL;

	int *prompt_tokens = malloc(MAX_PROMPT_BATCH_SIZE * sizeof(int));
	if (!prompt_tokens) {
		perror("Failed to allocate prompt tokens buffer");
		free(user_text_tokens);
		return NULL;
	}
	size_t prompt_len = 0;

	// BOS (only for the very first turn)
	if (ctx->model->add_bos_token == 1 && ctx->model->bos_token_sent == 0) {
		prompt_tokens[prompt_len++] = ctx->model->bos_token_id;
		ctx->model->bos_token_sent = 1;
	}

	// Start of User Turn
	prompt_tokens[prompt_len++] = def->params.sot_token_id;
	prompt_tokens[prompt_len++] = def->params.role_user_token_id;
	prompt_tokens[prompt_len++] = def->params.newline_token_id;

	// User's Text Message
	memcpy(&prompt_tokens[prompt_len], user_text_tokens, user_text_token_count * sizeof(int));
	prompt_len += user_text_token_count;

	if (has_image) {

		if (ctx->model->interface.build_vision_tokens) {
			// Call the model-specific function
			int num_added = ctx->model->interface.build_vision_tokens(ctx, prompt_tokens, prompt_len);
			prompt_len += num_added;

		} else {
			fprintf(stderr, "ERROR: Model does not support vision, but an image was provided.\n");
			*num_tokens = 0;
			free(user_text_tokens);
			return NULL;
		}
	}

	// End of User Turn & Start of Model Turn
	prompt_tokens[prompt_len++] = def->params.eot_token_id;
	prompt_tokens[prompt_len++] = def->params.newline_token_id;
	prompt_tokens[prompt_len++] = def->params.sot_token_id;
	prompt_tokens[prompt_len++] = def->params.role_model_token_id;
	prompt_tokens[prompt_len++] = def->params.newline_token_id;

	free(user_text_tokens);
	*num_tokens = prompt_len;

	if (ctx->model->use_mrope == 1) {
		int h_patches = 0;
		int w_patches = 0;

		if (has_image && ctx->model_vision) {
			// Retrieve dimensions from the most recently processed image
			int raw_w = ctx->vision_mem.image_raw_width;
			int raw_h = ctx->vision_mem.image_raw_height;
			w_patches = raw_w / ctx->model_vision->patch_size;
			h_patches = raw_h / ctx->model_vision->patch_size;
		}

		build_mrope_position_ids(ctx, prompt_tokens, prompt_len, has_image, ctx->kv_pos, h_patches, w_patches);
		text_rope_cache_init(ctx, prompt_len, ctx->kv_pos);
	}

	return prompt_tokens;
}

// Prefill (Process Prompt & Image)
int engine_prefill(struct TIEContext *ctx, int *prompt_tokens, size_t prompt_len, bool has_image)
{
	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	ModelDef *def = ctx->model->def;
	const int embed_dim = ctx->model->embed_dim;
	size_t current_pos = 0;
	size_t image_embed_idx = 0;
	MemType *image_embeddings = NULL;

	printf("--- Prefill Start: pos %u, len %zu ---\n", ctx->kv_pos, prompt_len);

	// Custom Prompt Processor (if defined)
	if (ctx->model->interface.process_prompt) {
		ctx->model->interface.process_prompt(ctx, prompt_tokens, prompt_len);
		return 0;
	}

	// Process Vision Encoder
	if (has_image) {
		printf("Processing image");
		fflush(stdout);
		if (ctx->model->interface.process_image_vision) {
			image_embeddings = ctx->model->interface.process_image_vision(ctx);
		}
		if (!image_embeddings) {
			fprintf(stderr, " [Failed]\n");
			return -1;
		}
		printf(" [Done]\n");
	}

	// Embedding Lookup Loop
	for (size_t i = 0; i < prompt_len; ++i) {
		int token_id = prompt_tokens[i];
		MemType dest_slice = mem_slice(&ctx->mem.hidden_state, current_pos * embed_dim);

		if (has_image && token_id == def->params.vision_embed_token_id) {
			// Copy image embedding
			MemType image_src_slice = mem_slice(image_embeddings, image_embed_idx * embed_dim);
			memcpy(dest_slice.data, image_src_slice.data, embed_dim * sizeof(float));
			image_embed_idx++;
		} else {
			// Standard text embedding
			dispatch_embedding_row(&ctx->model->token_embd, token_id, &dest_slice, embed_dim);
			if (ctx->model->interface.embedding_scale) {
				ctx->model->interface.embedding_scale(ctx, &dest_slice);
			}
		}
		current_pos++;
	}

	// Run Transformer Layers (Batch Processing)
	process_embeddings(ctx, &ctx->mem.hidden_state, current_pos);

	clock_gettime(CLOCK_REALTIME, &end);
	printf("--- Prefill Complete: %llu ms ---\n", elapsed_time_us(end, start) / 1000);
	return 0;
}

// Decode (Generate Tokens)
// Returns the number of tokens generated
int engine_decode(struct TIEContext *ctx, int max_new_tokens, int *out_tokens, size_t prompt_len)
{
	struct timespec start, end;
	int gen_len = 0;
	Tensor *output_tensor = ctx->model->output.mem.data == NULL ? &ctx->model->token_embd : &ctx->model->output;

	printf("\n--- Generation Start ---\n");
	clock_gettime(CLOCK_REALTIME, &start);

	for (int step = 0; step < max_new_tokens; step++) {
		if (ctx->kv_pos >= ctx->model->seq_length) {
			printf("\n[Context Limit Reached]\n");
			break;
		}

		// Architecture Specific Post-Processing (e.g., Gemma3N AltUp)
		handle_model_specific_post_step(ctx, step, prompt_len);

		// Output Norm & Head
		dispatch_rms_norm(&ctx->mem.hidden_state, &ctx->model->output_norm, &ctx->mem.normed_ffn_input,
				  ctx->model->embed_dim, ctx->model->norm_eps);

		dispatch_mat_vec(ctx, &ctx->mem.normed_ffn_input, output_tensor, &ctx->mem.logits,
				 ctx->model->embed_dim, ctx->model->vocab_size, true);

		// Logit Softcap (Gemma3)
		if (ctx->model->final_logit_softcap > 0.0f) {
			dispatch_softcap_logits(&ctx->mem.logits, ctx->model->vocab_size,
						ctx->model->final_logit_softcap);
		}

		// Sampling
		int next_token =
			predict_next_token((float *)ctx->mem.logits.data, ctx->model->vocab_size, "temperature", 0.7f,
					   20, 0.95f, NULL, 0, out_tokens, gen_len, ctx->model->repetition_penalty);

		/* tools process */
		bool token_consumed = tools_process_token(ctx, next_token);

		// If NOT consumed, print it normally
		if (!token_consumed) {
			generate_output(ctx, next_token);
		}

		// Check if a tool just finished executing
		if (ctx->tool_context.state == TOOL_STATE_RESULT_READY) {
			// We need to return to the main loop to insert the tool result.
			printf("\n[System] Tool execution complete. Returning control.\n");
			break;
		}

		if (gen_len < max_new_tokens) {
			out_tokens[gen_len++] = next_token;
		}

		// Run Next Token through Transformer
		ctx->model->interface.prepare_next_token(ctx, next_token);

		for (int l = 0; l < ctx->model->num_layers; l++) {
			ctx->model->interface.transformer_layer(ctx, l, 1);
		}

		if (next_token == ctx->model->eos_token_id) {
			printf("\n--- EOS ---\n");
			break;
		}

		// Advance KV Position
		ctx->kv_pos++;

	}

	clock_gettime(CLOCK_REALTIME, &end);
	float tps = (float)gen_len / (elapsed_time_us(end, start) / 1000000.0f);
	printf("\n--- Generation End: %d tokens, %.2f tps ---\n", gen_len, tps);

	return gen_len;
}

void read_user_input(char *input_buf, size_t buf_size)
{
	printf("\nYou: ");
	if (!fgets(input_buf, buf_size, stdin)) {
		input_buf[0] = '\0';
	}
	input_buf[strcspn(input_buf, "\n")] = 0;
}

// Main Chat Loop
void run_chat_session(struct TIEContext *ctx, int max_new_tokens, bool initial_has_image)
{
	char input_buf[MAX_PROMPT_BATCH_SIZE];
	struct timespec start, end;
	int *prompt_tokens = NULL;
	int *generated_tokens = calloc(max_new_tokens, sizeof(int));
	bool current_has_image = initial_has_image;
	ctx->model->bos_token_sent = 0;
	size_t system_prompt_len = 0;
	char *system_prompt = NULL;


	/* Init tool call handler */
	if (tools_init(ctx) == 0 && ctx->model->interface.build_system_prompt) {

		// Process system prompt immediately at start
		system_prompt = ctx->model->interface.build_system_prompt(ctx);

		int *system_prompt_tokens =
			ctx->model->interface.tokenize_prompt(ctx, system_prompt, &system_prompt_len);

		// If the model needs a BOS token, and it hasn't been sent, prepend it to the system prompt.
    		if (ctx->model->add_bos_token == 1 && ctx->model->bos_token_sent == 0)
        		ctx->model->bos_token_sent = 1;


		printf("--- Processing System Prompt: %zu tokens ---\n", system_prompt_len);
		clock_gettime(CLOCK_REALTIME, &start);

		if (ctx->model->use_mrope == 1) {
        		// System prompt is text-only, so h_patches=0, w_patches=0
	                // start_pos is ctx->kv_pos (usually 0 here)
        		build_mrope_position_ids(ctx, system_prompt_tokens, system_prompt_len,
                                     false, ctx->kv_pos, 0, 0);

        		text_rope_cache_init(ctx, system_prompt_len, ctx->kv_pos);
    		}

		engine_prefill(ctx, system_prompt_tokens, system_prompt_len, false);

		clock_gettime(CLOCK_REALTIME, &end);
		printf("--- System Prompt Processed, time: %llu msec ---\n", elapsed_time_us(end, start) / 1000);

		free(system_prompt);
		free(system_prompt_tokens);
	}

	while (1) {
		const char *input_ptr;

		// Check if we have a pending tool result
		if (ctx->tool_context.state == TOOL_STATE_RESULT_READY) {
			input_ptr = ctx->tool_context.result_prompt;

			// Reset state logic
			ctx->tool_context.state = TOOL_STATE_IDLE;

		} else {
			// Normal User Input
			read_user_input(input_buf, sizeof(input_buf));
			if (strncmp(input_buf, "/exit", 5) == 0)
				break;
			input_ptr = input_buf;
		}

		// Build Tokens (Includes M-RoPE calculation inside)
		size_t prompt_len = 0;
		if (prompt_tokens)
			free(prompt_tokens); // Free previous turn

		prompt_tokens = build_multimodal_turn_tokens(ctx, input_ptr, current_has_image, &prompt_len);

		if (!prompt_tokens || prompt_len == 0) {
			printf("Error: Failed to build prompt.\n");
			continue;
		}

		// If we used a tool result, free it now
		if (ctx->tool_context.result_prompt && input_ptr == ctx->tool_context.result_prompt) {
			free(ctx->tool_context.result_prompt);
			ctx->tool_context.result_prompt = NULL;
		}

		// Run Prefill
		if (engine_prefill(ctx, prompt_tokens, prompt_len, current_has_image) != 0) {
			continue;
		}

		// Image is consumed after the first turn
		current_has_image = false;

		// Run Decode
		engine_decode(ctx, max_new_tokens, generated_tokens, prompt_len);
	}

	free(generated_tokens);
	if (prompt_tokens)
		free(prompt_tokens);
}

AppConfig parse_args(int argc, char *argv[])
{
	AppConfig config = {0};
	config.num_threads = 1;

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0)
			config.model_path = argv[++i];
		else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0)
			config.num_threads = atoi(argv[++i]);
		else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--mmproj") == 0)
			config.mmproj_path = argv[++i];
		else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--image") == 0)
			config.image_path = argv[++i];
		else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--context") == 0)
			config.context_length = atoi(argv[++i]);
		else if (strcmp(argv[i], "--use-mmap") == 0)
			config.use_mmap = 1;
		else {
			printf("Unknown option: %s\n", argv[i]);
			exit(1);
		}
	}
	return config;
}

int main(int argc, char *argv[])
{
	printf("Toy Inference Engine v%u.%u\n", VERSION_MAJOR, VERSION_MINOR);
	if (argc < 2) {
		printf("Usage: %s -m <model.gguf> [-v <mmproj.gguf> -i <image.bmp>]\n", argv[0]);
		return 0;
	}

	setlocale(LC_ALL, "en_US.UTF-8");

	struct TIEContext *ctx = malloc(sizeof(struct TIEContext));
	if (!ctx)
		exit(EXIT_FAILURE);

	ctx->config = parse_args(argc, argv);

	// Init Engine
	engine_alloc(ctx, ctx->config.num_threads);

	// Load Models
	ctx->gguf_text = gguf_model_parse(ctx->config.model_path);
	if (!ctx->gguf_text)
		exit(EXIT_FAILURE);

	if (ctx->config.mmproj_path) {
		ctx->gguf_vision = gguf_model_parse(ctx->config.mmproj_path);
	}

	ModelDef *text_def = find_model_def(ctx->gguf_text);
	ModelDef *vision_def = ctx->gguf_vision ? find_model_def(ctx->gguf_vision) : NULL;

	model_load(ctx, ctx->gguf_text, (void **)&ctx->model, text_def, ctx->config.use_mmap);
	ctx->model->def = text_def;

	if (ctx->gguf_vision) {
		model_load(ctx, ctx->gguf_vision, (void **)&ctx->model_vision, vision_def, ctx->config.use_mmap);
		ctx->model_vision->def = vision_def;
	}

	// Allocate Memory & Init Weights
	model_language_init(ctx, (void *)ctx->model, text_def, 1.0f, 1.0f);
	language_model_info(ctx);

	if (ctx->model_vision) {
		model_vision_init(ctx, (void *)ctx->model_vision, vision_def);
		vision_model_info(ctx);
	}

	// Load Image (if any)
	bool has_image = false;
	if (ctx->model_vision && ctx->config.image_path) {
		if (load_bmp_clip(ctx->config.image_path, ctx) == 0) {
			has_image = true;
			printf("Loaded image: %s\n", ctx->config.image_path);
		}
	}
/*
	printf("sot_token_id: %d\n", ctx->model->def->params.sot_token_id);
	printf("eot_token_id: %d\n", ctx->model->def->params.eot_token_id);
	printf("eos_token_id: %d\n", ctx->model->def->params.eos_token_id);
	printf("newline_token_id: %d\n", ctx->model->def->params.newline_token_id);
	printf("role_user_token_id: %d\n", ctx->model->def->params.role_user_token_id);
	printf("role_model_token_id: %d\n", ctx->model->def->params.role_model_token_id);
	printf("double_newline_token_id: %d\n", ctx->model->def->params.double_newline_token_id);
	printf("vision_start_token_id: %d\n", ctx->model->def->params.vision_start_token_id);
	printf("vision_end_token_id: %d\n", ctx->model->def->params.vision_end_token_id);
	printf("vision_embed_token_id: %d\n", ctx->model->def->params.vision_embed_token_id);

	printf("bos_token_id: %d\n", ctx->model->bos_token_id);
	printf("unk_token_id: %d\n", ctx->model->unk_token_id);
	printf("pad_token_id: %d\n", ctx->model->pad_token_id);
	printf("eos_token_id: %d\n", ctx->model->eos_token_id);
*/

	// Run Chat
	run_chat_session(ctx, 8192, has_image);

	// Cleanup
	engine_release(ctx);
	printf("Done.\n");

	exit(EXIT_SUCCESS);

	return 0;
}