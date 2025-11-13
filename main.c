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
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <locale.h>

#include "main.h"
#include "threadpool.h"
#include "math_dispatch.h"
#include "model.h"
#include "engine.h"
#include "predict.h"
#include "tokenize.h"
#include "version.h"
#include "vision.h"

#ifdef CONFIG_ENABLE_AVX2
#include <immintrin.h>
#endif

const char *system_prompt_with_tools =
	"<|im_start|>system\n"
	"You may call one or more functions to assist with the user query.\n"
	"You are provided with function signatures within <tools></tools> XML tags:\n"
	"<tools>\n"
	"{\n"
	"\"name\": \"set_lamp_state\",\n"
	"\"description\": \"Control a smart lamp\",\n"
	"\"parameters\": {\n"
	"\"type\": \"object\",\n"
	"\"properties\": {\n"
	"\"state\": {\n"
	"\"type\": \"string\",\n"
	"\"enum\": [\"on\", \"off\"],\n"
	"\"description\": \"Turn the lamp on or off\"\n"
	"}\n"
	"},\n"
	"\"required\": [\"state\"]\n"
	"}\n"
	"}\n"
	"</tools>\n"
	"For each function call, return a json object with function name and arguments "
	"within <tool_call></tool_call> XML tags:\n"
	"<tool_call>\n"
	"{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
	"</tool_call>\n"
	"<|im_end|>\n";

static char *set_lamp_state(const char *location);

struct tool_entry_t tool_calls[] = {{
					    .name = "set_lamp_state",
					    .func = set_lamp_state,
				    },
				    {.name = NULL, .func = NULL}};


static char *set_lamp_state(const char *state)
{
	static char tool_result[256];
	snprintf(tool_result, sizeof(tool_result), "{\"state\": \"%s\"}", state);
	return tool_result;
}

int tool_call_init(struct TIEContext *ctx, struct tool_call_t *tool_call)
{
	tool_call->token_start = vocab_lookup_token_id(ctx->tokenizer.root, "<tool_call>", 11);
	tool_call->token_end = vocab_lookup_token_id(ctx->tokenizer.root, "</tool_call>", 12);
	tool_call->len = 0;
	tool_call->buffer[0] = '\0';
	tool_call->state = TOOL_CALL_STATE_IDLE;
	tool_call->result = NULL;

	return 0;
}

char *execute_tool_from_buffer(char *tool_call_buffer)
{
	printf("--- Tool Call Detected ---\n");
	replace_g_spaces(tool_call_buffer);
	printf("Raw Call: %s\n", tool_call_buffer);

	const char *name_field = "\"name\": \"";
	const char *args_field = "\"state\": \"";

	char function_name[64] = {0};
	char state[128] = {0};

	const char *name_start = strstr(tool_call_buffer, name_field);
	if (name_start) {
		sscanf(name_start + strlen(name_field), "%[^\"]", function_name);
	}

	const char *args_start = strstr(tool_call_buffer, args_field);
	if (args_start) {
		sscanf(args_start + strlen(args_field), "%[^\"]", state);
	}

	printf("Function: '%s', State: '%s'\n", function_name, state);

	char *tool_result = NULL;
	for (int i = 0; tool_calls[i].name != NULL; i++) {
		if (strcmp(tool_calls[i].name, function_name) == 0) {
			tool_result = tool_calls[i].func(state);
			goto _tool_call_found;
		}
	}
	tool_result = "{\"error\": \"Unknown tool requested.\"}";

_tool_call_found:
	printf("Tool Result: %s\n", tool_result);
	return tool_result;
}

int tool_call_handler(struct TIEContext *ctx, struct tool_call_t *tool_call, int token)
{
	switch (tool_call->state) {
	case TOOL_CALL_STATE_UNINITIALIZED:
		break;

	case TOOL_CALL_STATE_IDLE:
		if (token != tool_call->token_start)
			break;

		tool_call->state = TOOL_CALL_STATE_PROCESSING;
		break;

	case TOOL_CALL_STATE_PROCESSING:
		if (token != tool_call->token_end) {
			const char *token_str = get_token_string(ctx, token);
			int token_len = get_token_string_length(ctx, token);
			if (tool_call->len + token_len < TOOL_CALL_BUFFER_SIZE) {
				memcpy(tool_call->buffer + tool_call->len, token_str, token_len);
				tool_call->len += token_len;
			}
			break;
		}

		tool_call->buffer[tool_call->len] = '\0';
		tool_call->state = TOOL_CALL_STATE_END;
		break;

	case TOOL_CALL_STATE_END:
		if (token == ctx->model->eos_token_id) {
			tool_call->result = execute_tool_from_buffer(tool_call->buffer);
		}
		tool_call->len = 0;
		tool_call->buffer[0] = '\0';
		tool_call->state = TOOL_CALL_STATE_IDLE;
		break;
	}

	return 0;
}

int64_t elapsed_time_us(const struct timespec after, const struct timespec before)
{
	return ((int64_t)after.tv_sec - (int64_t)before.tv_sec) * (int64_t)1000000
	       + ((int64_t)after.tv_nsec - (int64_t)before.tv_nsec) / 1000;
}

void generate_output(struct TIEContext *ctx, int current_token)
{
	if (current_token != ctx->model->eos_token_id)
		ctx->model->interface.token_out(ctx, current_token);

	fflush(stdout);
}

void read_user_input(char *input_buf, size_t buf_size)
{
	printf("You: ");
	if (!fgets(input_buf, buf_size, stdin)) {
		input_buf[0] = '\0';
	}
	input_buf[strcspn(input_buf, "\n")] = 0;
}

int *build_multimodal_turn_tokens(struct TIEContext *ctx, const char *input_buf, bool has_image, size_t *num_tokens)
{
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
	prompt_tokens[prompt_len++] = ctx->model->sot_token_id;
	prompt_tokens[prompt_len++] = ctx->model->role_user_token_id;
	prompt_tokens[prompt_len++] = ctx->model->newline_token_id;

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
	prompt_tokens[prompt_len++] = ctx->model->eot_token_id;
	prompt_tokens[prompt_len++] = ctx->model->newline_token_id;
	prompt_tokens[prompt_len++] = ctx->model->sot_token_id;
	prompt_tokens[prompt_len++] = ctx->model->role_model_token_id;
	prompt_tokens[prompt_len++] = ctx->model->newline_token_id;

	free(user_text_tokens);
	*num_tokens = prompt_len;
	return prompt_tokens;
}

void run_multimodal_prompt(struct TIEContext *ctx, int *prompt_tokens, int prompt_len, int has_image)
{
	const int embed_dim = ctx->model->embed_dim;
	size_t current_pos = 0;
	size_t image_embed_idx = 0;
	MemType *image_embeddings = NULL;
	void (*embedding_scale_func)(struct TIEContext *, MemType *) = ctx->model->interface.embedding_scale;

	if (ctx->gguf_text->arch == ARCH_GEMMA3N) {
		process_prompt_gemma3n(ctx, prompt_tokens, prompt_len);
		return;
	}

	if (has_image == 1) {
		process_image_vision(ctx);
		image_embeddings = &ctx->vision_mem.projected_embeddings;
	}

	for (size_t i = 0; i < prompt_len; ++i) {
		int token_id = prompt_tokens[i];

		MemType dest_slice = mem_slice(&ctx->mem.hidden_state, current_pos * embed_dim);

		if (token_id == ctx->model->vision_embed_token_id) {
			// This is an image placeholder, copy the corresponding embedding slice
			MemType image_src_slice = mem_slice(image_embeddings, image_embed_idx * embed_dim);
			memcpy(dest_slice.data, image_src_slice.data, embed_dim * sizeof(float));
			image_embed_idx++;
		} else {

			// This is a normal text token, look it up and scale it
			dispatch_embedding_row(&ctx->model->token_embd, token_id, &dest_slice, embed_dim);

			if (embedding_scale_func) {
				embedding_scale_func(ctx, &dest_slice);
			}

		}

		current_pos++;
	}

	printf("--- Multimodal Prompt Processing at pos: %u (%zu tokens) ---\n", ctx->kv_pos, current_pos);
	process_embeddings(ctx, &ctx->mem.hidden_state, current_pos);
	printf("--- Multimodal Prompt Processing Complete ---\n");
}

void generate_interactive(struct TIEContext *ctx, int max_new_tokens, int has_image)
{
	char input_buf[2048];
	char prompt_buf[4096];
	struct timespec start, end;
	int gen_len;
	struct tool_call_t tool_call;
	Tensor *output_tensor = ctx->model->output.mem.data == NULL ? &ctx->model->token_embd : &ctx->model->output;
	ctx->model->bos_token_sent = 0;
	int *prompt_tokens;


	// Initialize tool call handling
	tool_call_init(ctx, &tool_call);

#if 0
	if (ctx->gguf_text->arch == ARCH_QWEN3 || ctx->gguf_text->arch == ARCH_QWEN3_MOE || ctx->gguf_text->arch == ARCH_QWEN3VL) {
	    // Process system prompt immediately at start
	    size_t system_prompt_len = 0;

	    int *system_prompt_tokens = tokenize_bpe(ctx, system_prompt_with_tools, &system_prompt_len);

	    printf("--- Processing System Prompt: %zu tokens ---\n", system_prompt_len);
	    clock_gettime(CLOCK_REALTIME, &start);

	    ctx->model->interface.process_prompt(ctx, system_prompt_tokens, system_prompt_len);

	    clock_gettime(CLOCK_REALTIME, &end);
	    printf("--- System Prompt Processed, time: %llu msec ---\n", elapsed_time_us(end, start) / 1000);
	    free(system_prompt_tokens);
	}
#endif

	while (1) {
		if (!tool_call.result) {

			read_user_input(input_buf, sizeof(input_buf));

			if (strncmp(input_buf, "/exit", 5) == 0)
				break;
		} else {

			if (ctx->gguf_text->arch == ARCH_QWEN3 || ctx->gguf_text->arch == ARCH_QWEN3_MOE
			    || ctx->gguf_text->arch == ARCH_QWEN3VL) {
				snprintf(
					prompt_buf, sizeof(prompt_buf),
					"<|im_start|>user\n<tool_response>%s</tool_response><|im_end|>\n<|im_start|>assistant\n",
					tool_call.result);
			}
			tool_call.result = NULL;
		}

		int *generated_tokens = calloc(max_new_tokens, sizeof(int));
		if (!generated_tokens) {
			fprintf(stderr, "Buffer allocation failed in generate()\n");
			free(generated_tokens);
			return;
		}

		size_t prompt_len = 0;

		prompt_tokens = build_multimodal_turn_tokens(ctx, input_buf, has_image, &prompt_len);

		if (prompt_tokens == NULL || prompt_len == 0) {
		    printf("Build prompt failed\n");
		    continue;
		}

		printf("--- Prompt Processing at pos: %u (Matrix Mode) %zu tokens ---\n", ctx->kv_pos, prompt_len);
		clock_gettime(CLOCK_REALTIME, &start);

		/* prompt processing */
		run_multimodal_prompt(ctx, prompt_tokens, prompt_len, has_image);

		has_image = 0;

		clock_gettime(CLOCK_REALTIME, &end);
		printf("--- Prompt Processing Complete %zu tokens, time: %llu msec ---\n", prompt_len,
		       elapsed_time_us(end, start) / 1000);


		printf("\n--- Generation Start (Max %d new tokens) ---\n", max_new_tokens);
		clock_gettime(CLOCK_REALTIME, &start);
		gen_len = 0;

		for (int step = 0; step < max_new_tokens; step++) {

			if (ctx->kv_pos >= ctx->model->seq_length) {
				printf("\nReached max sequence length.\n");
				break;
			}

			if (ctx->gguf_text->arch == ARCH_GEMMA3N) {
				// For the first generation step (step == 0), the altup_hidden_states buffer
				// contains `prompt_len` tokens. We need to process the LAST one.
				// For all subsequent steps, the buffer only contains 1 token.
				size_t n_tokens_in_buffer = (step == 0) ? prompt_len : 1;

				post_process_altup_states(ctx, &ctx->mem.hidden_state, ctx->mem.altup_hidden_states,
							  n_tokens_in_buffer);
			}

			dispatch_rms_norm(&ctx->mem.hidden_state, &ctx->model->output_norm, &ctx->mem.normed_ffn_input,
					  ctx->model->embed_dim, ctx->model->norm_eps);

			dispatch_mat_vec(ctx, &ctx->mem.normed_ffn_input, output_tensor, &ctx->mem.logits,
					 ctx->model->embed_dim, ctx->model->vocab_size, true);

			if (ctx->model->final_logit_softcap > 0.0f) {
				dispatch_softcap_logits(&ctx->mem.logits, ctx->model->vocab_size,
							ctx->model->final_logit_softcap);
			}

			int next_token = predict_next_token((float *)ctx->mem.logits.data, ctx->model->vocab_size,
							    "temperature", 0.7f, 20, 0.95f, prompt_tokens, prompt_len,
							    generated_tokens, gen_len, ctx->model->repetition_penalty);

			generate_output(ctx, next_token);

			tool_call_handler(ctx, &tool_call, next_token);

			if (gen_len < max_new_tokens)
				generated_tokens[gen_len++] = next_token;

			if (next_token == ctx->model->eos_token_id) {
				printf("\n--- EOS token reached ---\n");
				tool_call_handler(ctx, &tool_call, next_token);
				break;
			}

			ctx->model->interface.prepare_next_token(ctx, next_token);

			for (int l = 0; l < ctx->model->num_layers; l++) {
				ctx->model->interface.transformer_layer(ctx, l, 1);
			}

			ctx->kv_pos++;
		}

		if (!tool_call.result) {
			clock_gettime(CLOCK_REALTIME, &end);
			printf("\n--- Generation End --- %u tokens, %llu msec, tps: %.01f ---\n", gen_len,
			       elapsed_time_us(end, start) / 1000,
			       (float)(((float)gen_len) / (float)(elapsed_time_us(end, start) / 1000.0 / 1000.0)));
		}

		free(prompt_tokens);
		free(generated_tokens);
	}
}

static void print_usage(void)
{
	printf("Usage:\n");
	printf("\t -h, --help                     Show this help message\n");
	printf("\t -m, --model <file>             Model file path\n");
	printf("\t -c, --context <length>         Context length\n");
	printf("\t -t, --threads <num>            Number of threads\n");
	printf("\t -v, --mmproj <file>            Multi-modal vision projection file\n");
	printf("\t -i, --image <file>             Multi-modal vision image file\n");
	printf("\t --use-mmap                     Enable mmap for model loading\n");
}

void context_release(struct TIEContext *ctx)
{
	thread_pool_destroy(thread_pool);
	free(ctx);
}

int engine_alloc(struct TIEContext *ctx, int num_threads)
{
	printf("Create thread_pool: %u threads\n", num_threads);
	if ((thread_pool = thread_pool_create(num_threads)) == NULL) {
		free(ctx);
		exit(EXIT_FAILURE);
	}

	ctx->tokenizer.root = create_node();
	ctx->tokenizer.pool = create_string_pool(1024 * 1024 * 4);
	ctx->utf8_state = 0;
	ctx->utf8_codepoint = 0;

	// Initialize SiLU lookup table
	printf("init: SiLU table\n");
	silu_table_init();

	// Initialize RoPE cache
	printf("init: RoPE cache\n");
	ctx->rope_cache_local = malloc(sizeof(RopeCacheType));
	ctx->rope_cache_global = malloc(sizeof(RopeCacheType));
	if (!ctx->rope_cache_local || !ctx->rope_cache_global) {
		perror("Failed to allocate rope_cache");
		return -1;
	}

	return 0;
}

int main(int argc, char *argv[])
{
	struct TIEContext *ctx = NULL;
	int num_threads = 1;
	char *model_path;
	char *mmproj_path = NULL;
	char *image_path = NULL;
	int context_length = 0;
	int use_mmap = 0;
	int has_image = 0;
	ModelDef *text_def = NULL;
	ModelDef *vision_def = NULL;


	printf("Toy Inference Engine v%u.%u\n", VERSION_MAJOR, VERSION_MINOR);
	if (argc < 4) {
		print_usage();
		exit(EXIT_SUCCESS);
	}

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
			model_path = argv[++i];
		} else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) {
			num_threads = atoi(argv[++i]);
		} else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--mmproj") == 0) {
			mmproj_path = argv[++i];
		} else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--image") == 0) {
			image_path = argv[++i];
		} else if (strcmp(argv[i], "--use-mmap") == 0) {
			use_mmap = 1;
		} else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
			print_usage();
			exit(EXIT_SUCCESS);
		} else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--context") == 0) {
			context_length = atoi(argv[++i]);
		} else {
			fprintf(stderr, "Unknown option: %s\n", argv[i]);
			print_usage();
			exit(EXIT_FAILURE);
		}
	}

	printf("Model: %s\n", model_path);
	printf("MMProj: %s\n", mmproj_path ? mmproj_path : "(none)");
	printf("Threads: %d\n", num_threads);
	printf("Context length: %d\n", context_length);
	printf("Use mmap: %d\n", use_mmap);

	setlocale(LC_ALL, "en_US.UTF-8");
	srand(time(NULL));

#ifdef CONFIG_ENABLE_AVX2
	__builtin_cpu_init();
	if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
		printf("AVX2 and FMA supported\n");
	} else {
		printf("Error: AVX2 and FMA not supported, but enabled\n");
		exit(EXIT_FAILURE);
	}
#endif

	if ((ctx = malloc(sizeof(struct TIEContext))) == NULL) {
		perror("Failed to create context\n");
		exit(EXIT_FAILURE);
	}

	/* Alloc engine */
	engine_alloc(ctx, num_threads);

	/* Parse language model GGUF file */
	ctx->gguf_text = gguf_model_parse(model_path);
	if (ctx->gguf_text == NULL) {
		printf("Failed to detect model\n");
		exit(EXIT_FAILURE);
	}

	/* Parse optional vision model GGUF file */
	if (mmproj_path != NULL)
		ctx->gguf_vision = gguf_model_parse(mmproj_path);

	/* Load the language Model */
	if ((text_def = find_model_def(ctx->gguf_text)) == NULL) {
		printf("Failed to detect model def\n");
		exit(EXIT_FAILURE);
	}
	if (model_load(ctx, ctx->gguf_text, (void **)&ctx->model, (const ModelDef *)text_def, use_mmap) != 0) {
		printf("Failed to load model %s\n", vision_def->name);
		exit(EXIT_FAILURE);
	}
	ctx->model->def = text_def;

	/* Load the vision Model */
	if (ctx->gguf_vision) {
		if ((vision_def = find_model_def(ctx->gguf_vision)) == NULL) {
			printf("Failed to detect vision model def\n");
			exit(EXIT_FAILURE);
		}
		if (model_load(ctx, ctx->gguf_vision, (void **)&ctx->model_vision, (const ModelDef *)vision_def, use_mmap) != 0) {
			printf("Failed to load model %s\n", vision_def->name);
			exit(EXIT_FAILURE);
		}
		ctx->model_vision->def = vision_def;
	}

	/* Init language model defaults */
	ctx->model->shared_kv_layers = 0;
	ctx->model->final_logit_softcap = 0;
	ctx->model->weight_layout = LAYOUT_ROW_MAJOR;

	/* Fallbacks */
	if (ctx->model->rope_scale_factor == 0.0f)
		ctx->model->rope_scale_factor = 1.0f;
	if (context_length != 0)
		ctx->model->seq_length = context_length;


	/* Init language model */
	model_language_init(ctx, (void *)ctx->model, text_def, 1.0f, 1.0f);
	language_model_info(ctx);


	/* Init vision model */
	if (ctx->model_vision) {
		model_vision_init(ctx, (void *)ctx->model_vision, vision_def);
		vision_model_info(ctx);
	}

	if (ctx->model_vision && image_path != NULL) {
		if (load_bmp_clip(image_path, ctx) == 0) {
			has_image = 1;
			printf("Succesfully loaded %s image for vision process\n", image_path);
		} else {
			printf("Error loading image for vision process\n");
		}
	}

	/* start generation loop */
	printf("\nWelcome to interactive chat. Type '/exit' to quit.\n");
	generate_interactive(ctx, 8192, has_image);

_cleanup:
	gguf_model_close(ctx->gguf_text);

	model_language_cleanup(ctx, ctx->gguf_text, text_def, use_mmap);

	if (ctx->gguf_vision != NULL) {
		gguf_model_close(ctx->gguf_vision);
		model_vision_cleanup(ctx, ctx->gguf_vision, vision_def, use_mmap);
	}

	context_release(ctx);
	printf("Done.\n");

	exit(EXIT_SUCCESS);
}
