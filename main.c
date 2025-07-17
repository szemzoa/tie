#include <immintrin.h>
#include <assert.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
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

#include "main.h"
#include "threadpool.h"
#include "maths.h"
#include "model.h"
#include "engine.h"
#include "predict.h"
#include "tokenize.h"

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

int64_t elapsed_time_us(const struct timespec after, const struct timespec before)
{
	return ((int64_t)after.tv_sec - (int64_t)before.tv_sec) * (int64_t)1000000
	       + ((int64_t)after.tv_nsec - (int64_t)before.tv_nsec) / 1000;
}

void generate_output(struct ctx_t *ctx, int current_token)
{
	const char *p = get_token_string(ctx->pool, current_token);
	int len = get_token_string_length(ctx->pool, current_token);

	char buf[256] = {0};
	if (len > 255)
		len = 255; // Safety limit

	memcpy(buf, p, len);
	buf[len] = '\0'; // Null-terminate safely

	replace_g_spaces(buf);
	printf("%s", buf);
	fflush(stdout);
}

// C function that implements our tool
static char *set_lamp_state(const char *state)
{
	static char tool_result[256];
	snprintf(tool_result, sizeof(tool_result), "{\"state\": \"%s\"}", state);
	return tool_result;
}

// Function to parse the tool call string and execute the tool
char *execute_tool_from_buffer(char *tool_call_buffer)
{
	printf("--- Tool Call Detected ---\n");

	replace_g_spaces(tool_call_buffer);

	printf("Raw Call: %s\n", tool_call_buffer);

	// Simple parsing using strstr
	const char *name_field = "\"name\": \"";
	const char *args_field = "\"state\": \"";

	char function_name[64] = {0};
	char location[128] = {0};

	const char *name_start = strstr(tool_call_buffer, name_field);
	if (name_start) {
		sscanf(name_start + strlen(name_field), "%[^\"]", function_name);
	}

	const char *args_start = strstr(tool_call_buffer, args_field);
	if (args_start) {
		sscanf(args_start + strlen(args_field), "%[^\"]", location);
	}

	printf("Function: '%s', Location: '%s'\n", function_name, location);

	// --- Execute the corresponding C function ---
	char *tool_result = NULL;

	for (int i = 0; tool_calls[i].name != NULL; i++) {
		if (strcmp(tool_calls[i].name, function_name) == 0) {
			tool_result = tool_calls[i].func(location);
			goto _tool_call_found;
		}
	}

	tool_result = "{\"error\": \"Unknown tool requested.\"}";

_tool_call_found:
	printf("Tool Result: %s\n", tool_result);

	return tool_result;
}

void generate_interactive(struct ctx_t *ctx, int max_new_tokens, int use_threads)
{
	char input_buf[2048];
	char prompt_buf[4096];
	struct timespec start, end;
	int gen_len;
	int initial_prompt = 1;
	bool in_tool_call = false;
	char tool_call_buffer[1024] = {0}; // Buffer to accumulate the tool call string
	char *tool_result;
	int tool_call_len = 0;
	int tool_call_start_token = vocab_lookup_token_id(ctx->root, "<tool_call>", 11);
	int tool_call_end_token = vocab_lookup_token_id(ctx->root, "</tool_call>", 12);


	while (1) {
		printf("\nYou: ");

		if (tool_call_len == 0) {
			if (!fgets(input_buf, sizeof(input_buf), stdin))
				break;

			if (strncmp(input_buf, "exit", 4) == 0)
				break;

			if (initial_prompt == 1) {
				initial_prompt = 0;

				// Build Qwen3-style chat input
				snprintf(prompt_buf, sizeof(prompt_buf),
					 "%s<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", system_prompt_with_tools, input_buf);
			} else {

				// Build Qwen3-style chat input
				snprintf(prompt_buf, sizeof(prompt_buf),
				         "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", input_buf);
			}
		} else {
			snprintf(
				prompt_buf, sizeof(prompt_buf),
				"<|im_start|>user\n<tool_response>%s</tool_response><|im_end|>\n<|im_start|>assistant\n",
				tool_result);

			tool_call_len = 0;
			tool_call_buffer[0] = '\0';
		}

		/* --- Setup Buffers --- */
		int *generated_tokens = calloc(max_new_tokens, sizeof(int));
		if (!generated_tokens) {
			fprintf(stderr, "Buffer allocation failed in generate()\n");
			free(generated_tokens);
			return;
		}

		/* --- Tokenizer --- */
		size_t prompt_len = 0;
		int *prompt_tokens = tokenize_bpe(ctx, prompt_buf, &prompt_len);

		memset((void *)&prompt_buf, 0, sizeof(prompt_buf));

		/* --- Prompt Processing --- */
		printf("--- Prompt Processing at pos: %u (Matrix Mode) ---\n", ctx->kv_pos);
		clock_gettime(CLOCK_REALTIME, &start);

		// 1. Unified Embedding Lookup for the prompt
		for (int i = 0; i < prompt_len; i++) {

			// Get the destination pointer for this token's embedding in the workspace
			float *dest = ctx->mem.hidden_state + (long long)i * ctx->model->embed_dim;

			// Call the dispatcher to fetch and dequantize the row
			get_embedding_row(&ctx->model->token_embd, prompt_tokens[i], dest, ctx->model->embed_dim);
		}

		// 2. Unified Layer Processing for the prompt
		for (int l = 0; l < ctx->model->num_layers; l++) {
			transformer_layer_unified(ctx, l, prompt_len, use_threads);
		}

		// 3. Prepare for generation by copying the last token's state
		memcpy(ctx->mem.hidden_state,
		       ctx->mem.hidden_state + (long long)(prompt_len - 1) * ctx->model->embed_dim,
		       ctx->model->embed_dim * sizeof(float));

		ctx->kv_pos += prompt_len;

		clock_gettime(CLOCK_REALTIME, &end);
		printf("--- Prompt Processing Complete %zu tokens, time: %llu msec ---\n", prompt_len,
		       elapsed_time_us(end, start) / 1000);

		/* --- Generation Loop --- */
		printf("\n--- Generation Start at pos: %u (Max %d new tokens) ---\nQwen3: ", ctx->kv_pos,
		       max_new_tokens);
		clock_gettime(CLOCK_REALTIME, &start);

		gen_len = 0;

		for (int step = 0; step < max_new_tokens; step++) {
			if (ctx->kv_pos >= ctx->model->seq_length) {
				printf("\nReached max sequence length.\n");
				break;
			}

			// 1. Calculate logits from the final hidden state of the previous token
			rms_norm(ctx->mem.normed_ffn_input, ctx->mem.hidden_state, ctx->model->output_norm.data,
				 ctx->model->embed_dim, ctx->model->norm_eps);

			parallel_mat_vec_unified(ctx->mem.normed_ffn_input, &ctx->model->token_embd, ctx->mem.logits,
						 ctx->model->embed_dim, ctx->model->vocab_size, use_threads);

			// 2. Sample the next token
			int next_token = predict_next_token(ctx->mem.logits, ctx->model->vocab_size, "temperature",
							    0.7f, 20, 0.95f, prompt_tokens, prompt_len,
							    generated_tokens, gen_len, ctx->model->repetition_penalty);

			if (in_tool_call) {
				// If we are inside a tool call, append the token's string to our buffer
				const char *token_str = get_token_string(ctx->pool, next_token);
				int token_len = strlen(token_str);
				if (tool_call_len + token_len < sizeof(tool_call_buffer)) {
					strcat(tool_call_buffer, token_str);
					tool_call_len += token_len;
				}
			}

			if (next_token == tool_call_start_token) {
				in_tool_call = true;
			}

			if (next_token == tool_call_end_token) {
				in_tool_call = false;
			}

			// 3. Print the token and check for EOS
			generate_output(ctx, next_token);

			if (gen_len < max_new_tokens)
				generated_tokens[gen_len++] = next_token;

			if (next_token == ctx->model->eos_token) {
				printf("\n--- EOS token reached ---\n");
				break;
			}

			// 4. Unified Embedding Lookup for the single generated token
			// The destination is the beginning of the hidden_state buffer
			get_embedding_row(&ctx->model->token_embd, next_token, ctx->mem.hidden_state,
					  ctx->model->embed_dim);

			// 5. Unified Layer Processing for the single token (batch_len = 1)
			for (int l = 0; l < ctx->model->num_layers; l++) {
				transformer_layer_unified(ctx, l, 1, use_threads);
			}
			ctx->kv_pos++;
		}

		if (tool_call_len > 0) {
			tool_result = execute_tool_from_buffer(tool_call_buffer);
		} else {
			clock_gettime(CLOCK_REALTIME, &end);
			printf("\n--- Generation End --- %u tokens, %llu msec, tps: %.01f --- %u\n", gen_len,
			       elapsed_time_us(end, start) / 1000,
			       (float)(((float)gen_len) / (float)(elapsed_time_us(end, start) / 1000.0 / 1000.0)),
			       ctx->kv_pos);
		}
		free(generated_tokens);
	}
}

static void print_usage(void)
{
	printf("parameters:\r\n");
	printf("\t -h (help)\r\n");
	printf("\t -m [model file]\r\n");
	printf("\t -t [thread num]\r\n");
}

int main(int argc, char *argv[])
{
	struct ctx_t *ctx;
	int opt;
	int num_threads = 1;
	char *model_path;
	int use_mmap = 0;

	printf("Toy Inference Engine\n");
	if (argc < 3) {
		print_usage();
		exit(EXIT_SUCCESS);
	}

	while ((opt = getopt(argc, argv, "t:m:h")) != -1) {
		switch (opt) {
		case 't':
			num_threads = atoi(optarg);
			break;

		case 'h':
			print_usage();
			exit(EXIT_SUCCESS);
			break;

		case 'm':
			model_path = optarg;
			break;

		case '?':
			print_usage();
			exit(EXIT_FAILURE);
			break;
		}
	}

	srand(time(NULL));

#ifdef CONFIG_ENABLE_AVX2
	__builtin_cpu_init();

	if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
		printf("init: AVX2 and FMA supported\n");
	} else {
		printf("init: AVX2 and FMA not supported\n");
		exit(EXIT_FAILURE);
	}
#endif

	if ((ctx = malloc(sizeof(struct ctx_t))) == NULL)
		return -1;

	ctx->root = create_node();
	ctx->pool = create_string_pool(1024 * 1024 * 2);

	if (gguf_read(ctx, model_path) != 0) {
		free(ctx);
		return -1;
	}

	if (model_create(ctx, use_mmap) != 0) {
		gguf_close(ctx);
		free(ctx);
		return -1;
	}

	thread_pool = thread_pool_create(num_threads);
	printf("init: thread_pool enabled, %u threads\n", num_threads);

	if (model_init(ctx, 1.0f, 1.0f) != 0)
		goto _cleanup;

	ctx->kv_pos = 0;

	printf("Welcome to Qwen3 interactive chat. Type 'exit' to quit.\n");
	generate_interactive(ctx, 8192, 1);

_cleanup:
	model_cleanup(ctx, use_mmap);

	gguf_close(ctx);
	free(ctx);

	printf("Done.\n");
	return 0;
}
