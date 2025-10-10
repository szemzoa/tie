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
#include <locale.h>

#include "main.h"
#include "threadpool.h"
#include "math_dispatch.h"
#include "model.h"
#include "engine.h"
#include "predict.h"
#include "tokenize.h"
#include "version.h"

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

int tool_call_init(struct ctx_t *ctx, struct tool_call_t *tool_call)
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

int tool_call_handler(struct ctx_t *ctx, struct tool_call_t *tool_call, int token)
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

void generate_output(struct ctx_t *ctx, int current_token)
{
	const char *p = get_token_string(ctx, current_token);
	int len = get_token_string_length(ctx, current_token);

	if (current_token != ctx->model->eos_token_id)
		ctx->model->interface.token_out(ctx, p, len);

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


int *process_user_request(struct ctx_t *ctx, const char *input_buf, size_t *num_tokens)
{
	size_t user_text_token_count = 0;

	int *user_text_tokens = ctx->model->interface.tokenize_prompt(ctx, input_buf, &user_text_token_count);
	if (!user_text_tokens)
		return NULL;

	int *prompt_tokens = malloc(MAX_PROMPT_BATCH_SIZE * sizeof(int));
	size_t prompt_len = 0;

	// --- BOS ---
	if (ctx->model->add_bos_token == 1 && ctx->model->bos_token_sent == 0) {
		prompt_tokens[prompt_len++] = ctx->model->bos_token_id;
		ctx->model->bos_token_sent = 1;
	}

	// --- Start of Turn: User ---
	prompt_tokens[prompt_len++] = ctx->model->sot_token_id;
	memcpy(&prompt_tokens[prompt_len], &ctx->model->role_user_token_id, sizeof(int));
	prompt_len++;
	prompt_tokens[prompt_len++] = ctx->model->newline_token_id;

	// --- User's actual message ---
	memcpy(&prompt_tokens[prompt_len], user_text_tokens, user_text_token_count * sizeof(int));
	prompt_len += user_text_token_count;

	// --- Start of Turn: Model ---
	prompt_tokens[prompt_len++] = ctx->model->eot_token_id;
	prompt_tokens[prompt_len++] = ctx->model->newline_token_id;
	prompt_tokens[prompt_len++] = ctx->model->sot_token_id;
	memcpy(&prompt_tokens[prompt_len], &ctx->model->role_model_token_id, sizeof(int));
	prompt_len++;
	prompt_tokens[prompt_len++] = ctx->model->newline_token_id;

	free(user_text_tokens);
	*num_tokens = prompt_len;

	return prompt_tokens;
}

void generate_interactive(struct ctx_t *ctx, int max_new_tokens)
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
	if (ctx->model->arch == ARCH_QWEN3) {
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

			if (strncmp(input_buf, "exit", 4) == 0)
				break;
		} else {

			if (ctx->model->arch == ARCH_QWEN3) {
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
		prompt_tokens = process_user_request(ctx, input_buf, &prompt_len);

		printf("--- Prompt Processing at pos: %u (Matrix Mode) %zu tokens ---\n", ctx->kv_pos, prompt_len);
		clock_gettime(CLOCK_REALTIME, &start);

		/* prompt processing */
		ctx->model->interface.process_prompt(ctx, prompt_tokens, prompt_len);

		clock_gettime(CLOCK_REALTIME, &end);
		printf("--- Prompt Processing Complete %zu tokens, time: %llu msec ---\n", prompt_len,
		       elapsed_time_us(end, start) / 1000);


		printf("\n--- Generation Start (Max %d new tokens) ---\n%s: ", max_new_tokens, ctx->model->name);
		clock_gettime(CLOCK_REALTIME, &start);
		gen_len = 0;

		for (int step = 0; step < max_new_tokens; step++) {

			if (ctx->kv_pos >= ctx->model->seq_length) {
				printf("\nReached max sequence length.\n");
				break;
			}

			if (ctx->model->arch == ARCH_GEMMA3N) {
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

			int next_token = predict_next_token(
				(float *)ctx->mem.logits.data, ctx->model->vocab_size,
				//							    "temperature",
				// 0.7f, 64, 0.95f, prompt_tokens, prompt_len,
				"temperature", 0.7f, 20, 0.95f, prompt_tokens, prompt_len, generated_tokens, gen_len,
				ctx->model->repetition_penalty);

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
	printf("parameters:\r\n");
	printf("\t -h (help)\r\n");
	printf("\t -m [model file]\r\n");
	printf("\t -c [context length]\r\n");
	printf("\t -t [thread num]\r\n");
}

int main(int argc, char *argv[])
{
	struct ctx_t *ctx;
	int opt;
	int num_threads = 1;
	char *model_path;
	int context_length = 0;
	int use_mmap = 0;

	printf("Toy Inference Engine v%u.%u\n", VERSION_MAJOR, VERSION_MINOR);
	if (argc < 4) {
		print_usage();
		exit(EXIT_SUCCESS);
	}

	while ((opt = getopt(argc, argv, "t:m:h:c:")) != -1) {
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
		case 'c':
			context_length = atoi(optarg);
			break;
		case '?':
			print_usage();
			exit(EXIT_FAILURE);
			break;
		}
	}

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

	if ((ctx = malloc(sizeof(struct ctx_t))) == NULL)
		return -1;

	ctx->tokenizer.root = create_node();
	ctx->tokenizer.pool = create_string_pool(1024 * 1024 * 4);
	ctx->utf8_state = 0;
	ctx->utf8_codepoint = 0;

	if (gguf_parse(ctx, model_path) != 0) {
		free(ctx);
		return -1;
	}

	if (model_load(ctx, use_mmap, context_length) != 0) {
		gguf_close(ctx);
		free(ctx);
		return -1;
	}

	printf("Create thread_pool: %u threads\n", num_threads);
	thread_pool = thread_pool_create(num_threads);

	if (model_init(ctx, 1.0f, 1.0f) != 0)
		goto _cleanup;

	printf("Welcome to interactive chat. Type 'exit' to quit.\n");
	generate_interactive(ctx, 8192);

_cleanup:
	model_cleanup(ctx, use_mmap);
	gguf_close(ctx);
	free(ctx);

	printf("Done.\n");
	return 0;
}
