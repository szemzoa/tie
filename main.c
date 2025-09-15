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


#if 0
#include <poll.h>
#include <termios.h>

static struct termios orig_termios;

#define DEBOUNCE_MS 200
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
		if (token == ctx->model->eos_token) {
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

unsigned int decode_utf8(unsigned int *state, unsigned int *codep, unsigned char byte)
{
	if (*state == 0) {
		if (byte < 0x80) {
			*codep = byte;
			return *codep;
		} else if ((byte & 0xE0) == 0xC0) {
			*state = 1;
			*codep = byte & 0x1F;
		} else if ((byte & 0xF0) == 0xE0) {
			*state = 2;
			*codep = byte & 0x0F;
		} else if ((byte & 0xF8) == 0xF0) {
			*state = 3;
			*codep = byte & 0x07;
		} else {
			return 0xFFFD;
		}
	} else {
		if ((byte & 0xC0) == 0x80) {
			*codep = (*codep << 6) | (byte & 0x3F);
			(*state)--;
			if (*state == 0) {
				return *codep;
			}
		} else {
			*state = 0;
			*codep = 0;
			return 0xFFFD;
		}
	}
	return 0;
}

// Qwen3: needs UTF-8 streaming decode
void token_out_qwen3(struct ctx_t *ctx, const char *p, int len)
{
	for (char *c = (char *)p; len > 0; c++, len--) {
		unsigned int result = decode_utf8(&ctx->utf8_state, &ctx->utf8_codepoint, (unsigned char)*c);
		if (result != 0 && result != 0xFFFD) {
			printf("%c", result);
		}
	}
}

// Gemma3: SentencePiece detokenization
void token_out_gemma3(struct ctx_t *ctx, const char *p, int len)
{
	for (int i = 0; i < len; i++) {
		unsigned char ch = (unsigned char)p[i];
		if (ch == 0xE2 && i + 2 < len && (unsigned char)p[i + 1] == 0x96 && (unsigned char)p[i + 2] == 0x81) {
			// UTF-8 for "▁" (U+2581)
			putchar(' ');
			i += 2; // skip extra bytes
		} else {
			putchar(ch);
		}
	}
}

void generate_output(struct ctx_t *ctx, int current_token)
{
	const char *p = get_token_string(ctx, current_token);
	int len = get_token_string_length(ctx, current_token);

	if (current_token != ctx->model->eos_token)
		ctx->interface.token_out(ctx, p, len);

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

// Process prompt tokens
void process_prompt(struct ctx_t *ctx, int *prompt_tokens, size_t prompt_len)
{
	MemType hidden_state_slice;

	for (int i = 0; i < prompt_len; i++) {
		// Create a slice for this token's position in the hidden_state buffer
		hidden_state_slice = mem_slice(&ctx->mem.hidden_state, i * ctx->model->embed_dim);
		dispatch_embedding_row(&ctx->model->token_embd, prompt_tokens[i], &hidden_state_slice,
				       ctx->model->embed_dim);

		if (ctx->interface.embedding_scale != NULL)
			ctx->interface.embedding_scale(ctx, &hidden_state_slice);

	}

	for (int l = 0; l < ctx->model->num_layers; l++) {
		transformer_layer(ctx, l, prompt_len);
	}

	MemType hidden_state_first_slice = mem_slice(&ctx->mem.hidden_state, 0);
	memcpy(hidden_state_first_slice.data, hidden_state_slice.data,
	       ctx->model->embed_dim * ggml_type_size(ctx->mem.hidden_state.type));

	ctx->kv_pos += prompt_len;
}

#if 0
static void disable_raw_mode(void)
{
	tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
}

static void enable_raw_mode(void)
{
	tcgetattr(STDIN_FILENO, &orig_termios);
	atexit(disable_raw_mode);
	struct termios raw = orig_termios;
	raw.c_lflag &= ~(ECHO | ICANON);
	tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
}
#endif

int *process_user_request(struct ctx_t *ctx, const char *input_buf, size_t *num_tokens)
{
	const size_t user_role_token_count = 1;
	const size_t model_role_token_count = 1;
	size_t user_text_token_count = 0;

	int *user_text_tokens = ctx->interface.tokenize_prompt(ctx, input_buf, &user_text_token_count);
	if (!user_text_tokens)
		return NULL;

	int *prompt_tokens = malloc(MAX_PROMPT_BATCH_SIZE * sizeof(int));
	size_t prompt_len = 0;

	// --- BOS ---
	if (ctx->model->add_bos_token == 1 && ctx->model->bos_token_sent == 0) {
		prompt_tokens[prompt_len++] = ctx->model->bos_token;
		ctx->model->bos_token_sent = 1;
	}

	// --- Start of Turn: User ---
	prompt_tokens[prompt_len++] = ctx->model->sot_token;
	memcpy(&prompt_tokens[prompt_len], &ctx->model->role_user_token, user_role_token_count * sizeof(int));
	prompt_len += user_role_token_count;
	prompt_tokens[prompt_len++] = ctx->model->newline_token;

	// --- User's actual message ---
	memcpy(&prompt_tokens[prompt_len], user_text_tokens, user_text_token_count * sizeof(int));
	prompt_len += user_text_token_count;

	// --- Start of Turn: Model ---
	prompt_tokens[prompt_len++] = ctx->model->eot_token;
	prompt_tokens[prompt_len++] = ctx->model->newline_token;
	prompt_tokens[prompt_len++] = ctx->model->sot_token;
	memcpy(&prompt_tokens[prompt_len], &ctx->model->role_model_token, model_role_token_count * sizeof(int));
	prompt_len += model_role_token_count;
	prompt_tokens[prompt_len++] = ctx->model->newline_token;

	free(user_text_tokens);
	*num_tokens = prompt_len;

	char buf[256];
	printf("Tokens: \n");

	for (int i = 0; i < prompt_len; i++) {
		memset(buf, 0, 256);

		int token_length = get_token_string_length(ctx, prompt_tokens[i]);
		const unsigned char *token_str = get_token_string(ctx, prompt_tokens[i]);

		memcpy(buf, token_str, token_length < 256 ? token_length : 0);

		printf("#%u - %u, [%s]\n", i, prompt_tokens[i], buf);
	}

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

	    process_prompt(ctx, system_prompt_tokens, system_prompt_len);

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
		process_prompt(ctx, prompt_tokens, prompt_len);

		clock_gettime(CLOCK_REALTIME, &end);
		printf("--- Prompt Processing Complete %zu tokens, time: %llu msec ---\n", prompt_len,
		       elapsed_time_us(end, start) / 1000);


		printf("\n--- Generation Start (Max %d new tokens) ---\n%s: ", max_new_tokens,
		       ctx->model->arch == ARCH_QWEN3 ? "Qwen3" : "Gemma3");
		clock_gettime(CLOCK_REALTIME, &start);
		gen_len = 0;

		for (int step = 0; step < max_new_tokens; step++) {

			if (ctx->kv_pos >= ctx->model->seq_length) {
				printf("\nReached max sequence length.\n");
				break;
			}

			dispatch_rms_norm(&ctx->mem.hidden_state, &ctx->model->output_norm, &ctx->mem.normed_ffn_input,
					  ctx->model->embed_dim, ctx->model->norm_eps);

			dispatch_mat_vec(ctx, &ctx->mem.normed_ffn_input, output_tensor, &ctx->mem.logits,
					 ctx->model->embed_dim, ctx->model->vocab_size, true);

			int next_token = predict_next_token((float *)ctx->mem.logits.data, ctx->model->vocab_size,
							    "temperature", 0.7f, 64, 0.95f, prompt_tokens, prompt_len,
							    generated_tokens, gen_len, ctx->model->repetition_penalty);

			generate_output(ctx, next_token);

			tool_call_handler(ctx, &tool_call, next_token);

			if (gen_len < max_new_tokens)
				generated_tokens[gen_len++] = next_token;

			if (next_token == ctx->model->eos_token) {
				printf("\n--- EOS token reached ---\n");
				tool_call_handler(ctx, &tool_call, next_token);
				break;
			}

			// Create a slice for this token's position in the hidden_state buffer
			MemType hidden_state_slice = mem_slice(&ctx->mem.hidden_state, 0);
			dispatch_embedding_row(&ctx->model->token_embd, next_token, &hidden_state_slice,
					       ctx->model->embed_dim);

			if (ctx->interface.embedding_scale != NULL)
				ctx->interface.embedding_scale(ctx, &hidden_state_slice);

			for (int l = 0; l < ctx->model->num_layers; l++) {
				transformer_layer(ctx, l, 1);
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

#include "math_scalar.h"


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
	ctx->tokenizer.pool = create_string_pool(1024 * 1024 * 1);
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

	//	kv_cache_reset(ctx);
	generate_interactive(ctx, 8192);

_cleanup:
	model_cleanup(ctx, use_mmap);
	gguf_close(ctx);
	free(ctx);
	printf("Done.\n");
	return 0;
}
