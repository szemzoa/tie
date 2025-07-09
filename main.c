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

int64_t elapsed_time_us(const struct timespec after, const struct timespec before)
{
	return ((int64_t)after.tv_sec - (int64_t)before.tv_sec) * (int64_t)1000000 +
		   ((int64_t)after.tv_nsec - (int64_t)before.tv_nsec) / 1000;
}

void generate_output(struct ctx_t *ctx, int current_token)
{
	const char *p	= get_token_string(ctx->pool, current_token);
	int			len = get_token_string_length(ctx->pool, current_token);

	char buf[256] = {0};
	if (len > 255)
		len = 255; // Safety limit

	memcpy(buf, p, len);
	buf[len] = '\0'; // Null-terminate safely

	replace_g_spaces(buf);
	printf("%s", buf);
	fflush(stdout);
}

void generate_interactive(struct ctx_t *ctx, int max_new_tokens, int use_threads)
{
	struct timespec start, end;
	char			input_buf[2048];
	char			prompt_buf[4096];
	int				gen_len;

	while (1) {
		printf("\nYou: ");

		if (!fgets(input_buf, sizeof(input_buf), stdin))
			break;

		if (strncmp(input_buf, "exit", 4) == 0)
			break;

		// Build Qwen3-style chat input
		snprintf(prompt_buf, sizeof(prompt_buf), "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", input_buf);

		/* --- Setup Buffers --- */
		int *generated_tokens = calloc(max_new_tokens, sizeof(int));
		if (!generated_tokens) {
			fprintf(stderr, "Buffer allocation failed in generate()\n");
			free(generated_tokens);
			return;
		}

		/* --- Tokenizer --- */
		size_t prompt_len	 = 0;
		int	  *prompt_tokens = tokenize_bpe(ctx, prompt_buf, &prompt_len);

		/* --- Prompt Processing --- */
		printf("--- Prompt Processing at pos: %u (Matrix Mode) ---\n", ctx->kv_pos);
		clock_gettime(CLOCK_REALTIME, &start);

		// 1. Unified Embedding Lookup for the prompt
		for (int i = 0; i < prompt_len; i++) {
			memcpy(ctx->mem.hidden_state + (long long)i * ctx->model->embed_dim,
				   &ctx->model->token_embd[(long long)prompt_tokens[i] * ctx->model->embed_dim],
				   ctx->model->embed_dim * sizeof(float));
		}

		// 2. Unified Layer Processing for the prompt
		for (int l = 0; l < ctx->model->num_layers; l++) {
			transformer_layer_unified(ctx, l, prompt_len, use_threads);
		}

		// 3. Prepare for generation by copying the last token's state
		memcpy(ctx->mem.hidden_state, ctx->mem.hidden_state + (long long)(prompt_len - 1) * ctx->model->embed_dim,
			   ctx->model->embed_dim * sizeof(float));

		ctx->kv_pos += prompt_len;

		clock_gettime(CLOCK_REALTIME, &end);
		printf("--- Prompt Processing Complete %zu tokens, time: %llu msec ---\n", prompt_len,
			   elapsed_time_us(end, start) / 1000);

		/* --- Generation Loop --- */
		printf("\n--- Generation Start at pos: %u (Max %d new tokens) ---\nQwen3: ", ctx->kv_pos, max_new_tokens);
		clock_gettime(CLOCK_REALTIME, &start);

		gen_len = 0;

		for (int step = 0; step < max_new_tokens; step++) {
			if (ctx->kv_pos >= ctx->model->seq_length) {
				printf("\nReached max sequence length.\n");
				break;
			}

			// 1. Calculate logits from the final hidden state of the previous token
			rms_norm(ctx->mem.normed_ffn_input, ctx->mem.hidden_state, ctx->model->output_norm, ctx->model->embed_dim,
					 ctx->model->norm_eps);

			parallel_mat_vec(ctx->mem.normed_ffn_input, ctx->model->token_embd, ctx->mem.logits, ctx->model->embed_dim,
							 ctx->model->vocab_size, use_threads);

			// 2. Sample the next token
			int next_token = predict_next_token(ctx->mem.logits, ctx->model->vocab_size, "temperature", 0.7f, 20, 0.95f,
												prompt_tokens, prompt_len, generated_tokens, gen_len,
												ctx->model->repetition_penalty);

			if (gen_len < max_new_tokens)
				generated_tokens[gen_len++] = next_token;

			// 3. Print the token and check for EOS
			generate_output(ctx, next_token);

			if (next_token == ctx->model->eos_token) {
				//			ctx->kv_pos++;
				printf("\n--- EOS token reached ---\n");
				break;
			}

			// 4. Unified Embedding Lookup for the single generated token
			memcpy(ctx->mem.hidden_state, &ctx->model->token_embd[(long long)next_token * ctx->model->embed_dim],
				   ctx->model->embed_dim * sizeof(float));

			// 5. Unified Layer Processing for the single token (batch_len = 1)
			for (int l = 0; l < ctx->model->num_layers; l++) {
				transformer_layer_unified(ctx, l, 1, use_threads);
			}

			ctx->kv_pos++;
		}

		clock_gettime(CLOCK_REALTIME, &end);
		printf("\n--- Generation End --- %u tokens, %llu msec, tps: %.01f --- %u\n", gen_len,
			   elapsed_time_us(end, start) / 1000,
			   (float)(((float)gen_len) / (float)(elapsed_time_us(end, start) / 1000.0 / 1000.0)), ctx->kv_pos);

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

	printf("Tiny Inference Engine\n");
	if (argc < 3) {
	    print_usage();
	    exit(EXIT_SUCCESS);
	}

	while ((opt = getopt(argc, argv, "t:m:h")) != -1) {

	    switch(opt) {

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
	} else
#endif
	{
		printf("init: AVX2 and FMA NOT supported\n");
	}

	if ((ctx = malloc(sizeof(struct ctx_t))) == NULL)
		return -1;

	ctx->root = create_node();
	ctx->pool = create_string_pool(1024 * 1024 * 2);

	if (gguf_read(ctx, model_path) != 0) {
		free(ctx);
		return -1;
	}

	if (model_create(ctx) != 0) {
		free(ctx);
		gguf_close(ctx);
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
	model_cleanup(ctx);

	gguf_close(ctx);
	free(ctx);

	printf("Done.\n");
	return 0;
}
