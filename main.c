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
#include <dirent.h>
#include <sys/stat.h>
#include <termios.h>
#include <poll.h>
#include <errno.h>
#include <sys/ioctl.h>

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

CommandQueue cmd_q = {.mutex = PTHREAD_MUTEX_INITIALIZER, .cond = PTHREAD_COND_INITIALIZER};
EventQueue evt_q = {.mutex = PTHREAD_MUTEX_INITIALIZER, .cond = PTHREAD_COND_INITIALIZER};

int start_think, end_think;
char think_buf[65536];
size_t think_len = 0;
int in_think = 0;

bool think_visible = true;
bool think_active = false;
bool tab_pressed = false;

static struct termios orig_term;


int64_t elapsed_time_us(const struct timespec after, const struct timespec before)
{
	return ((int64_t)after.tv_sec - (int64_t)before.tv_sec) * (int64_t)1000000
	       + ((int64_t)after.tv_nsec - (int64_t)before.tv_nsec) / 1000;
}

void input_init(void)
{
	tcgetattr(STDIN_FILENO, &orig_term);
	struct termios raw = orig_term;
	raw.c_lflag &= ~(ICANON | ECHO);
	raw.c_iflag &= ~(IXON | IXOFF);
	tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
}

void input_shutdown(void)
{
	tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_term);
}

bool input_readline(char *buf, size_t buflen)
{
	size_t len = 0;
	memset(buf, 0, buflen);

	while (1) {
		char c;
		if (read(STDIN_FILENO, &c, 1) != 1)
			continue;

		if (c == '\t') {
			tab_pressed = true;
			return true;
		}

		// Enter or Ctrl+Q
		//        if (c == '\n' || c == '\r') {
		if (c == 0x11) {
			write(STDOUT_FILENO, "\n", 1);
			return true;
		}

		// Backspace handling
		if (c == 0x7f || c == '\b') {
			if (len > 0) {
				// UTF-8 Logic: Remove bytes until we hit the start of the character.
				// A UTF-8 continuation byte always starts with bits '10' (0x80-0xBF).
				// A start byte or ASCII starts with '0' or '11'.
				while (len > 0) {
					// Remove last byte
					len--;
					unsigned char deleted = (unsigned char)buf[len];
					buf[len] = 0;

					// If it was NOT a continuation byte, we have removed the whole char.
					if ((deleted & 0xC0) != 0x80) {
						break;
					}
				}

				// Erase the character from terminal
				write(STDOUT_FILENO, "\b \b", 3);
			}
			continue;
		}

		if (c == 0x04) { // Ctrl-D
			return false;
		}

		// Input acceptance
		if (len + 1 < buflen) {
			unsigned char uc = (unsigned char)c;

			if (uc >= 32 || uc == '\n' || uc == '\r') {
				buf[len++] = c;
				write(STDOUT_FILENO, &c, 1);
			}
		}
	}
}


void clear_reasoning_above_spinner()
{
	printf("\033[2J\033[H"); // clear + home
}

void print_reasoning_buffer()
{
	fwrite(think_buf, 1, think_len, stdout);
	fflush(stdout);
}

void append_to_think_buffer(const char *tok, int len)
{
	if (think_len + len < sizeof(think_buf) - 1) {
		memcpy(think_buf + think_len, tok, len);
		think_len += len;
		think_buf[think_len] = '\0';
	}
}

void spinner_tick(const char *status)
{
	static const char *frames[] = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};
	static int idx = 0;

	printf("\r\033[K" THINK "%s %s" CLR_RESET, frames[idx], status);
	fflush(stdout);

	idx = (idx + 1) % 10;
}

// ioctl(STDOUT_FILENO, TIOCGWINSZ, ...)
static int count_newlines(const char *s, size_t len)
{
	int count = 0;
	for (size_t i = 0; i < len; i++) {
		if (s[i] == '\n')
			count++;
	}
	return count;
}

void toggle_think_visibility(void)
{
	if (!in_think)
		return;

	int lines_to_clear = count_newlines(think_buf, think_len);

	think_visible = !think_visible;

	if (think_visible) {
		printf("\r\033[K");

		printf(THINK);
		if (think_len > 0) {
			fwrite(think_buf, 1, think_len, stdout);
		}
		fflush(stdout);

	} else {
		for (int i = 0; i < lines_to_clear; i++) {
			// \033[2K: Clear current line
			// \033[A: Move cursor up one line
			printf("\r\033[2K\033[A");
		}

		// Clear the line where the "<think>" header started
		printf("\r\033[2K");

		spinner_tick("Thinking...");
	}
}

void generate_output(struct TIEContext *ctx, int current_token)
{
	ModelDef *def = ctx->model->def;

	if (current_token == def->params.eos_token_id)
		return;

	if (current_token == start_think) {
		in_think = 1;
		printf(THINK);
		if (!think_visible) {
			spinner_tick("Thinking...");
		}
	}

	char piece[256];
	int len = ctx->model->interface.tokenize_decode(ctx, current_token, piece, sizeof(piece));

	if (len > 0) {
		if (in_think) {
			append_to_think_buffer(piece, len);

			if (think_visible) {
				fwrite(piece, 1, len, stdout);
			} else {
				spinner_tick("Thinking...");
			}
		} else {
			fwrite(piece, 1, len, stdout);
		}
	}

	if (current_token == end_think) {
		if (!think_visible) {

			printf("\r\033[K");
			printf(THINK);

			if (think_len > 0)
				fwrite(think_buf, 1, think_len, stdout);
		}

		in_think = 0;
		think_len = 0;

		printf(CLR_RESET ASSISTANT_OUT);
	}

	fflush(stdout);
}


int *build_multimodal_turn_tokens(struct TIEContext *ctx, const char *input_buf, bool has_image, size_t *num_tokens,
				  bool prefill_only)
{
	size_t user_text_token_count = 0;
	size_t prompt_len = 0;

	if (ctx->model_vision == NULL)
		has_image = false;

	int *prompt_tokens = malloc(MAX_PROMPT_BATCH_SIZE * sizeof(int));
	if (!prompt_tokens) {
		perror("Failed to allocate prompt tokens buffer");
		return NULL;
	}

	// Tokenize the actual user input text FIRST
	int *user_text_tokens = ctx->model->interface.tokenize_encode(ctx, input_buf, &user_text_token_count);
	if (!user_text_tokens)
		return NULL;

	if (prefill_only == true) {

		for (int i = 0; i < user_text_token_count; i++)
			prompt_tokens[i] = user_text_tokens[i];

		free(user_text_tokens);

#ifdef DEBUG_TOKENS
		char buf[256];
		for (int i = 0; i < user_text_token_count; i++) {
			const char *p = get_token_string(ctx, prompt_tokens[i]);
			int len = get_token_string_length(ctx, prompt_tokens[i]);

			memset(buf, 0, 256);
			memcpy(buf, p, len);

			printf("Prefill Token #%u [%u] %s\n", i, prompt_tokens[i], buf);
		}
#endif

		*num_tokens = user_text_token_count;
		return prompt_tokens;
	}

	if (ctx->model->interface.build_prompt) {
		prompt_len = ctx->model->interface.build_prompt(ctx, prompt_tokens, user_text_tokens,
								user_text_token_count, has_image);
	}

	free(user_text_tokens);

#ifdef DEBUG_TOKENS
	char buf[256];
	for (int i = 0; i < prompt_len; i++) {
		const char *p = get_token_string(ctx, prompt_tokens[i]);
		int len = get_token_string_length(ctx, prompt_tokens[i]);

		memset(buf, 0, 256);
		memcpy(buf, p, len);

		printf("Token #%u [%u] %s\n", i, prompt_tokens[i], buf);
	}
#endif

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

	*num_tokens = prompt_len;
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

	printf(DEBUG "--- Prefill Start: pos %u, len %zu ---\n" CLR_RESET, ctx->kv_pos, prompt_len);

	// Custom Prompt Processor (if defined)
	if (ctx->model->interface.process_prompt_custom) {
		ctx->model->interface.process_prompt_custom(ctx, prompt_tokens, prompt_len);
		return 0;
	}

	// Process Vision Encoder
	if (has_image) {
		printf(DEBUG "Processing image");
		fflush(stdout);
		if (ctx->model->interface.process_image_vision) {
			image_embeddings = ctx->model->interface.process_image_vision(ctx);
		}
		if (!image_embeddings) {
			fprintf(stderr, " [Failed]\n");
			return -1;
		}
		printf(" [Done]\n" CLR_RESET);
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

	// Run Transformer Layers
	process_embeddings(ctx, &ctx->mem.hidden_state, current_pos);

	clock_gettime(CLOCK_REALTIME, &end);
	float tps = (float)prompt_len / (elapsed_time_us(end, start) / 1000000.0f);
	printf(DEBUG "--- Prefill Complete: %llu ms %.2f tps---\n" CLR_RESET, elapsed_time_us(end, start) / 1000, tps);
	return 0;
}

void dispatch_scale_buffer(MemType *buffer, int size, float scale)
{
	float *data = (float *)buffer->data;
	for (int i = 0; i < size; i++) {
		data[i] *= scale;
	}
}

// Decode (Generate Tokens)
int engine_decode(struct TIEContext *ctx, int max_new_tokens, int *out_tokens, size_t prompt_len,
		  token_callback_t on_token, void *user_data)
{
	struct timespec start, end;
	int gen_len = 0;
	Tensor *output_tensor = ctx->model->output.mem.data == NULL ? &ctx->model->token_embd : &ctx->model->output;

	//	printf(DEBUG "\n--- Generation Start ---\n" ASSISTANT_OUT);
	printf("\n" ASSISTANT_OUT "%s: ", ctx->model->def->name);
	clock_gettime(CLOCK_REALTIME, &start);

	printf(CLR_RESET ASSISTANT_OUT);

	for (int step = 0; step < max_new_tokens; step++) {

		if (ctx->stop_generation)
			break;

		while (ctx->pause_generation) {
			usleep(100000);
			if (ctx->stop_generation)
				break;
		}

		// Architecture Specific Post-Processing (ex: Gemma3N AltUp)
		if (ctx->model->interface.post_generate_step)
			ctx->model->interface.post_generate_step(ctx, step, prompt_len);

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

		// GRANITE LOGIT SCALING
		if (ctx->model->logit_scale != 0.0f && ctx->model->logit_scale != 1.0f) {
			float inv_scale = 1.0f / ctx->model->logit_scale;
			dispatch_scale_buffer(&ctx->mem.logits, ctx->model->vocab_size, inv_scale);
		}

		// Sampling
		int next_token =
			predict_next_token((float *)ctx->mem.logits.data, ctx->model->vocab_size, "temperature",
					   ctx->config.temperature, ctx->config.top_k, ctx->config.top_p, NULL, 0,
					   out_tokens, gen_len, ctx->model->repetition_penalty);

		/* tools process */
		bool token_consumed = tools_process_token(ctx, next_token);

		// Check if a tool call is ready to be executed
		if (ctx->tool_context.state == TOOL_STATE_CALL_READY) {
			printf(DEBUG "\n[Model] Tool call detected. Pausing for execution.\n" CLR_RESET);
			// Break the loop to return control to model_thread_main.
			break;
		}

		// If NOT consumed, print it normally
		if (!token_consumed) {
			if (on_token)
				on_token(user_data, next_token);
		}

		if (gen_len < max_new_tokens) {
			out_tokens[gen_len++] = next_token;
		}

		// Run Next Token through Transformer, either custom or standard processor
		if (ctx->model->interface.prepare_next_token_custom) {
			ctx->model->interface.prepare_next_token_custom(ctx, next_token);
		} else {
			prepare_next_token_standard(ctx, next_token);
		}

		for (int l = 0; l < ctx->model->num_layers; l++) {
			ctx->model->interface.transformer_layer(ctx, l, 1);
		}

		if (next_token == ctx->model->eos_token_id)
			break;

		// Advance KV Position
		ctx->kv_pos++;
	}

	clock_gettime(CLOCK_REALTIME, &end);

	float tps = (float)gen_len / (elapsed_time_us(end, start) / 1000000.0f);
	printf(DEBUG "\n--- Generation End: %d tokens, %.2f tps ---\n" CLR_RESET, gen_len, tps);

	return gen_len;
}

void push_cmd(ModelCommand cmd)
{
	pthread_mutex_lock(&cmd_q.mutex);
	int next = (cmd_q.tail + 1) % QUEUE_SIZE;

	// Simple block if full
	while (next == cmd_q.head) {
		pthread_cond_wait(&cmd_q.cond, &cmd_q.mutex);
	}
	cmd_q.buffer[cmd_q.tail] = cmd;
	cmd_q.tail = next;

	pthread_cond_signal(&cmd_q.cond);
	//	pthread_cond_broadcast(&cmd_q.cond);

	pthread_mutex_unlock(&cmd_q.mutex);
}

ModelCommand pop_cmd()
{
	pthread_mutex_lock(&cmd_q.mutex);

	while (cmd_q.head == cmd_q.tail) {
		pthread_cond_wait(&cmd_q.cond, &cmd_q.mutex);
	}

	ModelCommand cmd = cmd_q.buffer[cmd_q.head];
	cmd_q.head = (cmd_q.head + 1) % QUEUE_SIZE;

	pthread_cond_signal(&cmd_q.cond);

	pthread_mutex_unlock(&cmd_q.mutex);
	return cmd;
}

void push_evt(ModelEvent evt)
{
	pthread_mutex_lock(&evt_q.mutex);

	int next = (evt_q.tail + 1) % (QUEUE_SIZE * 4);

	// If full, we block the model thread.
	while (next == evt_q.head) {
		pthread_cond_wait(&evt_q.cond, &evt_q.mutex);
	}
	evt_q.buffer[evt_q.tail] = evt;
	evt_q.tail = next;

	pthread_cond_signal(&evt_q.cond);

	//	pthread_cond_broadcast(&evt_q.cond);
	pthread_mutex_unlock(&evt_q.mutex);
}

ModelEvent pop_evt()
{
	pthread_mutex_lock(&evt_q.mutex);
	while (evt_q.head == evt_q.tail) {
		pthread_cond_wait(&evt_q.cond, &evt_q.mutex);
	}
	ModelEvent evt = evt_q.buffer[evt_q.head];
	evt_q.head = (evt_q.head + 1) % (QUEUE_SIZE * 4);

	pthread_cond_signal(&evt_q.cond); // Signal space available

	//	pthread_cond_broadcast(&evt_q.cond);
	pthread_mutex_unlock(&evt_q.mutex);
	return evt;
}

static void timespec_add_ms(struct timespec *ts, long ms)
{
	ts->tv_nsec += ms * 1000000L;
	if (ts->tv_nsec >= 1000000000L) {
		ts->tv_sec += ts->tv_nsec / 1000000000L;
		ts->tv_nsec %= 1000000000L;
	}
}

ModelEvent pop_evt_timed(int timeout_ms)
{
	struct timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);
	timespec_add_ms(&ts, timeout_ms);

	pthread_mutex_lock(&evt_q.mutex);

	// Wait until buffer is not empty OR timeout
	while (evt_q.head == evt_q.tail) {
		int rc = pthread_cond_timedwait(&evt_q.cond, &evt_q.mutex, &ts);
		if (rc == ETIMEDOUT) {
			pthread_mutex_unlock(&evt_q.mutex);
			return (ModelEvent){.type = EVT_TIMEOUT};
		}
	}

	ModelEvent evt = evt_q.buffer[evt_q.head];
	evt_q.head = (evt_q.head + 1) % (QUEUE_SIZE * 4);

	pthread_cond_signal(&evt_q.cond);
	pthread_mutex_unlock(&evt_q.mutex);
	return evt;
}

// Non-blocking check for keyboard input during generation
void check_generation_interaction(struct TIEContext *ctx)
{
	char c;
	struct pollfd pfd = {.fd = STDIN_FILENO, .events = POLLIN};

	if (poll(&pfd, 1, 0) > 0) {

		if (read(STDIN_FILENO, &c, 1) == 1) {

			if (c == '\t') {

				toggle_think_visibility();

			} else if (c == 0x10) { // Ctrl+P
				ctx->pause_generation = !ctx->pause_generation;
				if (ctx->pause_generation) {
					printf(DEBUG " [Paused] " CLR_RESET);
				} else {
					printf(DEBUG " [Resumed] " CLR_RESET);
					if (!in_think)
						printf(ASSISTANT_OUT);
					else
						printf(THINK);
				}
				fflush(stdout);

			} else if (c == 0x13) { // Ctrl+S
				ctx->stop_generation = true;
				printf(DEBUG " [Stopping...] " CLR_RESET);
				fflush(stdout);
			}
		}
	}
}

void model_on_token(void *user_data, int token_id)
{
	ModelEvent evt = {.type = EVT_TOKEN, .token_id = token_id};

	push_evt(evt);
}

void *model_thread_main(void *arg)
{
	struct TIEContext *ctx = (struct TIEContext *)arg;
	int *generated_tokens = calloc(8192, sizeof(int));

	while (1) {
		// Wait for work
		ModelCommand cmd = pop_cmd();

		if (cmd.type == CMD_EXIT)
			break;

		if (cmd.type == CMD_GENERATE) {

			ctx->stop_generation = false;
			ctx->pause_generation = false;

			// Build Prompt
			size_t prompt_len = 0;
			int *tokens = build_multimodal_turn_tokens(ctx, cmd.text, cmd.has_image, &prompt_len,
								   cmd.prefill_only);

			if (!tokens) {

				push_evt((ModelEvent){.type = EVT_FINISHED});

				if (cmd.text)
					free(cmd.text);
				continue;
			}

			// Run Prefill
			engine_prefill(ctx, tokens, prompt_len, cmd.has_image);
			if (cmd.prefill_only) {
				push_evt((ModelEvent){.type = EVT_FINISHED});
				continue;
			}


			// Run Decode
			engine_decode(ctx, MAX_GENERATE_TOKENS, generated_tokens, prompt_len, model_on_token, NULL);

			// Handle Result
			if (ctx->tool_context.state == TOOL_STATE_CALL_READY) {
				// Model wants to call a tool
				push_evt((ModelEvent){.type = EVT_TOOL_CALL});
				continue;
			}

			// Cleanup
			if (cmd.text)
				free(cmd.text);
			if (tokens)
				free(tokens);

			push_evt((ModelEvent){.type = EVT_FINISHED});
		}
	}

	free(generated_tokens);
	return NULL;
}

int read_user_input(char *input_buf, size_t buf_size)
{
	printf(USER_PROMPT "\n> ");
	fflush(stdout);

	if (!input_readline(input_buf, buf_size)) {
		//		push_cmd((ModelCommand){.type = CMD_EXIT});
		return -2;
	}

	return 0;
}

void run_ui_loop(struct TIEContext *ctx)
{
	char input_buf[MAX_PROMPT_BATCH_SIZE];
	bool image_loaded = false;

	// Start the background thread
	pthread_t thread_id;
	pthread_create(&thread_id, NULL, model_thread_main, ctx);

	// System Prompt
	if (tools_init(ctx) == 0 && ctx->model->interface.build_system_prompt) {

		char *sys_prompt = ctx->model->interface.build_system_prompt(ctx);

		if (sys_prompt) {
			printf(DEBUG "[Init] Processing System Prompt...\n" CLR_RESET);
			ModelCommand cmd = {.type = CMD_GENERATE,
					    .text = strdup(sys_prompt),
					    .has_image = false,
					    .prefill_only = true};
			push_cmd(cmd);

			// Wait for it to finish
			while (1) {
				ModelEvent evt = pop_evt();
				if (evt.type == EVT_FINISHED)
					break;
			}
		}
	}

	printf(CLR_RESET CLR_BOLD CLR_RED "\nWelcome to interactive chat. Available commands:\n");
	if (ctx->model_vision) {
		printf("'/img /path/xxx.bmp'\t- Load an image.\n");
	}
	printf("'CTRL+Q'\t\t- Send query.\n");
	printf("'CTRL+P'\t\t- Pause generation.\n");
	printf("'CTRL+S'\t\t- Stop generation.\n");
	printf("'CTRL+D'\t\t- Quit.\n");

	while (1) {

		// User Input
		if (read_user_input(input_buf, sizeof(input_buf)) == -2) {
			push_cmd((ModelCommand){.type = CMD_EXIT});
			break;
		}

		if (tab_pressed && think_active) {
			toggle_think_visibility();
			continue;
		}

		if (strncmp(input_buf, "/exit", 5) == 0) {
			push_cmd((ModelCommand){.type = CMD_EXIT});
			break;
		}

		if (strncmp(input_buf, "/img ", 5) == 0) {
			const char *path = input_buf + 5;

			if (!ctx->model_vision) {
				printf(ERR "[Vision is not supported]\n" CLR_RESET);
				continue;
			}

			if (load_bmp_clip(path, ctx) == 0) {
				printf(DEBUG "[Image loaded: %s]\n" CLR_RESET, path);
				image_loaded = true;
			} else {
				printf(ERR "[Failed to load image: %s]\n" CLR_RESET, path);
			}

			continue; // Go back and wait for actual user message
		}

		// Send Work to Model
		ModelCommand cmd = {.type = CMD_GENERATE,
				    .text = strdup(input_buf), // Duplicate because model thread frees it
				    .has_image = image_loaded};
		push_cmd(cmd);
		image_loaded = false;

		// Listening Mode (Blocking on Event Queue with Timeout)
		while (1) {

			// Check for User Input
			check_generation_interaction(ctx);

			// Wait for Event (50ms)
			ModelEvent evt = pop_evt_timed(50);

			if (evt.type == EVT_TIMEOUT) {
				continue;

			} else if (evt.type == EVT_FINISHED) {
				break;

			} else if (evt.type == EVT_TOKEN) {

				// Decode and print on UI thread
				generate_output(ctx, evt.token_id);

			} else if (evt.type == EVT_TOOL_CALL) {

				tools_execute_pending(ctx);

				// Send the result back to the model
				ToolContext *tc = &ctx->tool_context;
				tc->state = TOOL_STATE_IDLE;

				ModelCommand cmd = {
					.type = CMD_GENERATE, .text = tc->result_prompt, .has_image = false};

				push_cmd(cmd);

				printf(DEBUG "[Tool] Result sent to model: %s\n" CLR_RESET, tc->result_prompt);
			}
		}
	}

	printf(DEBUG);

	// exit_app:
	//	pthread_join(thread_id, NULL);
}

static void handle_model(AppConfig *cfg, const char *v)
{
	cfg->model_path = (char *)v;
}
static void handle_threads(AppConfig *cfg, const char *v)
{
	cfg->num_threads = atoi(v);
}
static void handle_mmproj(AppConfig *cfg, const char *v)
{
	cfg->mmproj_path = (char *)v;
}
static void handle_context(AppConfig *cfg, const char *v)
{
	cfg->context_length = atoi(v);
}
static void handle_temp(AppConfig *cfg, const char *v)
{
	cfg->temperature = atof(v);
}
static void handle_top_p(AppConfig *cfg, const char *v)
{
	cfg->top_p = atof(v);
}
static void handle_top_k(AppConfig *cfg, const char *v)
{
	cfg->top_k = atoi(v);
}
static void handle_mmap(AppConfig *cfg, const char *v)
{
	cfg->use_mmap = 1;
}

static void print_help(const char *progname);

static void handle_help(AppConfig *cfg, const char *v)
{
	print_help(v);
	exit(EXIT_SUCCESS);
}

static const ArgSpec arg_table[] = {
	{"model", 'm', 1, handle_model, "Path to model GGUF file"},
	{"threads", 't', 1, handle_threads, "Number of CPU threads"},
	{"mmproj", 'p', 1, handle_mmproj, "Multimodal projection file"},
	{"context", 'c', 1, handle_context, "Context window size"},
	{"temperature", 'T', 1, handle_temp, "Sampling temperature"},
	{"top-p", 'P', 1, handle_top_p, "Nucleus sampling (top-p)"},
	{"top-k", 'K', 1, handle_top_k, "Top-k sampling"},
	{"mmap", 'M', 0, handle_mmap, "Use mmap to load model"},
	{"help", 'h', 0, handle_help, "Show help message"},
};
static const int arg_table_count = sizeof(arg_table) / sizeof(arg_table[0]);

static const ArgSpec *find_long_opt(const char *name)
{
	for (int i = 0; i < arg_table_count; i++)
		if (strcmp(arg_table[i].long_opt, name) == 0)
			return &arg_table[i];
	return NULL;
}

static const ArgSpec *find_short_opt(char c)
{
	for (int i = 0; i < arg_table_count; i++)
		if (arg_table[i].short_opt == c)
			return &arg_table[i];
	return NULL;
}

static void print_help(const char *progname)
{
	printf("Usage: %s [OPTIONS]\n\n", progname);
	printf("Options:\n");

	int max_len = 0;
	for (int i = 0; i < arg_table_count; i++) {
		int len = 2; // two spaces
		if (arg_table[i].short_opt)
			len += 2; // "-x"
		if (arg_table[i].long_opt)
			len += 2 + strlen(arg_table[i].long_opt); // ", --long"
		if (arg_table[i].requires_value)
			len += 6; // " <val>"
		if (len > max_len)
			max_len = len;
	}

	// Print options aligned
	for (int i = 0; i < arg_table_count; i++) {
		const ArgSpec *a = &arg_table[i];
		char buf[128];

		if (a->requires_value) {
			snprintf(buf, sizeof(buf), "  -%c, --%s <val>", a->short_opt, a->long_opt);
		} else {
			snprintf(buf, sizeof(buf), "  -%c, --%s", a->short_opt, a->long_opt);
		}

		printf("  %-*s %s\n", max_len, buf, a->description);
	}

	printf("\n");
}

AppConfig parse_args(int argc, char *argv[])
{
	AppConfig config = {0};
	config.num_threads = 4;
	config.temperature = 0.7f;
	config.top_p = 0.95f;
	config.top_k = 20;

	if (argc == 1) {
		print_help(argv[0]);
		exit(EXIT_SUCCESS);
	}

	for (int i = 1; i < argc; i++) {
		const char *arg = argv[i];

		// Long option: --something
		if (strncmp(arg, "--", 2) == 0) {
			const char *name = arg + 2;
			const ArgSpec *spec = find_long_opt(name);
			if (!spec) {
				fprintf(stderr, "Unknown option: %s\n", arg);
				exit(1);
			}

			const char *value = NULL;
			if (spec->requires_value) {
				if (i + 1 >= argc) {
					fprintf(stderr, "Option --%s requires a value\n", name);
					exit(1);
				}
				value = argv[++i];
			}

			spec->handler(&config, value);
			continue;
		}

		// Short option: -x
		if (arg[0] == '-' && arg[1] != '\0') {
			const ArgSpec *spec = find_short_opt(arg[1]);
			if (!spec) {
				fprintf(stderr, "Unknown option: %s\n", arg);
				exit(1);
			}

			const char *value = NULL;
			if (spec->requires_value) {
				if (i + 1 >= argc) {
					fprintf(stderr, "Option -%c requires a value\n", arg[1]);
					exit(1);
				}
				value = argv[++i];
			} else {
				value = argv[0];
			}

			if (spec->handler) {
				spec->handler(&config, value);
			}
			continue;
		}

		fprintf(stderr, "Unknown argument: %s\n", arg);
		exit(1);
	}

	return config;
}

char banner[] = {
	"\n"
	"████████╗██╗███████╗  TIE\n"
	"╚══██╔══╝██║██╔════╝  Toy Inference Engine v%u.%u\n"
	"   ██║   ██║█████╗\n"
	"   ██║   ██║██╔══╝\n"
	"   ██║   ██║███████╗\n"
	"   ╚═╝   ╚═╝╚══════╝\n"};

int main(int argc, char *argv[])
{

	printf(banner, VERSION_MAJOR, VERSION_MINOR);
	// printf("Toy Inference Engine v%u.%u\n", VERSION_MAJOR, VERSION_MINOR);

	setlocale(LC_ALL, "en_US.UTF-8");

	struct TIEContext *ctx = malloc(sizeof(struct TIEContext));
	if (!ctx)
		exit(EXIT_FAILURE);

	ctx->config = parse_args(argc, argv);

	// Init Engine
	engine_alloc(ctx, ctx->config.num_threads);
	input_init();

	// Load Models
	ctx->gguf_text = gguf_model_parse(ctx->config.model_path);
	if (!ctx->gguf_text)
		exit(EXIT_FAILURE);

	if (ctx->config.mmproj_path)
		ctx->gguf_vision = gguf_model_parse(ctx->config.mmproj_path);

	ModelDef *text_def = find_model_def(ctx->gguf_text);
	ModelDef *vision_def = ctx->gguf_vision ? find_model_def(ctx->gguf_vision) : NULL;

	if (model_load(ctx, ctx->gguf_text, (void **)&ctx->model, text_def, ctx->config.use_mmap) != 0) {
		printf("Failed to load language model %s\n", ctx->config.model_path);
		exit(EXIT_FAILURE);
	}
	ctx->model->def = text_def;

	if (ctx->gguf_vision) {
		if (vision_def == NULL) {
			printf("Unsupported vision model\n");
			exit(EXIT_FAILURE);
		}
		if (model_load(ctx, ctx->gguf_vision, (void **)&ctx->model_vision, vision_def, ctx->config.use_mmap)
		    != 0) {
			printf("Failed to load vision model %s\n", ctx->config.mmproj_path);
			exit(EXIT_FAILURE);
		}
		ctx->model_vision->def = vision_def;
	}

	// Allocate Memory & Init Weights
	model_language_init(ctx, (void *)ctx->model, text_def, 1.0f, 1.0f);
	language_model_info(ctx);

	if (ctx->model_vision) {
		model_vision_init(ctx, (void *)ctx->model_vision, vision_def);
		vision_model_info(ctx);
	}

#if 0
	char buf[2048];
	for (int i=0; i < ctx->model->vocab_size; i++) {
		const unsigned char *token = get_token_string(ctx, i);
		int len = get_token_string_length(ctx, i);

		memset(buf, 0, sizeof(buf));
		memcpy(buf, token, len);

		printf("token #%u [%s]\n", i, buf);
	}

	fflush(stdout);
#endif

#if 0
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
#endif

	start_think = get_special_token_id(ctx, "<think>", -1);
	end_think = get_special_token_id(ctx, "</think>", -1);

	// Run Chat
	run_ui_loop(ctx);

	// Cleanup
	engine_release(ctx);
	input_shutdown();

	printf(CLR_RESET "Done.\n");
	exit(EXIT_SUCCESS);

	return 0;
}
