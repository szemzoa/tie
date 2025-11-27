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
#include <readline/readline.h>
#include <readline/history.h>

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

CommandQueue cmd_q = {
	.mutex = PTHREAD_MUTEX_INITIALIZER,
	.cond = PTHREAD_COND_INITIALIZER
};

EventQueue evt_q = {
	.mutex = PTHREAD_MUTEX_INITIALIZER,
	.cond = PTHREAD_COND_INITIALIZER
};


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
	if (def->params.sot_token_id != -1)
		prompt_tokens[prompt_len++] = def->params.sot_token_id;

	prompt_tokens[prompt_len++] = def->params.role_user_token_id;

	if (def->params.newline_token_id != -1)
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
	if (def->params.eot_token_id != -1)
		prompt_tokens[prompt_len++] = def->params.eot_token_id;

	if (def->params.newline_token_id != -1)
		prompt_tokens[prompt_len++] = def->params.newline_token_id;

	if (def->params.sot_token_id != -1)
		prompt_tokens[prompt_len++] = def->params.sot_token_id;

	prompt_tokens[prompt_len++] = def->params.role_model_token_id;

	if (def->params.newline_token_id != -1)
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

	printf(DEBUG "--- Prefill Start: pos %u, len %zu ---\n" CLR_RESET, ctx->kv_pos, prompt_len);
	// Custom Prompt Processor (if defined)
	if (ctx->model->interface.process_prompt) {
		ctx->model->interface.process_prompt(ctx, prompt_tokens, prompt_len);
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

	// Run Transformer Layers (Batch Processing)
	process_embeddings(ctx, &ctx->mem.hidden_state, current_pos);

	clock_gettime(CLOCK_REALTIME, &end);
	float tps = (float)prompt_len / (elapsed_time_us(end, start) / 1000000.0f);
	printf(DEBUG "--- Prefill Complete: %llu ms %.2f tps---\n" CLR_RESET, elapsed_time_us(end, start) / 1000, tps);
	return 0;
}


// Decode (Generate Tokens)
int engine_decode(struct TIEContext *ctx, int max_new_tokens, int *out_tokens, size_t prompt_len,
		  token_callback_t on_token, void *user_data)
{
	struct timespec start, end;
	int gen_len = 0;
	Tensor *output_tensor = ctx->model->output.mem.data == NULL ? &ctx->model->token_embd : &ctx->model->output;

	printf(DEBUG "\n--- Generation Start ---\n" ASSISTANT_OUT);
	clock_gettime(CLOCK_REALTIME, &start);

	printf(CLR_RESET ASSISTANT_OUT);

	for (int step = 0; step < max_new_tokens; step++) {

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

		// Run Next Token through Transformer
		ctx->model->interface.prepare_next_token(ctx, next_token);

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

void model_on_token(void *user_data, int token_id)
{
	ModelEvent evt = {
		.type = EVT_TOKEN,
		.token_id = token_id
	};

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

			// Build Prompt
			size_t prompt_len = 0;
			// cmd.text is owned by this thread now, free it later
			int *tokens = build_multimodal_turn_tokens(ctx, cmd.text, cmd.has_image, &prompt_len);

			if (!tokens) {
				// Error handling: push a FINISH event immediately
				push_evt((ModelEvent) { .type = EVT_FINISHED });

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
			engine_decode(ctx, 8192, generated_tokens, prompt_len, model_on_token, NULL);

			// Handle Result
			if (ctx->tool_context.state == TOOL_STATE_CALL_READY) {
				// Model wants to call a tool.
				push_evt((ModelEvent){.type = EVT_TOOL_CALL});
				continue;
			}

			// Cleanup & Signal Finish
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

static char *file_path_generator(const char *text, int state)
{
	static DIR *dir = NULL;
	static char directory[4096];
	static char prefix[4096];
	struct dirent *entry;

	// First call: reset state
	if (state == 0) {
		// Split into directory + prefix
		const char *slash = strrchr(text, '/');

		if (slash) {
			size_t len = slash - text + 1;
			strncpy(directory, text, len);
			directory[len] = '\0';

			strcpy(prefix, slash + 1);
		} else {
			strcpy(directory, "./");
			strcpy(prefix, text);
		}

		if (dir)
			closedir(dir);
		dir = opendir(directory);
		if (!dir)
			return NULL;
	}

	// Enumerate directory entries
	while ((entry = readdir(dir)) != NULL) {
		if (strncmp(entry->d_name, prefix, strlen(prefix)) == 0) {

			static char buffer[4096];
			snprintf(buffer, sizeof(buffer), "%s%s", directory, entry->d_name);

			// If directory, add trailing '/'
			struct stat st;
			if (stat(buffer, &st) == 0 && S_ISDIR(st.st_mode)) {
				strcat(buffer, "/");
			}

			return strdup(buffer);
		}
	}

	closedir(dir);
	dir = NULL;

	return NULL;
}

static char **tie_completion(const char *text, int start, int end)
{
	// Trigger file completion only for "/img <PATH>"
	if (strncmp(rl_line_buffer, "/img ", 5) == 0) {
		// Complete only after "/img " including multiple directories
		return rl_completion_matches(text, file_path_generator);
	}

	return NULL;
}

void init_readline()
{
	rl_attempted_completion_function = tie_completion;
	rl_completion_append_character = '\0';

#ifdef RL_COMPLETION_SUPPRESS_QUOTE
	rl_completion_suppress_quote = 1;
#endif
#ifdef RL_COMPLETER_QUOTE_CHARACTERS
	rl_completer_quote_characters = "";
#endif
}

void read_user_input(char *input_buf, size_t buf_size)
{
	char *line = readline(USER_PROMPT "\nYou: ");
	if (!line) {
		input_buf[0] = '\0';
		return;
	}

	strncpy(input_buf, line, buf_size);
	free(line);
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

	printf(ERR "\nWelcome to interactive chat. Type '/exit' to quit, '/img /path/xxx.bmp' to load an image.\n");

	while (1) {

		// User Input
		read_user_input(input_buf, sizeof(input_buf));

		if (strncmp(input_buf, "/exit", 5) == 0) {
			push_cmd((ModelCommand){.type = CMD_EXIT});
			break;
		}

		if (strncmp(input_buf, "/img ", 5) == 0) {
			const char *path = input_buf + 5;

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

		// Listening Mode (Blocking on Event Queue)
		while (1) {

			ModelEvent evt = pop_evt();

			if (evt.type == EVT_FINISHED) {
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

				// We stay in Listening Mode because the model will immediately start generating
				// the answer after processing the tool result.
				printf(DEBUG "[Tool] Result sent to model: %s\n" CLR_RESET, tc->result_prompt);
			}
		}
	}

	printf(DEBUG);

//exit_app:
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

static const ArgSpec arg_table[] = {
	{"model", 'm', 1, handle_model, "Path to model GGUF file"},
	{"threads", 't', 1, handle_threads, "Number of CPU threads"},
	{"mmproj", 'p', 1, handle_mmproj, "Multimodal projection file"},
	{"context", 'c', 1, handle_context, "Context window size"},
	{"temperature", 'T', 1, handle_temp, "Sampling temperature"},
	{"top-p", 'P', 1, handle_top_p, "Nucleus sampling (top-p)"},
	{"top-k", 'K', 1, handle_top_k, "Top-k sampling"},
	{"mmap", 'M', 0, handle_mmap, "Use mmap to load model"},
	{"help", 'h', 0, NULL, "Show help message"},
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
		exit(0);
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
			}

			spec->handler(&config, value);
			continue;
		}

		fprintf(stderr, "Unknown argument: %s\n", arg);
		exit(1);
	}

	return config;
}


int main(int argc, char *argv[])
{
	printf("Toy Inference Engine v%u.%u\n", VERSION_MAJOR, VERSION_MINOR);

	setlocale(LC_ALL, "en_US.UTF-8");

	struct TIEContext *ctx = malloc(sizeof(struct TIEContext));
	if (!ctx)
		exit(EXIT_FAILURE);

	ctx->config = parse_args(argc, argv);

	// Init Engine
	engine_alloc(ctx, ctx->config.num_threads);
	init_readline();

	// Load Models
	ctx->gguf_text = gguf_model_parse(ctx->config.model_path);
	if (!ctx->gguf_text)
		exit(EXIT_FAILURE);

	if (ctx->config.mmproj_path)
		ctx->gguf_vision = gguf_model_parse(ctx->config.mmproj_path);

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


	// Run Chat
	run_ui_loop(ctx);

	// Cleanup
	engine_release(ctx);
	printf("Done.\n");

	exit(EXIT_SUCCESS);

	return 0;
}
