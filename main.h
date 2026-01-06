#ifndef __MAIN_H__
#define __MAIN_H__

#include <stddef.h>
#include <signal.h>
#include <pthread.h>
#include "config.h"
#include "tokenize.h"
#include "gguf.h"
#include "engine.h"
#include "model.h"
#include "tools.h"

typedef enum {
	MODEL_TYPE_TEXT,
	MODEL_TYPE_VISION,
} ModelType;

#define QUEUE_SIZE 16

typedef enum { CMD_GENERATE, CMD_STOP, CMD_EXIT } CmdType;

typedef struct {
	CmdType type;
	char *text; // User input or Tool result
	bool has_image;
	bool prefill_only;
} ModelCommand;

typedef enum {
	EVT_TOKEN,     // A new token generated
	EVT_FINISHED,  // Generation complete
	EVT_TOOL_CALL, // Model wants to call a tool (pause generation)
	EVT_TIMEOUT,
} EvtType;

typedef struct {
	EvtType type;
	int token_id;
} ModelEvent;

typedef struct {
	ModelCommand buffer[QUEUE_SIZE];
	int head, tail;
	pthread_mutex_t mutex;
	pthread_cond_t cond;
} CommandQueue;

typedef struct {
	ModelEvent buffer[QUEUE_SIZE * 4]; // Larger buffer for tokens
	int head, tail;
	pthread_mutex_t mutex;
	pthread_cond_t cond;
} EventQueue;

typedef void (*token_callback_t)(void *user_data, int token_id);

typedef struct {
	char *model_path;
	char *mmproj_path;
	char *image_path;
	int context_length;
	int num_threads;
	int use_mmap;

	float temperature;
	float top_p;
	int top_k;
} AppConfig;

typedef struct {
	const char *long_opt;
	char short_opt;
	int requires_value;
	void (*handler)(AppConfig *cfg, const char *value);
	const char *description;
} ArgSpec;

struct GGUFModel {
	int fd;
	size_t file_size;
	void *mapped_data;
	uint8_t *fptr;

	uint64_t metadata_num;
	uint64_t tensor_count;
	uint64_t tensor_loaded;
	uint64_t tensor_data_offset;

	GGUFMetadata *metadata;
	GGUFTensor *tensors;

	int arch;
};

struct TIEContext {
	AppConfig config;

	struct GGUFModel *gguf_text;   // Container for the main text model GGUF
	struct GGUFModel *gguf_vision; // Container for the mmproj GGUF (can be NULL)

	Model *model;
	VisionModel *model_vision;

	MemLayout mem;
	MemLayoutVision vision_mem;

	uint32_t kv_pos;
	LayerKVCache *kv_cache;

	Tokenizer tokenizer;
	ToolContext tool_context;

	volatile bool stop_generation;
	volatile bool pause_generation;
};

#endif
