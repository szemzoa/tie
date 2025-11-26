#ifndef __MAIN_H__
#define __MAIN_H__

#include <stddef.h>
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

#define CLR_RESET   "\033[0m"
#define CLR_BOLD    "\033[1m"
#define CLR_DIM     "\033[2m"

#define CLR_RED     "\033[31m"
#define CLR_GREEN   "\033[32m"
#define CLR_YELLOW  "\033[33m"
#define CLR_BLUE    "\033[34m"
#define CLR_MAGENTA "\033[35m"
#define CLR_CYAN    "\033[36m"
#define CLR_WHITE   "\033[37m"

// helpers
#define USER_PROMPT   	CLR_BOLD CLR_GREEN
#define ASSISTANT_OUT 	CLR_CYAN
#define DEBUG          	CLR_DIM CLR_YELLOW
#define THINK          	CLR_DIM CLR_MAGENTA
#define ERR            	CLR_BOLD CLR_RED

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
};

#endif
