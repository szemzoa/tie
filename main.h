#ifndef __MAIN_H__
#define __MAIN_H__

#include <stddef.h>
#include "config.h"
#include "tokenize.h"
#include "gguf.h"
#include "engine.h"
#include "model.h"

typedef enum {
	MODEL_TYPE_TEXT,
	MODEL_TYPE_VISION,
} ModelType;

enum {
	TOOL_CALL_STATE_UNINITIALIZED = 0,
	TOOL_CALL_STATE_IDLE,
	TOOL_CALL_STATE_PROCESSING,
	TOOL_CALL_STATE_END,
};

struct tool_call_t {
	int state;
	int token_start;
	int token_end;
	char buffer[TOOL_CALL_BUFFER_SIZE];
	int len;
	char *result;
};

typedef char *(*tool_func_t)(const char *);

struct tool_entry_t {
	const char *name;
	tool_func_t func;
};

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
	struct GGUFModel *gguf_text;   // Container for the main text model GGUF
	struct GGUFModel *gguf_vision; // Container for the mmproj GGUF (can be NULL)

	Model *model;
	VisionModel *model_vision;

	MemLayout mem;
	MemLayoutVision vision_mem;

	uint32_t kv_pos;
	LayerKVCache *kv_cache;

	RopeCacheType *rope_cache_local;
	RopeCacheType *rope_cache_global;

	unsigned int utf8_state;
	unsigned int utf8_codepoint;

	Tokenizer tokenizer;
};

#endif
