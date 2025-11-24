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

typedef struct {
	char *model_path;
	char *mmproj_path;
	char *image_path;
	int context_length;
	int num_threads;
	int use_mmap;
} AppConfig;

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
