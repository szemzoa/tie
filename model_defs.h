#ifndef __MODEL_DEFS_H__
#define __MODEL_DEFS_H__

#include "model.h"

extern ModelDef QWEN3_DEF;
extern ModelDef QWEN3_MOE_DEF;
extern ModelDef GEMMA3_DEF;
extern ModelDef GEMMA3N_DEF;
extern ModelDef GEMMA3_CLIP_DEF;
extern ModelDef QWEN3VL_DEF;
extern ModelDef QWEN3VL_CLIP_DEF;
extern ModelDef QWEN3VL_MOE_DEF;
extern ModelDef DEEPSEEK_QWEN3_DEF;

// Common language model definitions
#define DECLARE_LANGUAGE_MODEL_DEF(model, layer_weights)		\
    .metadata_defs = model##_METADATA_DEFS,				\
    .num_metadata_defs = ARRAY_SIZE(model##_METADATA_DEFS),		\
    .global_tensors = model##_GLOBAL_TENSORS,				\
    .num_global_tensors = ARRAY_SIZE(model##_GLOBAL_TENSORS),		\
    .layer_tensors = layer_weights##_LAYER_TENSORS,			\
    .num_layer_tensors = ARRAY_SIZE(layer_weights##_LAYER_TENSORS),	\
    .buffer_defs = model##_BUFFERS,					\
    .num_buffer_defs = ARRAY_SIZE(model##_BUFFERS),			\
    .tokenize_defs = &model##_TOKENIZE_DEF,				\
    .struct_size = sizeof(Model),					\
    .layers_offset = offsetof(Model, layers),				\
    .layer_struct_size = sizeof(LayerWeights),				\
    .num_layers_offset = offsetof(Model, num_layers),			\


// Base metadata for Llama-like models
#define DECLARE_LLAMA_BASE_METADATA_DEFS \
    {"%s.context_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, seq_length), false, false}, 			\
    {"%s.embedding_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, embed_dim), false, false}, 		\
    {"%s.block_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, num_layers), false, false}, 			\
    {"%s.feed_forward_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, ffn_dim), false, false}, 		\
    {"%s.attention.head_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, num_heads), false, false}, 		\
    {"%s.attention.layer_norm_rms_epsilon", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, norm_eps), false, false}, \
    {"%s.attention.key_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, head_dim), false, false}, 		\
    {"%s.rope.freq_base", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(Model, rope_freq_base), false, false}, 		\
    {"%s.attention.head_count_kv", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, num_kv_heads), false, false}

// Base metadata for tokenizer
#define DECLARE_TOKENIZER_BASE_METADATA_DEFS \
    {"tokenizer.ggml.eos_token_id", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, eos_token_id), false, true}, 	\
    {"tokenizer.ggml.bos_token_id", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, bos_token_id), false, true}, 	\
    {"tokenizer.ggml.unknown_token_id", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, unk_token_id), false, true}, 	\
    {"tokenizer.ggml.pad_token_id", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, pad_token_id), false, true}, 	\
    {"tokenizer.ggml.add_bos_token", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, add_bos_token), false, true}, 	\
    {"tokenizer.ggml.add_eos_token", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(Model, add_eos_token), false, true}

// Base global tensors
#define DECLARE_GLOBAL_TENSORS_BASE_DEFS \
    {"token_embd.weight", offsetof(Model, token_embd), FLAG_NONE},	\
    {"output_norm.weight", offsetof(Model, output_norm), FLAG_NONE},	\
    {"output.weight", offsetof(Model, output), FLAG_OPTIONAL}

// Common vision model definitions
#define DECLARE_VISION_MODEL_DEF(model)				\
    .metadata_defs = model##_METADATA_DEFS,			\
    .num_metadata_defs = ARRAY_SIZE(model##_METADATA_DEFS),	\
    .global_tensors = model##_GLOBAL_TENSORS,			\
    .num_global_tensors = ARRAY_SIZE(model##_GLOBAL_TENSORS),	\
    .layer_tensors = model##_LAYER_TENSORS,			\
    .num_layer_tensors = ARRAY_SIZE(model##_LAYER_TENSORS),	\
    .buffer_defs = model##_BUFFERS,				\
    .num_buffer_defs = ARRAY_SIZE(model##_BUFFERS),		\
    .struct_size = sizeof(VisionModel),				\
    .layers_offset = offsetof(VisionModel, layers),		\
    .layer_struct_size = sizeof(VisionLayerWeights),		\
    .num_layers_offset = offsetof(VisionModel, num_layers)

#define DECLARE_BASE_VISION_METADATA_DEFS \
    {"%s.has_vision_encoder", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(VisionModel, has_vision_encoder), false, false},	\
    {"%s.vision.projection_dim", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(VisionModel, projection_dim), false, false},		\
    {"%s.vision.image_size", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(VisionModel, image_size), false, false},		\
    {"%s.vision.patch_size", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(VisionModel, patch_size), false, false},			\
    {"%s.vision.embedding_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(VisionModel, embed_dim), false, false},		\
    {"%s.vision.block_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(VisionModel, num_layers), false, false},		\
    {"%s.vision.feed_forward_length", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(VisionModel, ffn_dim), false, false},		\
    {"%s.vision.attention.head_count", GGUF_METADATA_VALUE_TYPE_UINT32, offsetof(VisionModel, num_heads), false, false},	\
    {"%s.vision.attention.layer_norm_epsilon", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(VisionModel, norm_eps), false, false},\
    {"%s.vision.image_mean", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(VisionModel, image_mean), true, false},			\
    {"%s.vision.image_std", GGUF_METADATA_VALUE_TYPE_FLOAT32, offsetof(VisionModel, image_std), true, false}


#endif

