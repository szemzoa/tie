#ifndef __GGUF_H__
#define __GGUF_H__

#include <inttypes.h>

enum gguf_metadata_value_type {
	// The value is a 8-bit unsigned integer.
	GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
	// The value is a 8-bit signed integer.
	GGUF_METADATA_VALUE_TYPE_INT8 = 1,
	// The value is a 16-bit unsigned little-endian integer.
	GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
	// The value is a 16-bit signed little-endian integer.
	GGUF_METADATA_VALUE_TYPE_INT16 = 3,
	// The value is a 32-bit unsigned little-endian integer.
	GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
	// The value is a 32-bit signed little-endian integer.
	GGUF_METADATA_VALUE_TYPE_INT32 = 5,
	// The value is a 32-bit IEEE754 floating point number.
	GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
	// The value is a boolean.
	// 1-byte value where 0 is false and 1 is true.
	// Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
	GGUF_METADATA_VALUE_TYPE_BOOL = 7,
	// The value is a UTF-8 non-null-terminated string, with length prepended.
	GGUF_METADATA_VALUE_TYPE_STRING = 8,
	// The value is an array of other values, with the length and type prepended.
	///
	// Arrays can be nested, and the length of the array is the number of elements in the array, not the number of
	// bytes.
	GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
	// The value is a 64-bit unsigned little-endian integer.
	GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
	// The value is a 64-bit signed little-endian integer.
	GGUF_METADATA_VALUE_TYPE_INT64 = 11,
	// The value is a 64-bit IEEE754 floating point number.
	GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
};

enum ggml_type {
	GGML_TYPE_F32 = 0,
	GGML_TYPE_F16 = 1,
	GGML_TYPE_Q4_0 = 2,
	GGML_TYPE_Q4_1 = 3,
	GGML_TYPE_Q5_0 = 6,
	GGML_TYPE_Q5_1 = 7,
	GGML_TYPE_Q8_0 = 8,
	GGML_TYPE_Q8_1 = 9,
	GGML_TYPE_Q2_K = 10,
	GGML_TYPE_Q3_K = 11,
	GGML_TYPE_Q4_K = 12,
	GGML_TYPE_Q5_K = 13,
	GGML_TYPE_Q6_K = 14,
	GGML_TYPE_Q8_K = 15,
	GGML_TYPE_IQ2_XXS = 16,
	GGML_TYPE_IQ2_XS = 17,
	GGML_TYPE_IQ3_XXS = 18,
	GGML_TYPE_IQ1_S = 19,
	GGML_TYPE_IQ4_NL = 20,
	GGML_TYPE_IQ3_S = 21,
	GGML_TYPE_IQ2_S = 22,
	GGML_TYPE_IQ4_XS = 23,
	GGML_TYPE_I8 = 24,
	GGML_TYPE_I16 = 25,
	GGML_TYPE_I32 = 26,
	GGML_TYPE_I64 = 27,
	GGML_TYPE_F64 = 28,
	GGML_TYPE_IQ1_M = 29,
	GGML_TYPE_BF16 = 30,
	GGML_TYPE_TQ1_0 = 34,
	GGML_TYPE_TQ2_0 = 35,
	GGML_TYPE_COUNT = 39,
};

typedef struct gguf_tensor_t {
	char *name;
	uint64_t dimensions[4];
	uint32_t n_dims;
	uint32_t type;
	void *data;
	uint64_t offset;
	uint64_t size;
} gguf_tensor;

struct gguf_metadata_kv_t {
	char *name;
	uint32_t type;
	uint64_t size;
	void *arr_offset;
	void *data;
};

#define QK_K 256

// A block of Q6_K quantized weights.
typedef struct {
	uint8_t ql[128];   // quants, lower 4 bits
	uint8_t qh[64];	   // quants, upper 2 bits
	int8_t scales[16]; // scales, 4-bit+4-bit, signed
	uint16_t d;	   // super-block scale (fp16)
} block_q6_k;

typedef struct {
	uint16_t d;	    // ggml_half (IEEE 754 half-precision)
	uint16_t dmin;	    // ggml_half
	uint8_t scales[12]; // Packed 6-bit scale/min pairs
	uint8_t qs[128];    // Packed 4-bit quantized weights
} block_q4_k;

struct ctx_t;

extern int gguf_read(struct ctx_t *ctx, char *path);
extern void gguf_close(struct ctx_t *ctx);

extern int gguf_read_metadata_type_string(struct ctx_t *ctx, struct gguf_metadata_kv_t *metadata);
extern int gguf_get_metadata_value(struct ctx_t *ctx, char *key, void *value);
extern char *gguf_get_metadata_string(struct ctx_t *ctx, char *key);
extern int gguf_get_metadata_size(struct ctx_t *ctx, char *key, uint64_t *size);
extern int gguf_metadata_read_tokens_embed(struct ctx_t *ctx, char *key);
extern int gguf_metadata_read_merges(struct ctx_t *ctx, char *key);

extern int gguf_map_weights(struct ctx_t *ctx);
extern void *get_tensor_data_ptr(struct ctx_t *ctx_ptr, const char *name, uint64_t *size_bytes);
extern gguf_tensor *get_tensor(struct ctx_t *ctx_ptr, const char *name);

extern void dump_tensors(struct ctx_t *ctx);

#endif
