#include <inttypes.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <unistd.h>

#include "gguf.h"
#include "main.h"
#include "threadpool.h"
#include "tokenize.h"

int gguf_read_metadata_type_uint8(struct ctx_t *ctx, struct gguf_metadata_kv_t *metadata)
{
	if ((metadata->data = (uint8_t *)malloc(sizeof(uint8_t))) == NULL) {
		return -1;
	}

	*(uint8_t *)metadata->data = *(uint8_t *)ctx->fptr;
	ctx->fptr += sizeof(uint8_t);
	metadata->size = sizeof(uint8_t);

	return 0;
}

int gguf_read_metadata_type_int8(struct ctx_t *ctx, struct gguf_metadata_kv_t *metadata)
{
	if ((metadata->data = (int8_t *)malloc(sizeof(int8_t))) == NULL) {
		return -1;
	}

	*(int8_t *)metadata->data = *(int8_t *)ctx->fptr;
	ctx->fptr += sizeof(int8_t);
	metadata->size = sizeof(int8_t);

	return 0;
}

int gguf_read_metadata_type_uint16(struct ctx_t *ctx, struct gguf_metadata_kv_t *metadata)
{
	if ((metadata->data = (uint16_t *)malloc(sizeof(uint16_t))) == NULL) {
		return -1;
	}

	*(uint16_t *)metadata->data = *(uint16_t *)ctx->fptr;
	ctx->fptr += sizeof(uint16_t);
	metadata->size = sizeof(uint16_t);

	return 0;
}

int gguf_read_metadata_type_int16(struct ctx_t *ctx, struct gguf_metadata_kv_t *metadata)
{
	if ((metadata->data = (int16_t *)malloc(sizeof(int16_t))) == NULL) {
		return -1;
	}

	*(int16_t *)metadata->data = *(int16_t *)ctx->fptr;
	ctx->fptr += sizeof(int16_t);
	metadata->size = sizeof(int16_t);

	return 0;
}

int gguf_read_metadata_type_uint32(struct ctx_t *ctx, struct gguf_metadata_kv_t *metadata)
{
	if ((metadata->data = (uint32_t *)malloc(sizeof(uint32_t))) == NULL) {
		return -1;
	}

	*(uint32_t *)metadata->data = *(uint32_t *)ctx->fptr;
	ctx->fptr += sizeof(uint32_t);
	metadata->size = sizeof(uint32_t);

	return 0;
}

int gguf_read_metadata_type_int32(struct ctx_t *ctx, struct gguf_metadata_kv_t *metadata)
{
	if ((metadata->data = (int32_t *)malloc(sizeof(int32_t))) == NULL) {
		return -1;
	}

	*(int32_t *)metadata->data = *(int32_t *)ctx->fptr;
	ctx->fptr += sizeof(int32_t);
	metadata->size = sizeof(int32_t);

	return 0;
}

int gguf_read_metadata_type_uint64(struct ctx_t *ctx, struct gguf_metadata_kv_t *metadata)
{
	if ((metadata->data = (uint64_t *)malloc(sizeof(uint64_t))) == NULL) {
		return -1;
	}

	*(uint64_t *)metadata->data = *(uint64_t *)ctx->fptr;
	ctx->fptr += sizeof(uint64_t);
	metadata->size = sizeof(uint64_t);

	return 0;
}

int gguf_read_metadata_type_int64(struct ctx_t *ctx, struct gguf_metadata_kv_t *metadata)
{
	if ((metadata->data = (int64_t *)malloc(sizeof(int64_t))) == NULL) {
		return -1;
	}

	*(int64_t *)metadata->data = *(int64_t *)ctx->fptr;
	ctx->fptr += sizeof(int64_t);
	metadata->size = sizeof(int64_t);

	return 0;
}

int gguf_read_metadata_type_fp32(struct ctx_t *ctx, struct gguf_metadata_kv_t *metadata)
{
	if ((metadata->data = (float *)malloc(sizeof(float))) == NULL) {
		return -1;
	}

	*(float *)metadata->data = *(float *)ctx->fptr;
	ctx->fptr += sizeof(float);
	metadata->size = sizeof(float);

	return 0;
}

int gguf_read_metadata_type_fp64(struct ctx_t *ctx, struct gguf_metadata_kv_t *metadata)
{
	if ((metadata->data = (double *)malloc(sizeof(double))) == NULL) {
		return -1;
	}

	*(double *)metadata->data = *(double *)ctx->fptr;
	ctx->fptr += sizeof(double);
	metadata->size = sizeof(double);

	return 0;
}

int gguf_read_metadata_type_string(struct ctx_t *ctx, struct gguf_metadata_kv_t *metadata)
{
	uint64_t size;
	char *str;

	size = *(uint64_t *)ctx->fptr;
	ctx->fptr += sizeof(uint64_t);

	if ((str = (char *)malloc(size + 1)) == NULL) {
		return -1;
	}

	memset(str, 0, size + 1);

	memcpy(str, ctx->fptr, size);
	ctx->fptr += size;
	metadata->size = size;

	metadata->data = (char *)str;
	return 0;
}

int gguf_read_metadata_type_array(struct ctx_t *ctx, struct gguf_metadata_kv_t *metadata)
{
	uint64_t i;
	uint64_t str_len;
	uint32_t type;
	uint64_t size;
	uint32_t val;

	type = *(uint32_t *)ctx->fptr;
	ctx->fptr += sizeof(uint32_t);

	size = *(uint64_t *)ctx->fptr;
	ctx->fptr += sizeof(uint64_t);
	metadata->size = size;
	metadata->arr_offset = ctx->fptr;

	for (i = 0; i < size; i++) {
		if (type == GGUF_METADATA_VALUE_TYPE_STRING) {
			str_len = *(uint64_t *)ctx->fptr;
			ctx->fptr += sizeof(uint64_t);

			ctx->fptr += str_len;
		}

		if (type == GGUF_METADATA_VALUE_TYPE_INT32) {
			val = *(uint32_t *)ctx->fptr;
			ctx->fptr += sizeof(uint32_t);
		}
	}

	return 0;
}

int gguf_get_metadata_value(struct ctx_t *ctx, char *key, void *value)
{
	struct gguf_metadata_kv_t *metadata;
	uint64_t i = 0;

	for (i = 0; i < ctx->metadata_kv_count; i++) {
		metadata = &ctx->metadata[i];

		if (!strcmp(ctx->metadata[i].name, key)) {
			memcpy((void *)value, metadata->data, metadata->size);
			return 0;
		}
	}

	return -1;
}

int gguf_get_metadata_size(struct ctx_t *ctx, char *key, uint64_t *size)
{
	struct gguf_metadata_kv_t *metadata;
	uint64_t i = 0;

	for (i = 0; i < ctx->metadata_kv_count; i++) {
		metadata = &ctx->metadata[i];

		if (!strcmp(ctx->metadata[i].name, key)) {
			*size = metadata->size;
			return 0;
		}
	}

	return -1;
}

#ifdef DEBUG_TENSORS
char *tensor_get_type_name(uint32_t type)
{
	switch (type) {
	case GGML_TYPE_F32:
		return "f32";
		break;
	case GGML_TYPE_F16:
		return "f16";
		break;
	case GGML_TYPE_Q4_0:
		return "q4_0";
		break;
	case GGML_TYPE_Q4_1:
		return "q4_1";
		break;
	case GGML_TYPE_Q5_0:
		return "q5_0";
		break;
	case GGML_TYPE_Q5_1:
		return "q5_1";
		break;
	case GGML_TYPE_Q8_0:
		return "q8_0";
		break;
	case GGML_TYPE_Q8_1:
		return "q8_1";
		break;
	case GGML_TYPE_Q2_K:
		return "q2_K";
		break;
	case GGML_TYPE_Q3_K:
		return "q3_K";
		break;
	case GGML_TYPE_Q4_K:
		return "q4_K";
		break;
	case GGML_TYPE_Q5_K:
		return "q5_K";
		break;
	case GGML_TYPE_Q6_K:
		return "q6_K";
		break;
	case GGML_TYPE_Q8_K:
		return "q8_K";
		break;
	case GGML_TYPE_IQ2_XXS:
		return "iq2_XXS";
		break;
	case GGML_TYPE_IQ2_XS:
		return "iq2_XS";
		break;
	case GGML_TYPE_IQ3_XXS:
		return "iq3_XXS";
		break;
	case GGML_TYPE_IQ1_S:
		return "iq1_S";
		break;
	case GGML_TYPE_IQ4_NL:
		return "iq4_NL";
		break;
	case GGML_TYPE_IQ3_S:
		return "iq3_S";
		break;
	case GGML_TYPE_IQ2_S:
		return "iq2_S";
		break;
	case GGML_TYPE_IQ4_XS:
		return "iq4_XS";
		break;
	case GGML_TYPE_I8:
		return "i8";
		break;
	case GGML_TYPE_I16:
		return "i16";
		break;
	case GGML_TYPE_I32:
		return "i32";
		break;
	case GGML_TYPE_I64:
		return "i64";
		break;
	case GGML_TYPE_F64:
		return "f64";
		break;
	case GGML_TYPE_IQ1_M:
		return "iq1_M";
		break;
	case GGML_TYPE_BF16:
		return "bf16";
		break;
	}

	return NULL;
}
#endif

void gguf_dump_metadata(struct ctx_t *ctx)
{
	uint64_t i = 0;
	struct gguf_metadata_kv_t *metadata;
	struct gguf_mdata_array_t *array;

	printf("gguf: metadata:\n");
	for (i = 0; i < ctx->metadata_kv_count; i++) {
		metadata = &ctx->metadata[i];

		printf("kv - #%3llu %-45s ", i, metadata->name);

		switch (metadata->type) {
		case GGUF_METADATA_VALUE_TYPE_UINT8:
			printf("u8\t = %u\n", *(uint8_t *)metadata->data);
			break;

		case GGUF_METADATA_VALUE_TYPE_INT8:
			printf("i8\t = %i\n", *(int8_t *)metadata->data);
			break;

		case GGUF_METADATA_VALUE_TYPE_UINT16:
			printf("u16\t = %u\n", *(uint16_t *)metadata->data);
			break;

		case GGUF_METADATA_VALUE_TYPE_INT16:
			printf("i16\t = %i\n", *(int16_t *)metadata->data);
			break;

		case GGUF_METADATA_VALUE_TYPE_UINT32:
			printf("u32\t = %u\n", *(uint32_t *)metadata->data);
			break;

		case GGUF_METADATA_VALUE_TYPE_INT32:
			printf("i32\t = %i\n", *(int32_t *)metadata->data);
			break;

		case GGUF_METADATA_VALUE_TYPE_FLOAT32:
			printf("fp32\t = %lf\n", *(float *)metadata->data);
			break;

		case GGUF_METADATA_VALUE_TYPE_BOOL:
			printf("bool\t = %s\n", *(uint8_t *)metadata->data == 0 ? "false" : "true");
			break;

		case GGUF_METADATA_VALUE_TYPE_STRING:
			printf("str\t = %.*s\n", 64, (char *)metadata->data);
			break;

		case GGUF_METADATA_VALUE_TYPE_ARRAY:
			array = (struct gguf_mdata_array_t *)metadata->data;
			printf("arr\t = \n");
			break;

		case GGUF_METADATA_VALUE_TYPE_UINT64:
			printf("u64\t = %llu\n", *(uint64_t *)metadata->data);
			break;

		case GGUF_METADATA_VALUE_TYPE_INT64:
			printf("i64\t = %lli\n", *(int64_t *)metadata->data);
			break;

		case GGUF_METADATA_VALUE_TYPE_FLOAT64:
			printf("fp64\t = %lf\n", *(double *)metadata->data);
			break;
		}
	}
}

void gguf_free_metadata(struct ctx_t *ctx)
{
	struct gguf_metadata_kv_t *metadata;
	uint64_t i;

	for (i = 0; i < ctx->metadata_kv_count; i++) {
		metadata = &ctx->metadata[i];

		if (metadata->name)
			free(metadata->name);

		if (metadata->data)
			free(metadata->data);
	}

	free(ctx->metadata);
}

int gguf_read_metadata(struct ctx_t *ctx)
{
	uint64_t i;
	struct gguf_metadata_kv_t *metadata;
	uint64_t key_len;
	int rc;

	if ((ctx->metadata =
		     (struct gguf_metadata_kv_t *)calloc(ctx->metadata_kv_count, sizeof(struct gguf_metadata_kv_t)))
	    == NULL) {
		perror("error alloc metadata info, OOM\n");
		return -1;
	}

	for (i = 0; i < ctx->metadata_kv_count; i++) {
		metadata = &ctx->metadata[i];

		key_len = *(uint64_t *)ctx->fptr;
		ctx->fptr += sizeof(uint64_t);

		if ((metadata->name = (char *)malloc(key_len + 1)) == NULL) {
			perror("error reading metadata name, OOM\n");
			return -1;
		}
		memset(metadata->name, 0, key_len + 1);

		memcpy(metadata->name, ctx->fptr, key_len);
		ctx->fptr += key_len;

		metadata->type = *(uint32_t *)ctx->fptr;
		ctx->fptr += sizeof(uint32_t);

		switch (metadata->type) {
		case GGUF_METADATA_VALUE_TYPE_UINT8:
			rc = gguf_read_metadata_type_uint8(ctx, metadata);
			break;

		case GGUF_METADATA_VALUE_TYPE_INT8:
			rc = gguf_read_metadata_type_int8(ctx, metadata);
			break;

		case GGUF_METADATA_VALUE_TYPE_UINT16:
			rc = gguf_read_metadata_type_uint16(ctx, metadata);
			break;

		case GGUF_METADATA_VALUE_TYPE_INT16:
			rc = gguf_read_metadata_type_int16(ctx, metadata);
			break;

		case GGUF_METADATA_VALUE_TYPE_UINT32:
			rc = gguf_read_metadata_type_uint32(ctx, metadata);
			break;

		case GGUF_METADATA_VALUE_TYPE_INT32:
			rc = gguf_read_metadata_type_int32(ctx, metadata);
			break;

		case GGUF_METADATA_VALUE_TYPE_FLOAT32:
			rc = gguf_read_metadata_type_fp32(ctx, metadata);
			break;

		case GGUF_METADATA_VALUE_TYPE_BOOL:
			rc = gguf_read_metadata_type_uint8(ctx, metadata);
			break;

		case GGUF_METADATA_VALUE_TYPE_STRING:
			rc = gguf_read_metadata_type_string(ctx, metadata);
			break;

		case GGUF_METADATA_VALUE_TYPE_ARRAY:
			rc = gguf_read_metadata_type_array(ctx, metadata);
			break;

		case GGUF_METADATA_VALUE_TYPE_UINT64:
			rc = gguf_read_metadata_type_uint64(ctx, metadata);
			break;

		case GGUF_METADATA_VALUE_TYPE_INT64:
			rc = gguf_read_metadata_type_int64(ctx, metadata);
			break;

		case GGUF_METADATA_VALUE_TYPE_FLOAT64:
			rc = gguf_read_metadata_type_fp64(ctx, metadata);
			break;
		default:
			rc = -1;
		}
	}

	if (rc == -1) {
		perror("error alloc metadata info, OOM\n");
		return -1;
	}

	gguf_dump_metadata(ctx);

	return 0;
}

#ifdef DEBUG_TENSORS
void dump_tensors(struct ctx_t *ctx)
{
	uint64_t i;

	for (i = 0; i < ctx->tensor_count; i++) {
		gguf_tensor *tensor = &ctx->tensors[i];

		printf("tensor #%4llu name: %32s, ", i, ctx->tensors[i].name);
		printf("type: %6s, ", tensor_get_type_name(tensor->type));
		printf("shape: [ %llu,\t%llu,\t%llu,\t%llu ],", tensor->dimensions[0], tensor->dimensions[1],
		       tensor->dimensions[2], tensor->dimensions[3]);
		printf("\tsize: %.02f MB, offset: %llu\n", (float)tensor->size / 1024 / 1024, tensor->offset);
//		printf("\tsize: %llu, offset: %llu\n", tensor->size, tensor->offset);
	}
}
#endif

void *get_tensor_data_ptr(struct ctx_t *ctx_ptr, const char *name, uint64_t *size_bytes)
{
	for (uint64_t i = 0; i < ctx_ptr->tensor_count; i++) {
		gguf_tensor *t = &ctx_ptr->tensors[i];
		if (strcmp(t->name, name) == 0) {
			*size_bytes = t->size;
			return t->data;
		}
	}
	fprintf(stderr, "Warning: Tensor '%s' not found in gguf file.\n", name);
	*size_bytes = 0;
	return NULL;
}

gguf_tensor *get_tensor(struct ctx_t *ctx_ptr, const char *name)
{
	for (uint64_t i = 0; i < ctx_ptr->tensor_count; i++) {
		gguf_tensor *t = &ctx_ptr->tensors[i];
		if (strcmp(t->name, name) == 0) {
			return t;
		}
	}

	fprintf(stderr, "Warning: Tensor '%s' not found in gguf file.\n", name);
	return NULL;
}

uint64_t calculate_tensor_size(uint32_t type, uint64_t *dims, uint32_t n_dims)
{
	uint64_t n_elements = 1;
	for (uint32_t i = 0; i < n_dims; i++) {
		n_elements *= dims[i];
	}
	if (n_elements == 0)
		return 0;

	switch (type) {
	case GGML_TYPE_F32:
		return n_elements * 4;
	case GGML_TYPE_F16:
		return n_elements * 2;
	case GGML_TYPE_BF16:
		return n_elements * 2;
	case GGML_TYPE_I8:
		return n_elements * 1;
	case GGML_TYPE_I16:
		return n_elements * 2;
	case GGML_TYPE_I32:
		return n_elements * 4;
	case GGML_TYPE_I64:
		return n_elements * 8;
	case GGML_TYPE_F64:
		return n_elements * 8;
	case GGML_TYPE_Q4_0: {
		const int QK = 32;
		const int block_size = 18; // 2 (delta) + 16 (quants)
		uint64_t blocks = (n_elements + QK - 1) / QK;
		return blocks * block_size;
	}
	case GGML_TYPE_Q4_1: {
		const int QK = 32;
		const int block_size = 20; // 2 (delta) + 2 (min) + 16 (quants)
		uint64_t blocks = (n_elements + QK - 1) / QK;
		return blocks * block_size;
	}
	case GGML_TYPE_Q5_0: {
		const int QK = 32;
		const int block_size = 22; // 2 (delta) + 4 (qh) + 16 (qs)
		uint64_t blocks = (n_elements + QK - 1) / QK;
		return blocks * block_size;
	}
	case GGML_TYPE_Q5_1: {
		const int QK = 32;
		const int block_size = 24; // 2 (delta) + 2 (min) + 4 (qh) + 16 (qs)
		uint64_t blocks = (n_elements + QK - 1) / QK;
		return blocks * block_size;
	}
	case GGML_TYPE_Q8_0: {
		const int QK = 32;
		const int block_size = 34; // 2 (delta) + 32 (quants)
		uint64_t blocks = (n_elements + QK - 1) / QK;
		return blocks * block_size;
	}
	case GGML_TYPE_Q8_1: {
		const int QK = 32;
		const int block_size = 40; // 4 (delta) + 4 (sum) + 32 (quants)
		uint64_t blocks = (n_elements + QK - 1) / QK;
		return blocks * block_size;
	}
	case GGML_TYPE_Q2_K: {
		const int QK_K = 256;
		const int block_size = 84; // 16 (scales) + 64 (quants) + 4 (d, dmin)
		uint64_t blocks = (n_elements + QK_K - 1) / QK_K;
		return blocks * block_size;
	}
	case GGML_TYPE_Q3_K: {
		const int QK_K = 256;
		const int block_size = 100; // 32 (hmask) + 64 (qs) + 2 (scales) + 2 (d)
		uint64_t blocks = (n_elements + QK_K - 1) / QK_K;
		return blocks * block_size;
	}
	case GGML_TYPE_Q4_K: {
		const int QK_K = 256;
		const int block_size = 148; // 2 (d) + 2 (m) + 128 (qs) + 16 (scales)
		uint64_t blocks = (n_elements + QK_K - 1) / QK_K;
		return blocks * block_size;
	}
	case GGML_TYPE_Q5_K: {
		const int QK_K = 256;
		const int block_size = 176; // validated from offsets
		uint64_t blocks = (n_elements + QK_K - 1) / QK_K;
		return blocks * block_size;
	}
	case GGML_TYPE_Q6_K: {
		const int QK_K = 256;
		const int block_size = sizeof(block_q6_k);
		uint64_t blocks = (n_elements + QK_K - 1) / QK_K;
		return blocks * block_size;
	}
	case GGML_TYPE_Q8_K: {
		const int QK_K = 256;
		const int block_size = 280; // 256 (qs) + 16 (scales) + 8 (d)
		uint64_t blocks = (n_elements + QK_K - 1) / QK_K;
		return blocks * block_size;
	}
	case GGML_TYPE_IQ2_XXS:
	case GGML_TYPE_IQ2_XS:
	case GGML_TYPE_IQ3_XXS:
	case GGML_TYPE_IQ1_S:
	case GGML_TYPE_IQ4_NL:
	case GGML_TYPE_IQ3_S:
	case GGML_TYPE_IQ2_S:
	case GGML_TYPE_IQ4_XS:
	case GGML_TYPE_IQ1_M: {
		fprintf(stderr, "Integer quantization type %u not fully supported\n", type);
		return 0;
	}
	default:
		fprintf(stderr, "Unsupported tensor type: %u\n", type);
		return 0;
	}
}

int gguf_map_weights(struct ctx_t *ctx)
{
	// Read tensor metadata
	ctx->tensors = (gguf_tensor *)calloc(ctx->tensor_count, sizeof(gguf_tensor));
	uint64_t data_offset = 0; // Will be set after alignment

	for (uint64_t i = 0; i < ctx->tensor_count; i++) {
		gguf_tensor *tensor = &ctx->tensors[i];

		uint64_t name_len = *(uint64_t *)ctx->fptr;
		ctx->fptr += sizeof(uint64_t);

		tensor->name = (char *)malloc(name_len + 1);
		memcpy(tensor->name, ctx->fptr, name_len);
		tensor->name[name_len] = '\0';
		ctx->fptr += name_len;

		// Read dimensions
		tensor->n_dims = *(uint32_t *)ctx->fptr;
		ctx->fptr += 4;

		if (tensor->n_dims > 4) {
			fprintf(stderr, "Invalid number of dimensions: %u\n", tensor->n_dims);
			return -1;
		}

		for (uint32_t j = 0; j < 4; j++)
			tensor->dimensions[j] = 1;

		for (uint32_t j = 0; j < tensor->n_dims; j++) {
			tensor->dimensions[j] = *(uint64_t *)ctx->fptr;
			ctx->fptr += 8;
		}

		// Read type
		tensor->type = *(uint32_t *)ctx->fptr;
		ctx->fptr += 4;

		tensor->offset = *(uint64_t *)ctx->fptr;
		ctx->fptr += sizeof(uint64_t);

		// Calculate size
		tensor->size = calculate_tensor_size(tensor->type, tensor->dimensions, tensor->n_dims);
		if (tensor->size == 0) {
			fprintf(stderr, "Failed to calculate size for tensor %s\n", tensor->name);
			return -1;
		}
	}

	// Align to 32-byte boundary
	ctx->fptr = (uint8_t *)(((uintptr_t)ctx->fptr + 31) & ~31);
	data_offset = ctx->fptr - (uint8_t *)ctx->mapped_data;
	ctx->tensor_data_offset = data_offset;

	// After reading all tensor metadata (name, n_dims, dimensions, type, offset)
	// and after calculating ctx->tensor_data_offset (the aligned start of the tensor data block)
	for (uint64_t i = 0; i < ctx->tensor_count; i++) {
		gguf_tensor *tensor = &ctx->tensors[i];
		// tensor->offset IS THE OFFSET FROM THE START OF THE TENSOR DATA BLOCK
		tensor->data = (uint8_t *)ctx->mapped_data + ctx->tensor_data_offset + tensor->offset;
	}

	// Check if the last tensor is within bounds
	if (ctx->tensor_count > 0) {
		gguf_tensor *last_tensor = &ctx->tensors[ctx->tensor_count - 1];

		if (ctx->tensor_data_offset + last_tensor->offset + last_tensor->size > ctx->file_size) {
			fprintf(stderr, "Error: End of last tensor exceeds file size.\n");
			return -1; // Or handle error appropriately
		} else {
			fprintf(stderr, "gguf: tensor pointers mapped and validated against file size.\n");
		}
	}

	return 0;
}

int gguf_read(struct ctx_t *ctx, char *path)
{
	struct stat st;
	uint32_t magic;
	uint32_t version;

	ctx->fd = open(path, O_RDONLY);

	if (ctx->fd == -1) {
		perror("error open model file\n");
		return -1;
	}

	if (fstat(ctx->fd, &st) == -1) {
		close(ctx->fd);
		return -1;
	}

	ctx->file_size = st.st_size;

	ctx->mapped_data = mmap(NULL, ctx->file_size, PROT_READ, MAP_PRIVATE, ctx->fd, 0);
	if (ctx->mapped_data == MAP_FAILED) {
		printf("Failed to mmap file\n");
		close(ctx->fd);
		return -1;
	}

	ctx->fptr = (uint8_t *)ctx->mapped_data;

	magic = *(uint32_t *)ctx->fptr;
	ctx->fptr += sizeof(uint32_t);
	if (magic != 0x46554747) {
		perror("invalid gguf file\n");
		return -1;
	}

	version = *(uint32_t *)ctx->fptr;
	ctx->fptr += sizeof(uint32_t);

	ctx->tensor_count = *(uint64_t *)ctx->fptr;
	ctx->fptr += sizeof(uint64_t);

	ctx->metadata_kv_count = *(uint64_t *)ctx->fptr;
	ctx->fptr += sizeof(uint64_t);

	gguf_read_metadata(ctx);

	if (gguf_map_weights(ctx) != 0) { /* ... error handling ... */
		return -1;
	}

	printf("gguf: Loaded %llu tensors (after map_weights).\n", ctx->tensor_count);

	return 0;
}

void gguf_close(struct ctx_t *ctx)
{
	gguf_free_metadata(ctx);

	munmap(ctx->mapped_data, ctx->file_size);
	close(ctx->fd);
}

static inline bool is_special_token_fast(const char *s, size_t len)
{
	return len >= 6 && s[0] == '<' && s[len - 1] == '>';
}

int gguf_metadata_read_tokens_embed(struct ctx_t *ctx, char *key)
{
	struct gguf_metadata_kv_t *metadata;
	char token[512];
	uint64_t i;
	uint64_t str_len;

	for (i = 0; i < ctx->metadata_kv_count; i++) {
		metadata = &ctx->metadata[i];

		if (!strcmp(ctx->metadata[i].name, key)) {
			goto _gguf_metadata_token_embed_found;
		}
	}

	printf("Failed to read %s\n", key);
	return -1;

_gguf_metadata_token_embed_found:

	if (metadata->type != GGUF_METADATA_VALUE_TYPE_ARRAY) {
		printf("%s has invalid type\n", key);
		return -1;
	}

	ctx->fptr = metadata->arr_offset;
	ctx->model->vocab_size = metadata->size;

	for (i = 0; i < metadata->size; i++) {
		str_len = *(uint64_t *)ctx->fptr;
		ctx->fptr += sizeof(uint64_t);

		memcpy(token, ctx->fptr, str_len);
		token[str_len] = '\0';

		// Append to string pool
		if (!append_to_pool(ctx->pool, token, str_len)) {
			printf("Failed to append token %llu to pool\n", i);
			return -1;
		}

		char *token_in_pool = ctx->pool->data + (ctx->pool->size - str_len - 1);
		insert_token(ctx->root, token_in_pool, str_len, (int)i);

		// Detect special tokens (cheap inline check)
		if (is_special_token_fast(token, str_len)) {
			if (special_tokens.count >= MAX_SPECIAL_TOKENS) {
				printf("Too many special tokens\n");
				return -1;
			}
			special_tokens.specials[special_tokens.count++] =
				(SpecialToken){.text = token_in_pool, .length = str_len, .token_id = (int)i};
		}

		ctx->fptr += str_len;
	}

	return 0;
}

int gguf_metadata_read_merges(struct ctx_t *ctx, char *key)
{
	struct gguf_metadata_kv_t *metadata = NULL;
	char merge_str[512];
	uint64_t i, str_len;

	// 1. Find metadata entry
	for (i = 0; i < ctx->metadata_kv_count; i++) {
		if (!strcmp(ctx->metadata[i].name, key)) {
			metadata = &ctx->metadata[i];
			break;
		}
	}

	if (!metadata) {
		printf("Failed to read %s\n", key);
		return -1;
	}

	if (metadata->type != GGUF_METADATA_VALUE_TYPE_ARRAY) {
		printf("%s has invalid type\n", key);
		return -1;
	}

	memset((void *)&bpe_merges_map, 0, sizeof(bpe_merge_map_t));

	ctx->fptr = metadata->arr_offset;
	ctx->model->merges_size = metadata->size;

	for (i = 0; i < metadata->size; i++) {
		str_len = *(uint64_t *)ctx->fptr;
		ctx->fptr += sizeof(uint64_t);

		if (str_len >= sizeof(merge_str)) {
			printf("Merge string too long\n");
			return -1;
		}

		memcpy(merge_str, ctx->fptr, str_len);
		merge_str[str_len] = '\0';
		ctx->fptr += str_len;

		// 2. Split into two tokens
		char *space = strchr(merge_str, ' ');
		if (!space) {
			printf("Malformed merge pair: %s\n", merge_str);
			return -1;
		}
		*space = '\0';
		const char *left = merge_str;
		const char *right = space + 1;

		// 3. Lookup token IDs
		int id1 = vocab_lookup_token_id(ctx->root, left, strlen(left));
		int id2 = vocab_lookup_token_id(ctx->root, right, strlen(right));

		// 4. Pack into key and insert
		uint64_t key = ((uint64_t)id1 << 32) | (uint32_t)id2;
		bpe_map_insert(&bpe_merges_map, key, (uint32_t)i);
	}

	return 0;
}
