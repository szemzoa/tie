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

#include "main.h"
#include "gguf.h"
#include "threadpool.h"
#include "tokenize.h"


size_t ggml_block_size(GGMLType type)
{
	switch (type) {
	case GGML_TYPE_Q4_K:
		return sizeof(block_q4_k);
	case GGML_TYPE_Q6_K:
		return sizeof(block_q6_k);
	case GGML_TYPE_Q8_0:
		return sizeof(block_q8_0);
	default:
		printf("FATAL: Unknown block size for type %d\n", type);
		return 0;
	}
}

size_t ggml_type_size(GGMLType type)
{
	switch (type) {
	case GGML_TYPE_F32:
		return sizeof(float);
	case GGML_TYPE_BF16:
		return sizeof(uint16_t);
	case GGML_TYPE_Q8_0:
		return sizeof(int8_t);
	default:
		printf("FATAL: Unknown size for type %d\n", type);
		return 0;
	}
}

int gguf_metadata_read_uint8(struct GGUFModel *gguf, GGUFMetadata *metadata)
{
	if ((metadata->data = (uint8_t *)malloc(sizeof(uint8_t))) == NULL) {
		return -1;
	}

	memcpy(metadata->data, gguf->fptr, sizeof(uint8_t));

	gguf->fptr += sizeof(uint8_t);
	metadata->size = sizeof(uint8_t);

	return 0;
}

int gguf_metadata_read_int8(struct GGUFModel *gguf, GGUFMetadata *metadata)
{
	if ((metadata->data = (int8_t *)malloc(sizeof(int8_t))) == NULL) {
		return -1;
	}

	memcpy(metadata->data, gguf->fptr, sizeof(int8_t));

	gguf->fptr += sizeof(int8_t);
	metadata->size = sizeof(int8_t);

	return 0;
}

int gguf_metadata_read_uint16(struct GGUFModel *gguf, GGUFMetadata *metadata)
{
	if ((metadata->data = (uint16_t *)malloc(sizeof(uint16_t))) == NULL) {
		return -1;
	}

	memcpy(metadata->data, gguf->fptr, sizeof(uint16_t));

	gguf->fptr += sizeof(uint16_t);
	metadata->size = sizeof(uint16_t);

	return 0;
}

int gguf_metadata_read_int16(struct GGUFModel *gguf, GGUFMetadata *metadata)
{
	if ((metadata->data = (int16_t *)malloc(sizeof(int16_t))) == NULL) {
		return -1;
	}

	memcpy(metadata->data, gguf->fptr, sizeof(int16_t));
	gguf->fptr += sizeof(int16_t);
	metadata->size = sizeof(int16_t);

	return 0;
}

int gguf_metadata_read_uint32(struct GGUFModel *gguf, GGUFMetadata *metadata)
{
	if ((metadata->data = (uint32_t *)malloc(sizeof(uint32_t))) == NULL) {
		return -1;
	}

	memcpy(metadata->data, gguf->fptr, sizeof(uint32_t));
	gguf->fptr += sizeof(uint32_t);
	metadata->size = sizeof(uint32_t);

	return 0;
}

int gguf_metadata_read_int32(struct GGUFModel *gguf, GGUFMetadata *metadata)
{
	if ((metadata->data = (int32_t *)malloc(sizeof(int32_t))) == NULL) {
		return -1;
	}

	memcpy(metadata->data, gguf->fptr, sizeof(int32_t));
	gguf->fptr += sizeof(int32_t);
	metadata->size = sizeof(int32_t);

	return 0;
}

int gguf_metadata_read_uint64(struct GGUFModel *gguf, GGUFMetadata *metadata)
{
	if ((metadata->data = (uint64_t *)malloc(sizeof(uint64_t))) == NULL) {
		return -1;
	}

	memcpy(metadata->data, gguf->fptr, sizeof(uint64_t));
	gguf->fptr += sizeof(uint64_t);
	metadata->size = sizeof(uint64_t);

	return 0;
}

int gguf_metadata_read_int64(struct GGUFModel *gguf, GGUFMetadata *metadata)
{
	if ((metadata->data = (int64_t *)malloc(sizeof(int64_t))) == NULL) {
		return -1;
	}

	memcpy(metadata->data, gguf->fptr, sizeof(int64_t));
	gguf->fptr += sizeof(int64_t);
	metadata->size = sizeof(int64_t);

	return 0;
}

int gguf_metadata_read_fp32(struct GGUFModel *gguf, GGUFMetadata *metadata)
{
	if ((metadata->data = (float *)malloc(sizeof(float))) == NULL) {
		return -1;
	}

	memcpy(metadata->data, gguf->fptr, sizeof(float));
	gguf->fptr += sizeof(float);
	metadata->size = sizeof(float);

	return 0;
}

int gguf_metadata_read_fp64(struct GGUFModel *gguf, GGUFMetadata *metadata)
{
	if ((metadata->data = (double *)malloc(sizeof(double))) == NULL) {
		return -1;
	}

	memcpy(metadata->data, gguf->fptr, sizeof(double));
	gguf->fptr += sizeof(double);
	metadata->size = sizeof(double);

	return 0;
}

int gguf_metadata_read_string(struct GGUFModel *gguf, GGUFMetadata *metadata)
{
	uint64_t size;
	char *str;

	size = *(uint64_t *)gguf->fptr;
	gguf->fptr += sizeof(uint64_t);

	if ((str = (char *)malloc(size + 1)) == NULL) {
		return -1;
	}

	memset(str, 0, size + 1);

	memcpy(str, gguf->fptr, size);
	gguf->fptr += size;
	metadata->size = size;

	metadata->data = (char *)str;
	return 0;
}

int gguf_metadata_read_array(struct GGUFModel *gguf, GGUFMetadata *metadata)
{
	uint32_t array_element_type;
	uint64_t array_element_count;
	uint64_t element_size_in_bytes = 0;
	uint64_t total_array_size_in_bytes = 0;

	// Read the array's element type (e.g., UINT32, FLOAT32, STRING)
	array_element_type = *(uint32_t *)gguf->fptr;
	gguf->fptr += sizeof(uint32_t);

	// Read the number of elements in the array
	array_element_count = *(uint64_t *)gguf->fptr;
	gguf->fptr += sizeof(uint64_t);

	// Store the element count
	metadata->size = array_element_count;
	// Store the file offset where the actual data begins
	metadata->arr_offset = gguf->fptr;

	// Special case: Array of strings
	if (array_element_type == GGUF_METADATA_VALUE_TYPE_STRING) {
		uint64_t i;
		uint64_t str_len;

		// Allocate memory for the array of pointers
		metadata->data = malloc(array_element_count * sizeof(char *));
		if (!metadata->data) {
			fprintf(stderr, "GGUF: Failed to allocate memory for string array pointers\n");
			return -1;
		}

		char **string_array = (char **)metadata->data;

		for (i = 0; i < array_element_count; i++) {
			// Read the length of the string
			str_len = *(uint64_t *)gguf->fptr;
			gguf->fptr += sizeof(uint64_t);

			// Allocate memory for the string data (+1 for null terminator)
			string_array[i] = malloc(str_len + 1);
			if (!string_array[i]) {
				fprintf(stderr, "GGUF: Failed to allocate memory for string data\n");
				return -1;
			}

			// Copy the string data
			memcpy(string_array[i], gguf->fptr, str_len);
			string_array[i][str_len] = '\0'; // Add null terminator

			// Advance pointer past the string data
			gguf->fptr += str_len;
		}
	} else {
		// Determine the size of a single element
		switch (array_element_type) {
		case GGUF_METADATA_VALUE_TYPE_UINT8:
			element_size_in_bytes = sizeof(uint8_t);
			break;
		case GGUF_METADATA_VALUE_TYPE_INT8:
			element_size_in_bytes = sizeof(int8_t);
			break;
		case GGUF_METADATA_VALUE_TYPE_UINT16:
			element_size_in_bytes = sizeof(uint16_t);
			break;
		case GGUF_METADATA_VALUE_TYPE_INT16:
			element_size_in_bytes = sizeof(int16_t);
			break;
		case GGUF_METADATA_VALUE_TYPE_UINT32:
			element_size_in_bytes = sizeof(uint32_t);
			break;
		case GGUF_METADATA_VALUE_TYPE_INT32:
			element_size_in_bytes = sizeof(int32_t);
			break;
		case GGUF_METADATA_VALUE_TYPE_FLOAT32:
			element_size_in_bytes = sizeof(float);
			break;
		case GGUF_METADATA_VALUE_TYPE_BOOL:
			element_size_in_bytes = sizeof(uint8_t); // GGUF bools are uint8_t
			break;
		case GGUF_METADATA_VALUE_TYPE_UINT64:
			element_size_in_bytes = sizeof(uint64_t);
			break;
		case GGUF_METADATA_VALUE_TYPE_INT64:
			element_size_in_bytes = sizeof(int64_t);
			break;
		case GGUF_METADATA_VALUE_TYPE_FLOAT64:
			element_size_in_bytes = sizeof(double);
			break;
		default:
			fprintf(stderr, "GGUF: Unknown or unsupported array element type: %u\n", array_element_type);
			return -1;
		}

		// Calculate total size
		total_array_size_in_bytes = element_size_in_bytes * array_element_count;

		// Allocate a single block for the entire array
		metadata->data = malloc(total_array_size_in_bytes);
		if (!metadata->data) {
			fprintf(stderr, "GGUF: Failed to allocate memory for metadata array\n");
			return -1;
		}

		// Copy the entire array data
		memcpy(metadata->data, gguf->fptr, total_array_size_in_bytes);

		// Advance the file pointer past the entire array
		gguf->fptr += total_array_size_in_bytes;
	}

	return 0;
}

int gguf_metadata_get_value(struct GGUFModel *gguf, char *key, void *value)
{
	GGUFMetadata *metadata;
	uint64_t i = 0;

	for (i = 0; i < gguf->metadata_num; i++) {
		metadata = &gguf->metadata[i];

		if (!strcmp(gguf->metadata[i].name, key)) {
			memcpy((void *)value, metadata->data, metadata->size);
			return 0;
		}
	}

	return -1;
}

int gguf_metadata_get_type(struct GGUFModel *gguf, char *key)
{
	GGUFMetadata *metadata;
	uint64_t i = 0;

	for (i = 0; i < gguf->metadata_num; i++) {
		metadata = &gguf->metadata[i];

		if (!strcmp(gguf->metadata[i].name, key)) {
			return gguf->metadata[i].type;
		}
	}

	return -1;
}

GGUFMetadata *gguf_metadata_get(struct GGUFModel *gguf, char *key)
{
	GGUFMetadata *metadata;
	uint64_t i = 0;

	for (i = 0; i < gguf->metadata_num; i++) {
		metadata = &gguf->metadata[i];

		if (!strcmp(gguf->metadata[i].name, key)) {
			return &gguf->metadata[i];
		}
	}

	return NULL;
}

char *gguf_metadata_get_string(struct GGUFModel *gguf, char *key)
{
	GGUFMetadata *metadata;
	uint64_t i = 0;

	for (i = 0; i < gguf->metadata_num; i++) {
		metadata = &gguf->metadata[i];

		if (!strcmp(gguf->metadata[i].name, key))
			return metadata->data;
	}

	return NULL;
}

int gguf_metadata_get_size(struct GGUFModel *gguf, char *key, uint64_t *size)
{
	GGUFMetadata *metadata;
	uint64_t i = 0;

	for (i = 0; i < gguf->metadata_num; i++) {
		metadata = &gguf->metadata[i];

		if (!strcmp(gguf->metadata[i].name, key)) {
			*size = metadata->size;
			return 0;
		}
	}

	return -1;
}

void *gguf_metadata_get_array_typed(struct GGUFModel *gguf, const char *key, uint32_t expected_type,
				    uint64_t *array_size)
{
	if (array_size)
		*array_size = 0;

	// Find the metadata
	GGUFMetadata *metadata = gguf_metadata_get(gguf, (char *)key);
	if (metadata == NULL)
		return NULL;

	if (metadata->data == NULL || metadata->size == 0) {
		fprintf(stderr, "GGUF: Metadata key '%s' has no data or is not an array.\n", key);
		return NULL;
	}

	if (metadata->type != expected_type) {
		fprintf(stderr, "GGUF: Type mismatch for key '%s'. Expected type %u, but file has %u.\n", key,
			expected_type, metadata->type);
		return NULL;
	}

	if (array_size)
		*array_size = metadata->size;

	return metadata->data;
}


char *gguf_get_type_name(uint32_t type)
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

#ifdef DEBUG_TENSORS
void gguf_model_tensors_dump(struct GGUFModel *gguf)
{
	uint64_t i;

	for (i = 0; i < gguf->tensor_count; i++) {
		GGUFTensor *tensor = &gguf->tensors[i];

		printf("tensor #%4llu name: %32s, ", i, gguf->tensors[i].name);
		printf("type: %6s, ", gguf_get_type_name(tensor->type));
		printf("shape: [ %llu,\t%llu,\t%llu,\t%llu ],", tensor->dimensions[0], tensor->dimensions[1],
		       tensor->dimensions[2], tensor->dimensions[3]);
		//		printf("\tsize: %.02f MB, offset: %llu\n", (float)tensor->size / 1024 / 1024,
		// tensor->offset);
		printf("\tsize: %llu, offset: %llu\n", tensor->size, tensor->offset);
	}
}
#endif

void *gguf_get_tensor_data_ptr(struct GGUFModel *gguf, const char *name, uint64_t *size_bytes)
{
	for (uint64_t i = 0; i < gguf->tensor_count; i++) {
		GGUFTensor *t = &gguf->tensors[i];
		if (strcmp(t->name, name) == 0) {
			*size_bytes = t->size;
			return t->data;
		}
	}

	//	fprintf(stderr, "Warning: Tensor '%s' not found in gguf file.\n", name);
	*size_bytes = 0;
	return NULL;
}

GGUFTensor *gguf_get_tensor(struct GGUFModel *gguf, const char *name)
{
	for (uint64_t i = 0; i < gguf->tensor_count; i++) {
		GGUFTensor *t = &gguf->tensors[i];
		if (strcmp(t->name, name) == 0) {
			return t;
		}
	}

	//	fprintf(stderr, "Warning: Tensor '%s' not found in gguf file.\n", name);
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
		const int block_size = 84; // 16 (scales) + 64 (quants) + 4 (d, dmin)
		uint64_t blocks = (n_elements + QK_K - 1) / QK_K;
		return blocks * block_size;
	}
	case GGML_TYPE_Q3_K: {
		const int block_size = 100; // 32 (hmask) + 64 (qs) + 2 (scales) + 2 (d)
		uint64_t blocks = (n_elements + QK_K - 1) / QK_K;
		return blocks * block_size;
	}
	case GGML_TYPE_Q4_K: {
		const int block_size = sizeof(block_q4_k);
		uint64_t blocks = (n_elements + QK_K - 1) / QK_K;
		return blocks * block_size;
	}
	case GGML_TYPE_Q5_K: {
		const int block_size = 176; // validated from offsets
		uint64_t blocks = (n_elements + QK_K - 1) / QK_K;
		return blocks * block_size;
	}
	case GGML_TYPE_Q6_K: {
		const int block_size = sizeof(block_q6_k);
		uint64_t blocks = (n_elements + QK_K - 1) / QK_K;
		return blocks * block_size;
	}
	case GGML_TYPE_Q8_K: {
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

void gguf_model_metadata_dump(struct GGUFModel *gguf)
{
	uint64_t i = 0;
	GGUFMetadata *metadata;
	struct gguf_mdata_array_t *array;

	printf("gguf: metadata(%llu):\n", gguf->metadata_num);
	for (i = 0; i < gguf->metadata_num; i++) {
		metadata = &gguf->metadata[i];

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

void gguf_model_metadata_free(struct GGUFModel *gguf)
{
	GGUFMetadata *metadata;
	uint64_t i;

	if (!gguf || !gguf->metadata)
		return;

	for (i = 0; i < gguf->metadata_num; i++) {
		metadata = &gguf->metadata[i];

//		printf("cleanup metadata: %s\n", metadata->name);

		// Free the metadata key name
		if (metadata->name) {
			free(metadata->name);
			metadata->name = NULL; // Good practice
		}

		if (metadata->type == GGUF_METADATA_VALUE_TYPE_ARRAY) {
		// Free the metadata value data
		if (metadata->data) {
			// Array of strings
			// We must free each individual string before freeing the array of pointers.
			// We identify this case if the type is STRING and the size > 0
			// (assuming single/non-array values have size 0)
			if (metadata->type == GGUF_METADATA_VALUE_TYPE_STRING && metadata->size > 0) {
				char **string_array = (char **)metadata->data;
				uint64_t j;
				for (j = 0; j < metadata->size; j++) {
					if (string_array[j]) {
						free(string_array[j]);
					}
				}
				// Now free the array of pointers itself
				free(metadata->data);
			} else {
				free(metadata->data);
			}

			metadata->data = NULL; // Good practice
		}
		}
	}

	// Finally, free the array of metadata structs
	free(gguf->metadata);
	gguf->metadata = NULL;
	gguf->metadata_num = 0;
}

int gguf_model_metadata_read(struct GGUFModel *gguf)
{
	uint64_t i;
	GGUFMetadata *metadata;
	uint64_t key_len;
	int rc;

	if ((gguf->metadata = (GGUFMetadata *)calloc(gguf->metadata_num, sizeof(GGUFMetadata))) == NULL) {
		perror("Failed to alloc metadata info, OOM\n");
		return -1;
	}

	for (i = 0; i < gguf->metadata_num; i++) {
		metadata = &gguf->metadata[i];

		key_len = *(uint64_t *)gguf->fptr;
		gguf->fptr += sizeof(uint64_t);

		if ((metadata->name = (char *)malloc(key_len + 1)) == NULL) {
			printf("Falied to read metadata name, OOM, key_len: %llu\n", key_len);
			return -1;
		}
		memset(metadata->name, 0, key_len + 1);

		memcpy(metadata->name, gguf->fptr, key_len);
		gguf->fptr += key_len;

		metadata->type = *(uint32_t *)gguf->fptr;
		gguf->fptr += sizeof(uint32_t);

		switch (metadata->type) {
		case GGUF_METADATA_VALUE_TYPE_UINT8:
			rc = gguf_metadata_read_uint8(gguf, metadata);
			break;

		case GGUF_METADATA_VALUE_TYPE_INT8:
			rc = gguf_metadata_read_int8(gguf, metadata);
			break;

		case GGUF_METADATA_VALUE_TYPE_UINT16:
			rc = gguf_metadata_read_uint16(gguf, metadata);
			break;

		case GGUF_METADATA_VALUE_TYPE_INT16:
			rc = gguf_metadata_read_int16(gguf, metadata);
			break;

		case GGUF_METADATA_VALUE_TYPE_UINT32:
			rc = gguf_metadata_read_uint32(gguf, metadata);
			break;

		case GGUF_METADATA_VALUE_TYPE_INT32:
			rc = gguf_metadata_read_int32(gguf, metadata);
			break;

		case GGUF_METADATA_VALUE_TYPE_FLOAT32:
			rc = gguf_metadata_read_fp32(gguf, metadata);
			break;

		case GGUF_METADATA_VALUE_TYPE_BOOL:
			rc = gguf_metadata_read_uint8(gguf, metadata);
			break;

		case GGUF_METADATA_VALUE_TYPE_STRING:
			rc = gguf_metadata_read_string(gguf, metadata);
			break;

		case GGUF_METADATA_VALUE_TYPE_ARRAY:
			rc = gguf_metadata_read_array(gguf, metadata);
			break;

		case GGUF_METADATA_VALUE_TYPE_UINT64:
			rc = gguf_metadata_read_uint64(gguf, metadata);
			break;

		case GGUF_METADATA_VALUE_TYPE_INT64:
			rc = gguf_metadata_read_int64(gguf, metadata);
			break;

		case GGUF_METADATA_VALUE_TYPE_FLOAT64:
			rc = gguf_metadata_read_fp64(gguf, metadata);
			break;
		default:
			rc = -1;
		}
	}

	if (rc == -1) {
		perror("Failed to alloc metadata info, OOM\n");
		return -1;
	}

	gguf_model_metadata_dump(gguf);

	return 0;
}

int gguf_model_map_weights(struct GGUFModel *gguf)
{
	// Read tensor metadata
	gguf->tensors = (GGUFTensor *)calloc(gguf->tensor_count, sizeof(GGUFTensor));
	uint64_t data_offset = 0; // Will be set after alignment

	for (uint64_t i = 0; i < gguf->tensor_count; i++) {
		GGUFTensor *tensor = &gguf->tensors[i];

		uint64_t name_len = *(uint64_t *)gguf->fptr;
		gguf->fptr += sizeof(uint64_t);

		tensor->name = (char *)malloc(name_len + 1);
		memcpy(tensor->name, gguf->fptr, name_len);
		tensor->name[name_len] = '\0';
		gguf->fptr += name_len;

		// Read dimensions
		tensor->n_dims = *(uint32_t *)gguf->fptr;
		gguf->fptr += 4;

		if (tensor->n_dims > 4) {
			fprintf(stderr, "Invalid number of dimensions: %u\n", tensor->n_dims);
			return -1;
		}

		for (uint32_t j = 0; j < 4; j++)
			tensor->dimensions[j] = 1;

		for (uint32_t j = 0; j < tensor->n_dims; j++) {
			tensor->dimensions[j] = *(uint64_t *)gguf->fptr;
			gguf->fptr += 8;
		}

		// Read type
		tensor->type = *(uint32_t *)gguf->fptr;
		gguf->fptr += 4;

		tensor->offset = *(uint64_t *)gguf->fptr;
		gguf->fptr += sizeof(uint64_t);

		// Calculate size
		tensor->size = calculate_tensor_size(tensor->type, tensor->dimensions, tensor->n_dims);
		if (tensor->size == 0) {
			fprintf(stderr, "Failed to calculate size for tensor %s\n", tensor->name);
			return -1;
		}
	}

	// Align to 32-byte boundary
	gguf->fptr = (uint8_t *)(((uintptr_t)gguf->fptr + 31) & ~31);
	data_offset = gguf->fptr - (uint8_t *)gguf->mapped_data;
	gguf->tensor_data_offset = data_offset;

	// After reading all tensor metadata (name, n_dims, dimensions, type, offset)
	// and after calculating gguf->tensor_data_offset (the aligned start of the tensor data block)
	for (uint64_t i = 0; i < gguf->tensor_count; i++) {
		GGUFTensor *tensor = &gguf->tensors[i];
		// tensor->offset IS THE OFFSET FROM THE START OF THE TENSOR DATA BLOCK
		tensor->data = (uint8_t *)gguf->mapped_data + gguf->tensor_data_offset + tensor->offset;
	}

	// Check if the last tensor is within bounds
	if (gguf->tensor_count > 0) {
		GGUFTensor *last_tensor = &gguf->tensors[gguf->tensor_count - 1];

		if (gguf->tensor_data_offset + last_tensor->offset + last_tensor->size > gguf->file_size) {
			fprintf(stderr, "Error: End of last tensor exceeds file size.\n");
			return -1; // Or handle error appropriately
		} else {
			fprintf(stderr, "gguf: tensor pointers mapped and validated against file size.\n");
		}
	}

	return 0;
}

struct GGUFModel *gguf_model_parse(char *path)
{
	struct GGUFModel *gguf;
	struct stat st;
	uint32_t magic;
	uint32_t version;


	if ((gguf = malloc(sizeof(struct GGUFModel))) == NULL) {
		perror("Failed to allocate gguf");
		return NULL;
	}

	gguf->fd = open(path, O_RDONLY);

	if (gguf->fd == -1) {
		printf("Failed to open gguf file\n");
		free(gguf);
		return NULL;
	}

	if (fstat(gguf->fd, &st) == -1)
		goto _gguf_model_open_error;

	gguf->file_size = st.st_size;

	gguf->mapped_data = mmap(NULL, gguf->file_size, PROT_READ, MAP_PRIVATE, gguf->fd, 0);
	if (gguf->mapped_data == MAP_FAILED) {
		printf("Failed to mmap gguf file\n");
		goto _gguf_model_open_error;
	}

	gguf->fptr = (uint8_t *)gguf->mapped_data;

	magic = *(uint32_t *)gguf->fptr;
	gguf->fptr += sizeof(uint32_t);
	if (magic != 0x46554747) {
		perror("Unsupported GGUF file\n");
		goto _gguf_model_open_error_unmap;
	}

	version = *(uint32_t *)gguf->fptr;
	gguf->fptr += sizeof(uint32_t);

	gguf->tensor_count = *(uint64_t *)gguf->fptr;
	gguf->fptr += sizeof(uint64_t);

	gguf->metadata_num = *(uint64_t *)gguf->fptr;
	gguf->fptr += sizeof(uint64_t);

	if (gguf_model_metadata_read(gguf) != 0)
		goto _gguf_model_open_error_free;

	if (gguf_model_map_weights(gguf) != 0)
		goto _gguf_model_open_error_free;

	if ((gguf->arch = detect_architecture(gguf_metadata_get_string(gguf, "general.architecture"))) == -1)
		goto _gguf_model_open_error_free;

#ifdef DEBUG_TENSORS
	gguf_model_tensors_dump(gguf);
#endif

	return gguf;

_gguf_model_open_error_free:
	gguf_model_metadata_free(gguf);

_gguf_model_open_error_unmap:
	munmap(gguf->mapped_data, gguf->file_size);

_gguf_model_open_error:
	close(gguf->fd);

	return NULL;
}

void gguf_model_close(struct GGUFModel *gguf)
{
	gguf_model_metadata_free(gguf);

	munmap(gguf->mapped_data, gguf->file_size);

	close(gguf->fd);
}

static inline bool is_special_token_fast(const char *s, size_t len)
{
	return len >= 6 && s[0] == '<' && s[len - 1] == '>';
}

int gguf_model_read_token_embeds(struct TIEContext *ctx, struct GGUFModel *gguf, char *key, int detect_special)
{
	GGUFMetadata *metadata = NULL;
	uint64_t i, len;

	for (i = 0; i < gguf->metadata_num; i++) {
		if (!strcmp(gguf->metadata[i].name, key)) {
			metadata = &gguf->metadata[i];
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

	gguf->fptr = metadata->arr_offset;
	ctx->model->vocab_size = metadata->size;

	if (init_token_table(ctx, ctx->model->vocab_size) != 0) {
		printf("Failed to init token table\n");
		return -1;
	}

	for (i = 0; i < metadata->size; i++) {
		memcpy(&len, gguf->fptr, sizeof(uint64_t));

		gguf->fptr += sizeof(uint64_t);

		const unsigned char *raw_token = (const unsigned char *)gguf->fptr;

		unsigned char *token_ptr = ctx->tokenizer.pool->data + ctx->tokenizer.pool->size;
		if (!append_to_pool(ctx->tokenizer.pool, raw_token, len)) {
			printf("Failed to append token %llu to pool\n", i);
			return -1;
		}

		ctx->tokenizer.token_table[i] = token_ptr;
		ctx->tokenizer.token_lens[i] = len;

		insert_token(ctx->tokenizer.root, token_ptr, len, (int)i);

		// Detect special tokens
		if (detect_special == 1) {
			if (is_special_token_fast(token_ptr, len)) {
				if (special_tokens.count >= MAX_SPECIAL_TOKENS) {
					printf("Too many special tokens\n");
					return -1;
				}
				special_tokens.specials[special_tokens.count++] =
					(SpecialToken){.text = token_ptr, .length = len, .token_id = (int)i};

				/*				char buf[256];
								memset(buf, 0, sizeof(buf));
								memcpy(buf, token_ptr, len);
								printf("special_token#%llu [%s]\n", i, token_ptr);
				*/
			}
		}

		gguf->fptr += len;
	}

	return 0;
}

int gguf_model_read_token_types(struct TIEContext *ctx, struct GGUFModel *gguf, char *key, int detect_special)
{
	GGUFMetadata *metadata = NULL;
	uint64_t i;
	uint32_t type;

	for (i = 0; i < gguf->metadata_num; i++) {
		if (!strcmp(gguf->metadata[i].name, key)) {
			metadata = &gguf->metadata[i];
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

	gguf->fptr = metadata->arr_offset;

	for (i = 0; i < metadata->size; i++) {
		memcpy(&type, gguf->fptr, sizeof(int32_t));
		gguf->fptr += sizeof(int32_t);

		ctx->tokenizer.token_types[i] = type;

		if (detect_special == 1)
			continue;

		if (type == GGUF_TOKEN_TYPE_CONTROL) {

			special_tokens.specials[special_tokens.count++] =
				(SpecialToken){.text = ctx->tokenizer.token_table[i],
					       .length = ctx->tokenizer.token_lens[i],
					       .token_id = (int)i};
		}
	}

	return 0;
}

int gguf_model_read_token_scores(struct TIEContext *ctx, struct GGUFModel *gguf, char *key)
{
	GGUFMetadata *metadata = NULL;
	uint64_t i;
	float score;

	for (i = 0; i < gguf->metadata_num; i++) {
		if (!strcmp(gguf->metadata[i].name, key)) {
			metadata = &gguf->metadata[i];
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

	gguf->fptr = metadata->arr_offset;

	for (i = 0; i < metadata->size; i++) {
		memcpy(&score, gguf->fptr, sizeof(float));
		gguf->fptr += sizeof(float);

		ctx->tokenizer.token_scores[i] = score;
	}

	return 0;
}

int gguf_model_read_token_merges(struct TIEContext *ctx, struct GGUFModel *gguf, char *key)
{
	GGUFMetadata *metadata = NULL;
	char merge_str[512];
	uint64_t i, str_len;

	// 1. Find metadata entry
	for (i = 0; i < gguf->metadata_num; i++) {
		if (!strcmp(gguf->metadata[i].name, key)) {
			metadata = &gguf->metadata[i];
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

	memset((void *)&bpe_merges_map, 0, sizeof(BpeMergeMap));

	gguf->fptr = metadata->arr_offset;
	ctx->model->merges_size = metadata->size;
	printf("model merges_size: %llu\n", ctx->model->merges_size);

	for (i = 0; i < metadata->size; i++) {
		str_len = *(uint64_t *)gguf->fptr;
		gguf->fptr += sizeof(uint64_t);

		if (str_len >= sizeof(merge_str)) {
			printf("Merge string too long\n");
			return -1;
		}

		memcpy(merge_str, gguf->fptr, str_len);
		merge_str[str_len] = '\0';
		gguf->fptr += str_len;

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
		int id1 = vocab_lookup_token_id(ctx->tokenizer.root, left, strlen(left));
		int id2 = vocab_lookup_token_id(ctx->tokenizer.root, right, strlen(right));

		// 4. Pack into key and insert
		uint64_t key = ((uint64_t)id1 << 32) | (uint32_t)id2;
		bpe_map_insert(&bpe_merges_map, key, (uint32_t)i);
	}

	return 0;
}

size_t gguf_get_type_size(uint32_t type)
{
    switch (type) {
        case GGUF_METADATA_VALUE_TYPE_UINT8:   return sizeof(uint8_t);
        case GGUF_METADATA_VALUE_TYPE_INT8:    return sizeof(int8_t);
        case GGUF_METADATA_VALUE_TYPE_BOOL:    return sizeof(uint8_t); // Bools are uint8_t
        case GGUF_METADATA_VALUE_TYPE_UINT16:  return sizeof(uint16_t);
        case GGUF_METADATA_VALUE_TYPE_INT16:   return sizeof(int16_t);
        case GGUF_METADATA_VALUE_TYPE_UINT32:  return sizeof(uint32_t);
        case GGUF_METADATA_VALUE_TYPE_INT32:   return sizeof(int32_t);
        case GGUF_METADATA_VALUE_TYPE_FLOAT32: return sizeof(float);
        case GGUF_METADATA_VALUE_TYPE_UINT64:  return sizeof(uint64_t);
        case GGUF_METADATA_VALUE_TYPE_INT64:   return sizeof(int64_t);
        case GGUF_METADATA_VALUE_TYPE_FLOAT64: return sizeof(double);
        default: return 0;
    }
}