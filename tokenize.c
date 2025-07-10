#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <stdbool.h>

#include "tokenize.h"
#include "main.h"

char merge_pool[MAX_SPANS][128];
bpe_merge_map_t bpe_merges_map;
SpecialTokenList special_tokens;

TrieNode *create_node()
{
	TrieNode *node = (TrieNode *)calloc(1, sizeof(TrieNode));
	if (node)
		node->token_id = -1;
	return node;
}

StringPool *create_string_pool(size_t initial_capacity)
{
	StringPool *pool = (StringPool *)malloc(sizeof(StringPool));
	if (!pool)
		return NULL;
	pool->data = (char *)malloc(initial_capacity);
	if (!pool->data) {
		free(pool);
		return NULL;
	}
	pool->size = 0;
	pool->capacity = initial_capacity;
	return pool;
}

int append_to_pool(StringPool *pool, const char *str, size_t len)
{
	if (pool->size + len + 1 > pool->capacity) {
		size_t new_capacity =
			pool->capacity * 2 > pool->size + len + 1 ? pool->capacity * 2 : pool->size + len + 1;
		char *new_data = (char *)realloc(pool->data, new_capacity);
		if (!new_data)
			return 0;
		pool->data = new_data;
		pool->capacity = new_capacity;
	}
	memcpy(pool->data + pool->size, str, len);
	pool->size += len;
	pool->data[pool->size] = '\0';
	pool->size++;
	return 1;
}

const char *get_token_string(const StringPool *pool, int token_id)
{
	const char *ptr = pool->data;
	int count = 0;
	if (token_id < 0)
		return NULL;
	while (count < token_id) {
		while (*ptr != '\0' && ptr < pool->data + pool->size)
			ptr++;
		if (ptr >= pool->data + pool->size)
			return NULL;
		ptr++;
		count++;
	}
	if (ptr >= pool->data + pool->size)
		return NULL;
	return ptr;
}

void insert_token(TrieNode *root, const char *token, size_t len, int token_id)
{
	TrieNode *node = root;
	for (size_t i = 0; i < len; i++) {
		unsigned char byte = (unsigned char)token[i];
		if (!node->children[byte])
			node->children[byte] = create_node();
		node = node->children[byte];
	}
	node->token_id = token_id;
}

void free_trie(TrieNode *node)
{
	if (!node)
		return;
	for (int i = 0; i < 256; i++) {
		if (node->children[i])
			free_trie(node->children[i]);
	}
	free(node);
}

void free_string_pool(StringPool *pool)
{
	if (pool) {
		free(pool->data);
		free(pool);
	}
}

void replace_g_spaces(char *s)
{
	for (char *p = s; *p;) {
		if ((unsigned char)p[0] == 0xC4 && (unsigned char)p[1] == 0xA0) { // Ġ
			*p = ' ';
			memmove(p + 1, p + 2, strlen(p + 2) + 1);
		} else if ((unsigned char)p[0] == 0xC4 && (unsigned char)p[1] == 0x8A) { // Ċ
			*p = '\n';
			memmove(p + 1, p + 2, strlen(p + 2) + 1);
		} else {
			p++;
		}
	}
}

int get_token_string_length(StringPool *pool, int token_id)
{
	const char *start = get_token_string(pool, token_id);
	if (!start)
		return 0;
	size_t len = 0;
	while ((unsigned char)start[len] >= 0x20 || start[len] == 0x09 || start[len] == 0x0A) {
		len++;
		if ((size_t)start + len >= (size_t)pool->data + pool->size)
			break;
		if (start[len] == '\0')
			break;
	}
	return (int)len;
}

int vocab_lookup_token_id(TrieNode *root, const char *token, size_t len)
{
	TrieNode *node = root;
	for (size_t i = 0; i < len; i++) {
		unsigned char byte = (unsigned char)token[i];
		node = node->children[byte];
		if (!node)
			return -1;
	}
	return node->token_id;
}

static inline uint64_t hash64(uint64_t x)
{
	x ^= x >> 30;
	x *= 0xbf58476d1ce4e5b9;
	x ^= x >> 27;
	x *= 0x94d049bb133111eb;
	x ^= x >> 31;
	return x;
}

void bpe_map_insert(bpe_merge_map_t *map, uint64_t key, uint32_t rank)
{
	uint64_t h = hash64(key);
	size_t idx = h & (BPE_MAP_CAPACITY - 1);
	while (map->table[idx].occupied) {
		if (map->table[idx].key == key) {
			map->table[idx].rank = rank;
			return;
		}
		idx = (idx + 1) & (BPE_MAP_CAPACITY - 1);
	}
	map->table[idx].key = key;
	map->table[idx].rank = rank;
	map->table[idx].occupied = true;
}

bool bpe_map_lookup(const bpe_merge_map_t *map, uint64_t key, uint32_t *out_rank)
{
	uint64_t h = hash64(key);
	size_t idx = h & (BPE_MAP_CAPACITY - 1);
	while (map->table[idx].occupied) {
		if (map->table[idx].key == key) {
			*out_rank = map->table[idx].rank;
			return true;
		}
		idx = (idx + 1) & (BPE_MAP_CAPACITY - 1);
	}
	return false;
}

int tokenize_step(TrieNode *root, const char *input, size_t len, size_t *pos, int *token_id)
{
	TrieNode *node = root;
	int last_token_id = -1;
	size_t last_match_len = 0;
	size_t match_len = 0;

	// Ensure we do not go past the input buffer
	while ((*pos + match_len) < len) {
		unsigned char byte = (unsigned char)input[*pos + match_len];

		if (!node->children[byte])
			break;

		node = node->children[byte];
		match_len++;

		if (node->token_id != -1) {
			last_token_id = node->token_id;
			last_match_len = match_len;
		}
	}

	if (last_token_id != -1) {
		*token_id = last_token_id;
		*pos += last_match_len;
		return 1;
	}

	return 0;
}

char *preprocess_input(const char *text, size_t len, size_t *out_len)
{
	char *out = malloc(len * 3 + 1); // Expanded buffer for worst-case UTF-8
	if (!out) {
		*out_len = 0;
		return NULL;
	}

	size_t j = 0;
	for (size_t i = 0; i < len; i++) {
		if (text[i] == ' ') {
			out[j++] = (char)0xC4; // 'Ġ'
			out[j++] = (char)0xA0;
		} else if (text[i] == '\n') {
			out[j++] = (char)0xC4; // 'Ċ'
			out[j++] = (char)0x8A;
		} else {
			out[j++] = text[i];
		}
	}
	out[j] = '\0'; // Null-terminate the output string
	*out_len = j;
	return out;
}

int *tokenize_bpe(struct ctx_t *ctx, const char *text, size_t *num_tokens)
{
	TokenChunk chunks[MAX_CHUNKS];
	int chunk_count = 0;
	size_t text_len = strlen(text);
	size_t p = 0;
	while (p < text_len) {
		bool matched = false;
		for (int s = 0; s < special_tokens.count; s++) {
			SpecialToken *sp = &special_tokens.specials[s];
			if (p + sp->length <= text_len && memcmp(&text[p], sp->text, sp->length) == 0) {
				chunks[chunk_count++] = (TokenChunk){.ptr = &text[p],
								     .len = sp->length,
								     .is_special = true,
								     .token_id = sp->token_id};
				p += sp->length;
				matched = true;
				break;
			}
		}

		if (!matched) {
			// Group consecutive non-special bytes into one chunk
			size_t start = p;
			while (p < text_len) {
				bool is_special = false;
				for (int s = 0; s < special_tokens.count; s++) {
					SpecialToken *sp = &special_tokens.specials[s];
					if (p + sp->length <= text_len && memcmp(&text[p], sp->text, sp->length) == 0) {
						is_special = true;
						break;
					}
				}
				if (is_special)
					break;
				p++;
			}
			size_t non_special_len = p - start;
			chunks[chunk_count++] =
				(TokenChunk){.ptr = &text[start], .len = non_special_len, .is_special = false};
		}
	}

	BpeTokenSpan spans[MAX_SPANS];
	int span_count = 0;

	for (int i = 0; i < chunk_count; i++) {
		TokenChunk *c = &chunks[i];

		if (c->is_special) {
			spans[span_count++] = (BpeTokenSpan){
				.token_id = c->token_id, .start = c->ptr, .length = c->len, .is_special = true};
			continue;
		}

		size_t pre_len;
		char *processed = preprocess_input(c->ptr, c->len, &pre_len);
		if (!processed)
			return NULL;

		size_t p = 0;
		while (p < pre_len) {
			int token_id;
			size_t prev_p = p;
			if (tokenize_step(ctx->root, processed, pre_len, &p, &token_id)) {
				spans[span_count++] = (BpeTokenSpan){.token_id = token_id,
								     .start = &processed[prev_p],
								     .length = (int)(p - prev_p),
								     .is_special = false};
			} else {
				spans[span_count++] = (BpeTokenSpan){.token_id = (unsigned char)processed[p],
								     .start = &processed[p],
								     .length = 1,
								     .is_special = false};
				p++;
			}
		}
		free(processed);
	}

	for (int i = 0; i < span_count; i++) {
		char buf[64];
		memset((void *)&buf, 0, 64);
		memcpy((void *)&buf, spans[i].start, spans[i].length);
	}

	while (1) {
		int best_rank = INT32_MAX, best_idx = -1;

		for (int i = 0; i < span_count - 1; i++) {
			if (spans[i].is_special || spans[i + 1].is_special)
				continue;

			uint64_t key = ((uint64_t)spans[i].token_id << 32) | spans[i + 1].token_id;
			uint32_t rank;
			if (bpe_map_lookup(&bpe_merges_map, key, &rank)) {
				if (rank < best_rank) {
					best_rank = rank;
					best_idx = i;
				}
			}
		}

		if (best_idx == -1)
			break;

		int len1 = spans[best_idx].length;
		int len2 = spans[best_idx + 1].length;

		if (len1 + len2 > 127)
			continue;
		char *buf = merge_pool[best_idx];

		memcpy(buf, spans[best_idx].start, len1);
		memcpy(buf + len1, spans[best_idx + 1].start, len2);
		int merged_len = len1 + len2;

		int merged_id = vocab_lookup_token_id(ctx->root, buf, merged_len);
		if (merged_id < 0) {
			fprintf(stderr, "ERROR: vocab lookup failed for merge '%.*s'\n", merged_len, buf);
			break;
		}

		spans[best_idx].token_id = merged_id;
		spans[best_idx].start = buf;
		spans[best_idx].length = merged_len;
		spans[best_idx].is_special = false;

		for (int i = best_idx + 1; i < span_count - 1; i++) {
			spans[i] = spans[i + 1];
		}
		span_count--;
	}

	int *output_ids = malloc(span_count * sizeof(int));
	if (!output_ids)
		return NULL;
	for (int i = 0; i < span_count; i++) {
		output_ids[i] = spans[i].token_id;
	}
	*num_tokens = span_count;
	return output_ids;
}
