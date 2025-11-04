#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <ctype.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#include "tokenize.h"
#include "main.h"

char merge_pool[MAX_SPANS][128];
BpeMergeMap bpe_merges_map;
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
	return 1;
}

const unsigned char *get_token_string(const struct TIEContext *ctx, int token_id)
{
	if (token_id < 0 || token_id >= ctx->tokenizer.token_count)
		return NULL;

	return ctx->tokenizer.token_table[token_id];
}

int get_token_string_length(const struct TIEContext *ctx, int token_id)
{
	if (token_id < 0 || token_id >= ctx->tokenizer.token_count)
		return 0;

	return ctx->tokenizer.token_lens[token_id];
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

void bpe_map_insert(BpeMergeMap *map, uint64_t key, uint32_t rank)
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

bool bpe_map_lookup(const BpeMergeMap *map, uint64_t key, uint32_t *out_rank)
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

int *tokenize_bpe(struct TIEContext *ctx, const char *text, size_t *num_tokens)
{
	TokenChunk chunks[MAX_CHUNKS];
	BpeTokenSpan spans[MAX_SPANS];
	int chunk_count = 0;
	int span_count = 0;
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
			if (tokenize_step(ctx->tokenizer.root, processed, pre_len, &p, &token_id)) {
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

		int merged_id = vocab_lookup_token_id(ctx->tokenizer.root, buf, merged_len);
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

char *sp_preprocess_input(const char *text, size_t *out_len)
{
	size_t text_len = strlen(text);
	size_t space_count = 0;
	for (size_t i = 0; i < text_len; ++i) {
		if (text[i] == ' ') {
			space_count++;
		}
	}

	// Each space ' ' (1 byte) becomes ' ' (3 bytes), so we need 2 extra bytes per space.
	size_t new_len = text_len + space_count * 2;
	char *out = malloc(new_len + 1);
	if (!out) {
		*out_len = 0;
		return NULL;
	}

	size_t j = 0;
	for (size_t i = 0; i < text_len; i++) {
		if (text[i] == ' ') {
			out[j++] = (char)0xE2;
			out[j++] = (char)0x96;
			out[j++] = (char)0x81;
		} else {
			out[j++] = text[i];
		}
	}
	out[j] = '\0';
	*out_len = j;
	return out;
}

int *tokenize_sp(struct TIEContext *ctx, const char *text, size_t *num_tokens)
{
	size_t processed_text_len;
	DP_Entry *dp;

	char *processed_text = sp_preprocess_input(text, &processed_text_len);
	if (!processed_text) {
		*num_tokens = 0;
		return NULL;
	}

	// Init Dynamic Programming (DP) Table
	if ((dp = (DP_Entry *)malloc((processed_text_len + 1) * sizeof(DP_Entry))) == NULL) {
		fprintf(stderr, "ERROR: %s OOM\n", __FUNCTION__);
		free(processed_text);
		return NULL;
	}

	dp[0].score = 0.0f;
	for (size_t i = 1; i <= processed_text_len; i++) {
		dp[i].score = -INFINITY;
		dp[i].token_id = -1;
		dp[i].backpointer = -1;
	}

	// FORWARD PASS
	for (int i = 0; i < processed_text_len; ++i) {
		// If we can't reach this position, we can't start a token from here
		if (dp[i].score == -INFINITY) {
			continue;
		}

		// Trie Traversal
		TrieNode *node = ctx->tokenizer.root;

		// Iterate through characters starting from position i
		for (int j = i; j < processed_text_len; ++j) {
			unsigned char byte = (unsigned char)processed_text[j];

			// Move to the next node in the Trie
			if (!node->children[byte]) {
				break; // No more possible tokens from this path
			}
			node = node->children[byte];

			// Check if the current node represents a valid token
			if (node->token_id != -1) {
				// We found a valid token. Let's get its properties.
				int matched_token_id = node->token_id;
				int matched_token_len = j - i + 1;
				int end_pos = i + matched_token_len;

				float candidate_score = dp[i].score + ctx->tokenizer.token_scores[matched_token_id];

				// If this path is better than any previous path to end_pos...
				if (candidate_score > dp[end_pos].score) {
					// ...update the DP table with this new best path!
					dp[end_pos].score = candidate_score;
					dp[end_pos].token_id = matched_token_id;
					dp[end_pos].backpointer = i;
				}
			}
		}
	}

	// Backward Pass - Reconstruct the best path
	// Check if a valid path was found to the end of the string
	if (dp[processed_text_len].score == -INFINITY) {
		fprintf(stderr, "ERROR: Could not tokenize the entire string.\n");
		goto _tokenize_sp_error;
	}

	int temp_tokens[MAX_PROMPT_BATCH_SIZE];
	int count = 0;
	int current_pos = processed_text_len;

	while (current_pos > 0) {
		if (count >= MAX_PROMPT_BATCH_SIZE) {
			fprintf(stderr, "ERROR: Exceeded MAX_TOKENS during tokenization.\n");
			goto _tokenize_sp_error;
		}
		temp_tokens[count++] = dp[current_pos].token_id;
		current_pos = dp[current_pos].backpointer;
	}

	int *output_ids = malloc(count * sizeof(int));
	if (!output_ids) {
		fprintf(stderr, "ERROR: %s OOM\n", __FUNCTION__);
		goto _tokenize_sp_error;
	}

	// The tokens are in reverse order, so now copy them back in the correct order
	for (int i = 0; i < count; ++i) {
		output_ids[i] = temp_tokens[count - 1 - i];
	}

	*num_tokens = count;
	free(processed_text);
	free(dp);
	return output_ids;

_tokenize_sp_error:
	free(dp);
	free(processed_text);
	*num_tokens = 0;
	return NULL;
}

unsigned int decode_utf8(unsigned int *state, unsigned int *codep, unsigned char byte)
{
	if (*state == 0) {
		if (byte < 0x80) {
			*codep = byte;
			return *codep;
		} else if ((byte & 0xE0) == 0xC0) {
			*state = 1;
			*codep = byte & 0x1F;
		} else if ((byte & 0xF0) == 0xE0) {
			*state = 2;
			*codep = byte & 0x0F;
		} else if ((byte & 0xF8) == 0xF0) {
			*state = 3;
			*codep = byte & 0x07;
		} else {
			return 0xFFFD;
		}
	} else {
		if ((byte & 0xC0) == 0x80) {
			*codep = (*codep << 6) | (byte & 0x3F);
			(*state)--;
			if (*state == 0) {
				return *codep;
			}
		} else {
			*state = 0;
			*codep = 0;
			return 0xFFFD;
		}
	}
	return 0;
}

// UTF-8 streaming decode
void token_out_utf8_stream(struct TIEContext *ctx, const char *p, int len)
{
	for (char *c = (char *)p; len > 0; c++, len--) {
		unsigned int result = decode_utf8(&ctx->utf8_state, &ctx->utf8_codepoint, (unsigned char)*c);
		if (result != 0 && result != 0xFFFD) {
			printf("%c", result);
		}
	}
}

// SentencePiece detokenization
void token_out_sp(struct TIEContext *ctx, const char *p, int len)
{
	for (int i = 0; i < len; i++) {
		unsigned char ch = (unsigned char)p[i];
		if (ch == 0xE2 && i + 2 < len && (unsigned char)p[i + 1] == 0x96 && (unsigned char)p[i + 2] == 0x81) {
			// UTF-8 for "▁" (U+2581)
			putchar(' ');
			i += 2; // skip extra bytes
		} else {
			putchar(ch);
		}
	}
}

int init_token_table(struct TIEContext *ctx, int num_tokens)
{
	ctx->tokenizer.token_table = calloc(num_tokens, sizeof(unsigned char *));

	ctx->tokenizer.token_lens = calloc(num_tokens, sizeof(int));
	ctx->tokenizer.token_types = calloc(num_tokens, sizeof(int));
	ctx->tokenizer.token_scores = calloc(num_tokens, sizeof(float));

	ctx->tokenizer.token_count = num_tokens;

	if (!ctx->tokenizer.token_table || !ctx->tokenizer.token_lens || !ctx->tokenizer.token_types) {
		fprintf(stderr, "Failed to allocate token table\n");
		return -1;
	}

	return 0;
}
