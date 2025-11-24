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

static inline void buf_write_byte(char *buf, int *pos, int max_len, unsigned char byte)
{
	if (*pos < max_len - 1) { // Leave room for null terminator
		buf[(*pos)++] = (char)byte;
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

// Encodes a single Unicode codepoint into a UTF-8 byte sequence.
void encode_cp_to_utf8(uint32_t cp, char *dest_buf, size_t *len)
{
	if (cp < 0x80) { // 1 byte
		dest_buf[0] = (char)cp;
		*len = 1;
	} else if (cp < 0x800) { // 2 bytes
		dest_buf[0] = (char)(0xC0 | (cp >> 6));
		dest_buf[1] = (char)(0x80 | (cp & 0x3F));
		*len = 2;
	} else if (cp < 0x10000) { // 3 bytes
		dest_buf[0] = (char)(0xE0 | (cp >> 12));
		dest_buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
		dest_buf[2] = (char)(0x80 | (cp & 0x3F));
		*len = 3;
	} else if (cp <= 0x10FFFF) { // 4 bytes
		dest_buf[0] = (char)(0xF0 | (cp >> 18));
		dest_buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
		dest_buf[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
		dest_buf[3] = (char)(0x80 | (cp & 0x3F));
		*len = 4;
	} else {
		// Error or 0xFFFD
		dest_buf[0] = '?';
		*len = 1;
	}
}

char *munge_chunk(struct TIEContext *ctx, const char *chunk, size_t chunk_len, size_t *out_munged_len)
{
	// Allocate a worst-case buffer (each byte can become 4 UTF-8 bytes)
	char *munged_buf = malloc(chunk_len * 4 + 1);
	size_t out_idx = 0;

	for (size_t i = 0; i < chunk_len; i++) {
		unsigned char byte = (unsigned char)chunk[i];

		// Look up byte -> codepoint
		uint32_t cp = ctx->tokenizer.bpe_encoder_map[byte];

		// Encode codepoint -> UTF-8 bytes
		size_t utf8_len;
		encode_cp_to_utf8(cp, &munged_buf[out_idx], &utf8_len);
		out_idx += utf8_len;
	}
	munged_buf[out_idx] = '\0';
	*out_munged_len = out_idx;
	return munged_buf;
}

int run_bpe_on_munged_string(struct TIEContext *ctx, const char *munged_text, size_t len, int *out_tokens)
{
	BpeTokenSpan spans[MAX_SPANS];
	int span_count = 0;
	size_t p = 0;

	// Find initial tokens using the Trie
	while (p < len) {
		int token_id;
		size_t prev_p = p;
		if (tokenize_step(ctx->tokenizer.root, munged_text, len, &p, &token_id)) {
			spans[span_count++] = (BpeTokenSpan){.token_id = token_id,
							     .start = &munged_text[prev_p],
							     .length = (int)(p - prev_p),
							     .is_special = false};
		} else {
			// This should not happen if the vocab is correct, but as a fallback:
			spans[span_count++] = (BpeTokenSpan){.token_id = -1, // Will be handled as '?'
							     .start = &munged_text[p],
							     .length = 1,
							     .is_special = false};
			p++;
		}
	}

	// Run BPE merge loop
	while (1) {
		int best_rank = INT32_MAX, best_idx = -1;

		for (int i = 0; i < span_count - 1; i++) {
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
			break; // No more merges found

		// Merge the best pair
		int len1 = spans[best_idx].length;
		int len2 = spans[best_idx + 1].length;

		if (len1 + len2 > 127) // Safety check for merge_pool buffer
			continue;

		char *buf = merge_pool[best_idx]; // Reuse the static merge_pool
		memcpy(buf, spans[best_idx].start, len1);
		memcpy(buf + len1, spans[best_idx + 1].start, len2);
		int merged_len = len1 + len2;

		int merged_id = vocab_lookup_token_id(ctx->tokenizer.root, buf, merged_len);
		if (merged_id < 0) {
			fprintf(stderr, "ERROR: vocab lookup failed for merge '%.*s'\n", merged_len, buf);
			// This merge is invalid, let's stop this path
			spans[best_idx].token_id = -2; // Mark as "don't merge again"
			spans[best_idx + 1].token_id = -2;
			continue;
		}

		spans[best_idx].token_id = merged_id;
		spans[best_idx].start = buf;
		spans[best_idx].length = merged_len;
		spans[best_idx].is_special = false;

		// Shift remaining spans over
		for (int i = best_idx + 1; i < span_count - 1; i++) {
			spans[i] = spans[i + 1];
		}
		span_count--;
	}

	// Copy final token IDs to output
	for (int i = 0; i < span_count; i++) {
		out_tokens[i] = spans[i].token_id;
	}

	return span_count;
}

int *tokenize_bpe(struct TIEContext *ctx, const char *text, size_t *num_tokens)
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
			chunks[chunk_count++] =
				(TokenChunk){.ptr = &text[start], .len = p - start, .is_special = false};
		}
	}

	// Process each chunk
	int *all_tokens = malloc(text_len * 2 * sizeof(int)); // Worst-case allocation
	int total_token_count = 0;

	for (int i = 0; i < chunk_count; i++) {
		TokenChunk *c = &chunks[i];

		if (c->is_special) {
			// It's a special token, just add its ID
			all_tokens[total_token_count++] = c->token_id;

		} else {
			const char *chunk_ptr = c->ptr;
			size_t chunk_len = c->len;
			size_t sub_p = 0;
			const char *sub_chunk_start = chunk_ptr;

			while (sub_p <= chunk_len) {
				// Find next space or end of chunk
				if (sub_p == chunk_len || chunk_ptr[sub_p] == ' ') {

					size_t sub_chunk_len = sub_p - (sub_chunk_start - chunk_ptr);

					// A. Process the word (if it exists)
					if (sub_chunk_len > 0) {
						size_t munged_len;
						char *munged_chunk =
							munge_chunk(ctx, sub_chunk_start, sub_chunk_len, &munged_len);

						int chunk_tokens[256]; // Temp buffer for one word
						int num_chunk_tokens = run_bpe_on_munged_string(
							ctx, munged_chunk, munged_len, chunk_tokens);

						memcpy(&all_tokens[total_token_count], chunk_tokens,
						       num_chunk_tokens * sizeof(int));
						total_token_count += num_chunk_tokens;
						free(munged_chunk);
					}

					// Process the space (if it exists)
					if (sub_p < chunk_len) {
						// " " (byte 32) -> "Ġ" (codepoint 288) -> bytes C4 A0
						// The 'munged' string for space is "Ġ"
						// We can look it up directly.

						// We need the token ID for the "munged" space
						// Munge the space
						size_t munged_space_len;
						char *munged_space = munge_chunk(ctx, " ", 1, &munged_space_len);

						// Look up its ID
						int space_token_id = vocab_lookup_token_id(
							ctx->tokenizer.root, munged_space, munged_space_len);
						if (space_token_id != -1) {
							all_tokens[total_token_count++] = space_token_id;
						}
						free(munged_space);
					}

					sub_chunk_start = &chunk_ptr[sub_p + 1];
				}
				sub_p++;
			}
		}
	}

	*num_tokens = total_token_count;
	return all_tokens;
}

// BPE detokenization
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

static void write_codepoint_to_buf(unsigned int cp, char *buf, int *pos, int max_len)
{
	if (cp == 0xFFFD) {
		// Replacement char EF BF BD
		buf_write_byte(buf, pos, max_len, 0xEF);
		buf_write_byte(buf, pos, max_len, 0xBF);
		buf_write_byte(buf, pos, max_len, 0xBD);
		return;
	}
	if (cp < 0x80) {
		buf_write_byte(buf, pos, max_len, (unsigned char)cp);
	} else if (cp < 0x800) {
		buf_write_byte(buf, pos, max_len, (0xC0 | (cp >> 6)));
		buf_write_byte(buf, pos, max_len, (0x80 | (cp & 0x3F)));
	} else if (cp < 0x10000) {
		buf_write_byte(buf, pos, max_len, (0xE0 | (cp >> 12)));
		buf_write_byte(buf, pos, max_len, (0x80 | ((cp >> 6) & 0x3F)));
		buf_write_byte(buf, pos, max_len, (0x80 | (cp & 0x3F)));
	} else if (cp <= 0x10FFFF) {
		buf_write_byte(buf, pos, max_len, (0xF0 | (cp >> 18)));
		buf_write_byte(buf, pos, max_len, (0x80 | ((cp >> 12) & 0x3F)));
		buf_write_byte(buf, pos, max_len, (0x80 | ((cp >> 6) & 0x3F)));
		buf_write_byte(buf, pos, max_len, (0x80 | (cp & 0x3F)));
	}
}

int decode_token_bpe(struct TIEContext *ctx, int token_id, char *buf, int buf_len)
{
	const char *p = (const char *)get_token_string(ctx, token_id);
	if (p == NULL)
		return 0;

	int len = get_token_string_length(ctx, token_id);
	int token_type = ctx->tokenizer.token_types[token_id];
	int written = 0;

	if (token_type == GGUF_TOKEN_TYPE_NORMAL) {
		for (int i = 0; i < len; i++) {
			unsigned int cp = decode_utf8(&ctx->tokenizer.utf8_state, &ctx->tokenizer.utf8_codepoint,
						      (unsigned char)p[i]);

			if (cp == 0)
				continue; // Partial UTF-8 byte

			if (cp == 0xFFFD || cp > 511) {
				buf_write_byte(buf, &written, buf_len, '?');
				continue;
			}

			int real_byte = ctx->tokenizer.bpe_decoder_map[cp];
			if (real_byte != -1) {
				buf_write_byte(buf, &written, buf_len, (unsigned char)real_byte);
			} else {
				buf_write_byte(buf, &written, buf_len, '?');
			}
		}
	} else {
		// Special Token / User Defined -> Literal Output
		for (int i = 0; i < len; i++) {
			unsigned int cp = decode_utf8(&ctx->tokenizer.utf8_state, &ctx->tokenizer.utf8_codepoint,
						      (unsigned char)p[i]);
			if (cp == 0)
				continue;
			write_codepoint_to_buf(cp, buf, &written, buf_len);
		}
	}

	buf[written] = '\0';
	return written;
}


// SentencePiece detokenization
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

int decode_token_sp(struct TIEContext *ctx, int token_id, char *buf, int buf_len)
{
	const char *p = get_token_string(ctx, token_id);
	int len = get_token_string_length(ctx, token_id);
	int written = 0;

	for (int i = 0; i < len; i++) {
		unsigned char ch = (unsigned char)p[i];
		// Handle Lower One Eighth Block (U+2581) -> Space
		if (ch == 0xE2 && i + 2 < len && (unsigned char)p[i + 1] == 0x96 && (unsigned char)p[i + 2] == 0x81) {
			buf_write_byte(buf, &written, buf_len, ' ');
			i += 2;
		} else {
			buf_write_byte(buf, &written, buf_len, ch);
		}
	}

	buf[written] = '\0';
	return written;
}

void bpe_map_init_decoder(int *map)
{
	// Initialize all entries to -1 (not found)
	for (int i = 0; i < 512; i++) {
		map[i] = -1;
	}

	// Map "printable" ASCII bytes (33 '!' to 126 '~')
	for (int b = 33; b <= 126; b++) {
		map[b] = b;
	}

	// Map "printable" Latin-1 bytes (161 '¡' to 255 'ÿ')
	//    (skipping 127-160 and 173, which are control/whitespace)
	for (int b = 161; b <= 172; b++) {
		map[b] = b;
	}
	for (int b = 174; b <= 255; b++) {
		map[b] = b;
	}

	// Map the 100 "munged" non-printable bytes (0-31, 127, etc.)
	int n = 0;
	for (int b = 0; b < 256; b++) {
		// If this byte 'b' does *not* have a direct 1-to-1 mapping...
		if (map[b] == -1) {
			// ...assign it the next "munged" codepoint (256 + n)
			map[256 + n] = b;
			n++;
		}
	}
}

void bpe_map_init_encoder(uint32_t *map)
{
	int n = 0;
	for (int b = 0; b < 256; b++) {
		// First, check if it's a "printable" 1-to-1 byte
		if ((b >= 33 && b <= 126) ||  // '!' to '~'
		    (b >= 161 && b <= 172) || // '¡' to '¬'
		    (b >= 174 && b <= 255))   // '®' to 'ÿ'
		{
			map[b] = b;
		} else {
			// It's a "non-printable" byte, map it to 256 + n
			map[b] = 256 + n;
			n++;
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

	// Generate the BPE byte-encoder/decoder map
	bpe_map_init_encoder(ctx->tokenizer.bpe_encoder_map);
	bpe_map_init_decoder(ctx->tokenizer.bpe_decoder_map);

	ctx->tokenizer.utf8_state = 0;
	ctx->tokenizer.utf8_codepoint = 0;

	return 0;
}

int build_vision_tokens_gemma3(struct TIEContext *ctx, int *token_buf, int buf_pos)
{
	ModelDef *def = ctx->model->def;
	int pos = buf_pos;

	token_buf[pos++] = def->params.double_newline_token_id; // DOUBLE NEWLINE
	token_buf[pos++] = def->params.vision_start_token_id;	// <start_of_image>

	for (int i = 0; i < 256; ++i)
		token_buf[pos++] = def->params.vision_embed_token_id;

	token_buf[pos++] = def->params.vision_end_token_id;	// <end_of_image>
	token_buf[pos++] = def->params.double_newline_token_id; // DOUBLE NEWLINE

	return pos - buf_pos;
}

int build_vision_tokens_qwen3vl(struct TIEContext *ctx, int *token_buf, int buf_pos)
{
	int pos = buf_pos;
	VisionModel *vm = ctx->model_vision;
	ModelDef *def = ctx->model->def;

	// Use Dynamic Dimensions from the loaded image
	int raw_w = ctx->vision_mem.image_raw_width;
	int raw_h = ctx->vision_mem.image_raw_height;

	// Calculate patches
	int w_patches = raw_w / vm->patch_size;
	int h_patches = raw_h / vm->patch_size;

	// Calculate merged grid size
	// Qwen3-VL reduces resolution by spatial_merge_size
	int merged_w = w_patches / vm->spatial_merge_size;
	int merged_h = h_patches / vm->spatial_merge_size;

	int num_patches = merged_w * merged_h;

	printf("Dynamic Vision Tokens: %dx%d image -> %dx%d patches -> %d tokens\n", raw_w, raw_h, merged_w, merged_h,
	       num_patches);

	// Add <|vision_start|> token
	token_buf[pos++] = def->params.vision_start_token_id;

	// Add the *same* <|vision_pad|> token N times
	int patch_token = def->params.vision_embed_token_id;

	for (int i = 0; i < num_patches; i++) {
		token_buf[pos++] = patch_token;
	}

	// Add <|vision_end|> token
	token_buf[pos++] = def->params.vision_end_token_id;

	// Add the required newline
	token_buf[pos++] = def->params.newline_token_id;

	return pos - buf_pos;
}
