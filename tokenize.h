#ifndef __TOKENIZE_H__
#define __TOKENIZE_H__

#include <inttypes.h>
#include <stdbool.h>

#define BPE_MAP_CAPACITY (256 * 1024)
#define MAX_SPECIAL_TOKENS 8192
#define MAX_CHUNKS 512
#define MAX_SPANS 2048

typedef struct TrieNode {
	struct TrieNode *children[256];
	int token_id;
} TrieNode;

typedef struct StringPool {
	char *data;
	size_t size;
	size_t capacity;
} StringPool;

typedef struct {
	const char *ptr;
	int len;
	bool is_special;
	int token_id;
} TokenChunk;

typedef struct {
	int token_id;
	const char *start;
	int length;
	bool is_special;
} BpeTokenSpan;

typedef struct {
	uint64_t key;
	uint32_t rank;
	bool occupied;
} bpe_entry_t;

typedef struct {
	bpe_entry_t table[BPE_MAP_CAPACITY];
} bpe_merge_map_t;

typedef struct {
	const char *text; // Pointer into string pool
	int token_id;
	size_t length;
} SpecialToken;

typedef struct {
	SpecialToken specials[MAX_SPECIAL_TOKENS];
	int count;
} SpecialTokenList;

typedef struct {
    float score;      // The best score to reach this position
    int token_id;   // The ID of the last token on the best path
    int backpointer; // The starting index of that last token
} DP_Entry;

typedef struct {
	TrieNode *root;
	StringPool *pool;
	unsigned char **token_table; // Points to each token string in pool->data
	int *token_lens;	     // Length of each token
	int *token_types;	     	// Token type
	float *token_scores;	     	// Token scores
	int token_count;	     // Total number of tokens
} Tokenizer;

extern struct ctx_t *ctx;

extern char merge_pool[MAX_SPANS][128];
extern bpe_merge_map_t bpe_merges_map;
extern SpecialTokenList special_tokens;

extern TrieNode *create_node(void);
extern StringPool *create_string_pool(size_t initial_capacity);
extern int append_to_pool(StringPool *pool, const char *str, size_t len);
extern void insert_token(TrieNode *root, const char *token, size_t len, int token_id);
extern void free_trie(TrieNode *node);
extern void free_string_pool(StringPool *pool);

extern void bpe_map_insert(bpe_merge_map_t *map, uint64_t key, uint32_t rank);
extern void replace_g_spaces(char *s);

extern const unsigned char *get_token_string(const struct ctx_t *ctx, int token_id);
extern int get_token_string_length(const struct ctx_t *ctx, int token_id);

extern int vocab_lookup_token_id(TrieNode *root, const char *token, size_t len);

extern int *tokenize_bpe(struct ctx_t *ctx, const char *text, size_t *num_tokens);
extern int *tokenize_sp(struct ctx_t *ctx, const char *text, size_t *num_tokens);

#endif
