#ifndef __TOKENIZE_H__
#define __TOKENIZE_H__

#include <inttypes.h>
#include <stdbool.h>

#define BPE_MAP_CAPACITY (512 * 1024)
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
} BpeEntry;

typedef struct {
	BpeEntry table[BPE_MAP_CAPACITY];
} BpeMergeMap;

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
	float score;	 // The best score to reach this position
	int token_id;	 // The ID of the last token on the best path
	int backpointer; // The starting index of that last token
} DP_Entry;

typedef struct {
	TrieNode *root;
	StringPool *pool;
	unsigned char **token_table; // Points to each token string in pool->data
	int *token_lens;	     // Length of each token
	int *token_types;	     // Token type
	float *token_scores;	     // Token scores
	int token_count;	     // Total number of tokens

	// Direct-lookup map: bpe_decoder_map[codepoint] -> byte_value
	int bpe_decoder_map[512];
	uint32_t bpe_encoder_map[256];

	unsigned int utf8_state;
	unsigned int utf8_codepoint;
} Tokenizer;

struct TIEContext;

extern char merge_pool[MAX_SPANS][128];
extern BpeMergeMap bpe_merges_map;
extern SpecialTokenList special_tokens;

extern TrieNode *create_node(void);
extern StringPool *create_string_pool(size_t initial_capacity);
extern int append_to_pool(StringPool *pool, const char *str, size_t len);
extern void insert_token(TrieNode *root, const char *token, size_t len, int token_id);
extern void free_trie(TrieNode *node);
extern void free_string_pool(StringPool *pool);

extern void bpe_map_insert(BpeMergeMap *map, uint64_t key, uint32_t rank);
extern void replace_g_spaces(char *s);

extern const unsigned char *get_token_string(const struct TIEContext *ctx, int token_id);
extern int get_token_string_length(const struct TIEContext *ctx, int token_id);

extern int vocab_lookup_token_id(TrieNode *root, const char *token, size_t len);

extern int *tokenize_bpe(struct TIEContext *ctx, const char *text, size_t *num_tokens);
extern int *tokenize_sp(struct TIEContext *ctx, const char *text, size_t *num_tokens);

extern void token_out_utf8_stream(struct TIEContext *ctx, int token_id);
extern void token_out_sp(struct TIEContext *ctx, int token_id);

extern int init_token_table(struct TIEContext *ctx, int num_tokens);

extern int build_vision_tokens_gemma3(struct TIEContext *ctx, int *token_buf, int buf_pos);
extern int build_vision_tokens_qwen3vl(struct TIEContext *ctx, int *token_buf, int buf_pos);

#endif
