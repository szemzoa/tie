#ifndef __ENGINE_H__
#define __ENGINE_H__

#include "main.h"

#define SILU_TABLE_SIZE 	(8 * 1024)
#define SILU_X_MIN 		-10.0f
#define SILU_X_MAX 		10.0f

extern float silu_table[SILU_TABLE_SIZE];

extern void rope_cache_init(struct ctx_t *ctx, int max_pos, int head_dim, float base);

extern void silu_table_init(void);

extern float silu_lookup(float x);

extern void kv_cache_reset(struct ctx_t *ctx);

extern MemType mem_slice(MemType *buffer, size_t offset_elements);

extern int transformer_layer(struct ctx_t *ctx, int layer_idx, int batch_len);

#endif
