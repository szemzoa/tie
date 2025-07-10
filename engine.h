#ifndef __ENGINE_H__
#define __ENGINE_H__

#include "main.h"

#define SILU_TABLE_SIZE (8 * 1024)
#define SILU_X_MIN -10.0f
#define SILU_X_MAX 10.0f

extern float silu_table[SILU_TABLE_SIZE];

extern void silu_table_init(void);
extern void rope_cache_init(struct ctx_t *ctx, int max_pos, int head_dim, float base);

extern void rms_norm(float *__restrict o, const float *__restrict x, const float *__restrict weight, int size,
		     float eps);

extern int transformer_layer_unified(struct ctx_t *ctx, int layer_idx, int batch_len, bool use_threads);

#endif
