#ifndef __MODEL_H__
#define __MODEL_H__

#include <stdint.h>
#include <stdbool.h>
#include "main.h"

extern int model_load(struct ctx_t *ctx, int use_mmap, int context_length);
extern int model_init(struct ctx_t *ctx, float yarn_scale_factor, float repetiton_penality);
extern void model_cleanup(struct ctx_t *ctx, int use_mmap);

#endif
