#ifndef __PREDICT_H__
#define __PREDICT_H__

#include <inttypes.h>

extern int predict_next_token(float *logits, int vocab_size, const char *method, float temperature, int k, float p,
							  const int *prompt_tokens, int prompt_len, const int *generated_tokens, int gen_len,
							  float repetition_penalty);

#endif
