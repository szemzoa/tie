#ifndef __CONFIG_H__
#define __CONFIG_H__

#define CONFIG_ENABLE_AVX2

/*  Internal memory buffer types:
 *	- GGML_TYPE_BF16	- incomplete
 *	- GGML_TYPE_F32
 */
#define INTERNAL_MEMORY_TYPE	GGML_TYPE_F32
#define KV_CACHE_TYPE		GGML_TYPE_BF16

#define MAX_PROMPT_BATCH_SIZE 	1024
#define MAX_THREADS 		32
#define MAX_TASKS 		128

#define TOOL_CALL_BUFFER_SIZE 	1024

//#define DEBUG_TENSORS
//#define DEBUG_TOKENS

#endif
