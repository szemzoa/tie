#ifndef __CONFIG_H__
#define __CONFIG_H__

#define CONFIG_ENABLE_AVX2

/*  Internal memory buffer types:
 *	- GGML_TYPE_F32
 */
#define INTERNAL_MEMORY_TYPE	GGML_TYPE_F32
#define KV_CACHE_TYPE		GGML_TYPE_BF16

#define MAX_PROMPT_BATCH_SIZE 	4096
#define MAX_THREADS 		32
#define MAX_TASKS 		128

#define TOOL_BUFFER_SIZE 	2048

//#define DEBUG_TENSORS
//#define DEBUG_TOKENS



#define CLR_RESET "\033[0m"
#define CLR_BOLD "\033[1m"
#define CLR_DIM "\033[2m"

#define CLR_RED "\033[31m"
#define CLR_GREEN "\033[32m"
#define CLR_YELLOW "\033[33m"
#define CLR_BLUE "\033[34m"
#define CLR_MAGENTA "\033[35m"
#define CLR_CYAN "\033[36m"
#define CLR_WHITE "\033[37m"

// helpers
#define USER_PROMPT 	CLR_BOLD CLR_GREEN
#define ASSISTANT_OUT 	CLR_BOLD CLR_CYAN
#define DEBUG 		CLR_DIM CLR_YELLOW
#define THINK 		CLR_MAGENTA
#define ERR 		CLR_BOLD CLR_RED

#endif
