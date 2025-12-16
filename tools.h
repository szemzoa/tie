#ifndef __TOOLS_H__
#define __TOOLS_H__

#include <stdbool.h>
#include "config.h"

struct TIEContext;

typedef enum { TOOL_STATE_IDLE, TOOL_STATE_BUFFERING, TOOL_STATE_CALL_READY, TOOL_STATE_RESULT_READY } ToolState;

typedef struct {
	ToolState state;
	char buffer[TOOL_BUFFER_SIZE];
	size_t buf_len;
	char *result_prompt;
	int token_start_id;
	int token_end_id;
	void (*output_callback)(const char *text, void *user_data);
} ToolContext;

typedef struct {
	char *name;
	char *description;
	char *parameters_json;
	char *(*func)(const char *args);
} ToolDef;

extern int  tools_init(struct TIEContext *ctx);
extern void tools_release(struct TIEContext *ctx);
extern void tools_register_default(struct TIEContext *ctx);
extern bool tools_process_token(struct TIEContext *ctx, int token);
extern void tools_execute_pending(struct TIEContext *ctx);

extern char *build_system_prompt_qwen3(struct TIEContext *ctx);
extern char *build_system_prompt_gemma3(struct TIEContext *ctx);
extern char *build_system_prompt_granite(struct TIEContext *ctx);

#endif
