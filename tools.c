#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tools.h"
#include "model.h"
#include "main.h"


static char *set_lamp_state(const char *json_args)
{
	static char result[256];

	if (strstr(json_args, "on")) {
		snprintf(result, sizeof(result), "{\"status\": \"success\", \"state\": \"on\"}");
	} else {
		snprintf(result, sizeof(result), "{\"status\": \"success\", \"state\": \"off\"}");
	}
	return result;
}

// Initialize the tool system
int tools_init(struct TIEContext *ctx)
{
	ToolContext *tc = &ctx->tool_context;

	memset(tc, 0, sizeof(ToolContext));

	tc->state = TOOL_STATE_IDLE;

	// Architecture Detection & Token Lookup
	if (ctx->gguf_text->arch == ARCH_QWEN3 || ctx->gguf_text->arch == ARCH_QWEN3_MOE
	    || ctx->gguf_text->arch == ARCH_QWEN3VL || ctx->gguf_text->arch == ARCH_QWEN3VL_MOE
	    || ctx->gguf_text->arch == ARCH_GRANITE || ctx->gguf_text->arch == ARCH_QWEN35 || ctx->gguf_text->arch == ARCH_QWEN35_MOE) {

		tc->token_start_id = vocab_lookup_token_id(ctx->tokenizer.root, "<tool_call>", 11);
		tc->token_end_id = vocab_lookup_token_id(ctx->tokenizer.root, "</tool_call>", 12);

		if (tc->token_start_id == -1)
			printf("WARN: <tool_call> token not found!\n");

		return 0;

	}
#if 1
	 else if (ctx->gguf_text->arch == ARCH_LFM2 || ctx->gguf_text->arch == ARCH_LFM2_MOE) {

		tc->token_start_id = vocab_lookup_token_id(ctx->tokenizer.root, "<|tool_call_start|>", 19);
		tc->token_end_id = vocab_lookup_token_id(ctx->tokenizer.root, "<|tool_call_end|>", 17);

		if (tc->token_start_id == -1)
			printf("WARN: <|tool_call_start|> token not found!\n");

		return 0;
	}
#endif
	else {
		// Default
		tc->token_start_id = -1; // Disable if not supported
		return -1;
	}
}

void tools_release(struct TIEContext *ctx)
{
	//	ToolContext *tc = &ctx->tool_context;
	//	if (tc->result_prompt)
	//		free(tc->result_prompt);
}

// Execution Logic
void tools_execute_pending(struct TIEContext *ctx)
{
	ToolContext *tc = &ctx->tool_context;

	// Parse the buffer (e.g., {"name": "set_lamp_state", ...})
	printf(DEBUG "[System] Executing Tool: %s\n", tc->buffer);

	// Simple parser for demo
	char *execution_result = NULL;
	if (strstr(tc->buffer, "set_lamp_state")) {
		execution_result = set_lamp_state(tc->buffer);
	} else {
		execution_result = "{\"error\": \"Unknown tool\"}";
	}

	// Format the result for the model
	// Qwen expects:
	// <|im_start|>user\n<tool_response>\n{result}\n</tool_response><|im_end|>\n<|im_start|>assistant\n
	char fmt_buf[4096];
//	snprintf(fmt_buf, sizeof(fmt_buf),
//		 "<|im_start|>user\n<tool_response>\n%s\n</tool_response><|im_end|>\n<|im_start|>assistant\n",
//		 execution_result);

	snprintf(fmt_buf, sizeof(fmt_buf),
		 "<|im_start|>tool\n<|tool_response_start|>\n%s\n<|tool_response_end|><|im_end|>\n<|im_start|>assistant\n",
		 execution_result);

	tc->result_prompt = strdup(fmt_buf);

	tc->state = TOOL_STATE_RESULT_READY;
}

// Process a token to see if it triggers/continues a tool call
bool tools_process_token(struct TIEContext *ctx, int token)
{
	ToolContext *tc = &ctx->tool_context;


	// Safety check
	if (tc->token_start_id == -1)
		return false; // Tools disabled

	// STATE: IDLE
	if (tc->state == TOOL_STATE_IDLE) {
		if (token == tc->token_start_id) {
			// Found <tool_call>. Start buffering. Swallow this token.
			printf("tool call start\n");
			tc->state = TOOL_STATE_BUFFERING;
			tc->buf_len = 0;
			tc->buffer[0] = '\0';
			return true; // Consume token
		}
		return false;
	}

	// STATE: BUFFERING
	if (tc->state == TOOL_STATE_BUFFERING) {
		if (token == tc->token_end_id) {
			// Found </tool_call>. Execute.
			tc->buffer[tc->buf_len] = '\0';

			printf("tool call end\n");

			tc->state = TOOL_STATE_CALL_READY;
			return true;
		}

		// Append token string to buffer
		char piece[256];
		int len = ctx->model->interface.tokenize_decode(ctx, token, piece, sizeof(piece));

		if (tc->buf_len + len < TOOL_BUFFER_SIZE - 1) {
			memcpy(tc->buffer + tc->buf_len, piece, len);
			tc->buf_len += len;
			tc->buffer[tc->buf_len] = '\0';
		}

		return true;
	}

	return false;
}
