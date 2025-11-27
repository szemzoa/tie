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
	    || ctx->gguf_text->arch == ARCH_QWEN3VL || ctx->gguf_text->arch == ARCH_QWEN3VL_MOE) {

		tc->token_start_id = vocab_lookup_token_id(ctx->tokenizer.root, "<tool_call>", 11);
		tc->token_end_id = vocab_lookup_token_id(ctx->tokenizer.root, "</tool_call>", 12);

		if (tc->token_start_id == -1)
			printf("WARN: <tool_call> token not found!\n");

		return 0;

	} else {
		// Default
		tc->token_start_id = -1; // Disable if not supported
		return -1;
	}
}

void tools_release(struct TIEContext *ctx)
{
	ToolContext *tc = &ctx->tool_context;

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
	snprintf(fmt_buf, sizeof(fmt_buf),
		 "<|im_start|>user\n<tool_response>\n%s\n</tool_response><|im_end|>\n<|im_start|>assistant\n",
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

			tc->state = TOOL_STATE_CALL_READY;
			return true;
		}

		// Append token string to buffer
		char piece[256];
		int len = ctx->model->interface.decode_token(ctx, token, piece, sizeof(piece));

		if (tc->buf_len + len < TOOL_BUFFER_SIZE - 1) {
			memcpy(tc->buffer + tc->buf_len, piece, len);
			tc->buf_len += len;
			tc->buffer[tc->buf_len] = '\0';
		}

		return true;
	}

	return false;
}

// Build the architecture-specific system prompt
char *build_system_prompt_qwen3(struct TIEContext *ctx)
{
	// For Qwen, this is the standard tool prompt
	return strdup(
		"<|im_start|>system\n"
		"You are a helpful assistant.\n"
		"\n# Tools\n"
		"You may call one or more functions to assist with the user query.\n"
		"You are provided with function signatures within <tools></tools> XML tags:\n"
		"<tools>\n"
		"{\"type\": \"function\", \"function\": { \"name\": \"set_lamp_state\", \"description\": \"Turn lamp on/off\", \"parameters\": { \"type\": \"object\", \"properties\": { \"state\": {\"type\": \"string\", \"enum\": [\"on\", \"off\"]} }, \"required\": [\"state\"] } }}\n"
		"</tools>\n"
		"\n# Tool Instructions\n"
		"For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
		"<tool_call>\n"
		"{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
		"</tool_call>\n"
		"<|im_end|>\n");
}

char *build_system_prompt_gemma3(struct TIEContext *ctx)
{
	return strdup(
		"<bos><start_of_turn>system\n"
		"You have access to functions. If you decide to invoke any of the function(s), you MUST put it in the format of\n"
		"{\"name\": function name, \"parameters\": dictionary of argument name and its value}\n"
		"\n"
		"IMPORTANT: You must use the EXACT function names and parameter names defined below.\n"
		"You SHOULD NOT include any other text in the response if you call a function\n"
		"[\n"
		"  {\n"
		"    \"name\": \"set_lamp_state\",\n"
		"    \"description\": \"Control a smart lamp\",\n"
		"    \"parameters\": {\n"
		"      \"type\": \"object\",\n"
		"      \"properties\": {\n"
		"        \"state\": {\n"
		"          \"type\": \"string\",\n"
		"          \"enum\": [\"on\", \"off\"],\n"
		"          \"description\": \"Turn the lamp on or off\"\n"
		"        }\n"
		"      },\n"
		"      \"required\": [\"state\"]\n"
		"    }\n"
		"  }\n"
		"]\n"
		"<end_of_turn>\n");
}
