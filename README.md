# Toy Inference Engine (TIE)

A **minimal CPU-only** transformer inference engine designed to run Qwen3 models (currently supports Qwen3 0.6B / 1.7B / 4B).  
It includes an interactive mode, tokenizer, rope, KV cache, attention, and tool-call support.

Not a speed daemon (about 2.4 TPS for 4B model) — but designed for simplicity, portability, and experimentation.

---

## Features

-  Runs Qwen3 GGUF models (0.6B, 1.7B, 4B tested)
-  Implements tokenizer, RMS norm, RoPE, attention, FFN
-  Interactive chat with multi-turn memory
-  `<tool_call>` + `<tool_response>` support
-  AVX2 + FMA acceleration
-  Batched prompt processing
-  Multi-threaded via custom thread pool
-  Loads GGUF metadata and tensor layout

---

## Example: Tool Call in Action

```
Welcome to Qwen3 interactive chat. Type 'exit' to quit.

You: please turn off the light

--- Prompt Processing at pos: 0 (Matrix Mode) ---
--- Prompt Processing Complete 163 tokens, time: 22639 msec ---

--- Generation Start at pos: 163 (Max 8192 new tokens) ---
Qwen3: <think>
Okay, the user says "please turn off the light"... 
</think>

<tool_call>
{"name": "set_lamp_state", "arguments": {"state": "off"}}
</tool_call><|im_end|>

--- Tool Call Detected ---
Function: 'set_lamp_state', Location: 'off'
Tool Result: {"state": "off"}

You: new prompt: 
<|im_start|>user
<tool_response>{"state": "off"}</tool_response><|im_end|>
<|im_start|>assistant

--- Generation Start at pos: 321 (Max 8192 new tokens) ---
Qwen3: <think>
... I used the set_lamp_state function with "off" as the state ...
</think>

The lamp has been turned off.<|im_end|>
```

---

## Model Details (example config)

```
Model: Qwen3-4B (Unsloth)
Embed Dim: 2560
Layers: 36
Attention Heads: 32 (KV Heads: 8)
FFN Dim: 9728
Context Length: 40960
Rope Base: 1000000.0
Tokenizer: GPT2-style (Qwen2 pretokenizer)
Special tokens: 26
```

---

## Requirements

- Clang or GCC (x86_64 with AVX2 + FMA)
- 16GB RAM for 4B model
- macOS

---

## License

MIT — use it for research, experimentation, or as a foundation for your own CPU-based inference ideas.

---

## Notes

- Currently supports BF16 and Q6_K models only
- Hackable

---

## Contact

Built by [szemzoa](https://github.com/szemzoa)
