# Toy Inference Engine (TIE)

A **minimal CPU-only** transformer inference engine. Supports Qwen3, Qwen3-VL, Gemma-3 and Gemma-3n based models.
It includes an interactive mode, tokenizer, rope, KV cache, attention, and tool-call support.

Not breaking any speed records (~4.5 TPS for 4B model (Q4_K_M), ~10 for 1.7B Qwen3) — but designed for simplicity and experimentation.

## Features
- Runs Qwen3, Qwen3-MoE and Qwen3-VL GGUF models (Dense 0.6B/1.7B/4B/8B, MoE 30B-A3B tested)
- Runs Gemma3 GGUF model (Gemma-3 270m/1b/4b/12b-it tested)
- Runs Gemma-3n GGUF model (Gemma-3n-E2B/E4B-it tested)
- Minimal Gemma3 clip vision support (896x896 uncompressed BMP only)
- Minimal Qwen3-VL vision support with dynamic resolution
- Implements tokenizer, RMS norm, RoPE, attention, FFN
- Interactive chat with multi-turn memory
- Minimal tool call support
- AVX2 + FMA acceleration
- Batched prompt processing
- Multi-threaded via custom thread pool
- Loads GGUF metadata and tensor layout
- No external library dependency
---

## Example: 0.6B processing
```
Welcome to Qwen3 interactive chat. Type '/exit' to quit.
You: Hi! Please tell me something about you! /no_think
--- Prompt Processing at pos: 0 (Matrix Mode) 21 tokens---
--- Prompt Processing Complete 21 tokens, time: 408 msec ---
--- Generation Start at pos: 21 (Max 8192 new tokens) ---

Qwen3: <think>
</think>

Hello! I'm an AI assistant that helps with various tasks and questions. I don't have personal experiences or feelings, but I can assist you with any questions you have. Let me know how I can help!<|im_end|>
--- Generation End --- 48 tokens, 2055 msec, tps: 23.4
```
---

## Example: image processing
```
[Qwen3-VL] init
init: RoPE cache
build_rope_cache_dynamic seq_len: 4096
Initializing KV cache
KV cache elements per layer: 4194304
KV cache uses: 576 MB
Initializing memory buffers...
Initialized Qwen3-VL language model with the following configuration:
Embed Dim: 2560, Layers: 36, Heads: 32, KV Heads: 8, Head Dim: 128, Shared KV layers: 0
FFN Dim: 9728, Rope Base: 5000000.0, Seq Len: 4096, Vocab: 151936
Yarn Scale: 1.00, eps: 0.000001, rope_scale: 1.0, sliding_window: 0
Context_length: 4096
M-RoPE sections: [24][20][20][0]
  
[Qwen3-VL-clip] init
Initializing vision memory buffers...
Initialized Qwen3-VL vision model with the following configuration:
Image size: 768, Proj Dim: 2560, Patch size: 16
Embed Dim: 1024, FFN Dim: 4096, Layers: 24, Heads: 16, eps: 0.000001
Proj Scale Factor: 1, Deepstack layers: 3
Succesfully loaded test-384x256.bmp image for vision process
  
Welcome to interactive chat. Type '/exit' to quit.
You: Please describe this image for me!
Dynamic Vision Tokens: 384x256 image -> 12x8 patches -> 96 tokens
--- Prompt Processing at pos: 0 (Matrix Mode) 119 tokens ---
Processing image QWEN3-VL
........................done
--- Multimodal Prompt Processing at pos: 0 (119 tokens) ---
--- Multimodal Prompt Processing Complete ---
--- Prompt Processing Complete 119 tokens, time: 24490 msec ---
  
--- Generation Start (Max 8192 new tokens) ---
This is a striking, high-contrast studio portrait of a woman in a dynamic, expressive pose...
```
---

## Example: Tool Call in Action

```

Welcome to Qwen3 interactive chat. Type '/exit' to quit.
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
  

## Requirements
- Clang or GCC (x86_64 with AVX2 + FMA)
- 8GB RAM for 4B model
- macOS, Linux

---

## License

MIT — use it for research, experimentation, or as a foundation for your own CPU-based inference ideas.

---

## Notes

- Currently supports BF16, Q6_K and Q4_K_M models only
- Hackable

---

## Contact

Built by [szemzoa](https://github.com/szemzoa)
