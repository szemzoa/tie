# Toy Inference Engine (TIE)

A **minimal CPU-only** transformer inference engine. Supports Qwen3, Gemma-3 and Gemma-3n based models.  
It includes an interactive mode, tokenizer, rope, KV cache, attention, and tool-call support.

Not breaking any speed records (~4.5 TPS for 4B model (Q4_K_M), ~10 for 1.7B Qwen3) — but designed for simplicity and experimentation.

## Features

-  Runs Qwen3, Qwen3-MoE GGUF models (Dense 0.6B/1.7B/4B/8B, MoE 30B-A3B tested)
-  Runs Gemma3 GGUF model (Gemma-3 270m/1b/4b/12b-it tested)
-  Runs Gemma-3n GGUF model (Gemma-3n-E2B/E4B-it tested)
-  Minimal Gemma3 clip vision support
-  Implements tokenizer, RMS norm, RoPE, attention, FFN
-  Interactive chat with multi-turn memory
-  Minimal tool call support
-  AVX2 + FMA acceleration
-  Batched prompt processing
-  Multi-threaded via custom thread pool
-  Loads GGUF metadata and tensor layout
-  No external library dependency

---

## Example: 0.6B processing

```
Welcome to Qwen3 interactive chat. Type 'exit' to quit.

You: Hi! Please tell me something about you! /no_think
--- Prompt Processing at pos: 0 (Matrix Mode) 21 tokens---
--- Prompt Processing Complete 21 tokens, time: 408 msec ---

--- Generation Start at pos: 21 (Max 8192 new tokens) ---
Qwen3: <think>

</think>

Hello! I'm an AI assistant that helps with various tasks and questions. I don't have personal experiences or feelings, but I can assist you with any questions you have. Let me know how I can help!<|im_end|>
--- Generation End --- 48 tokens, 2055 msec, tps: 23.4
```

--- Prompt Processing at pos: 0 (Matrix Mode) 21 tokens---
--- Prompt Processing Complete 21 tokens, time: 408 msec ---

--- Generation Start at pos: 21 (Max 8192 new tokens) ---
Qwen3: <think>

</think>

Hello! I'm an AI assistant that helps with various tasks and questions. I don't have personal experiences or feelings, but I can assist you with any questions you have. Let me know how I can help!<|im_end|>
--- Generation End --- 48 tokens, 2055 msec, tps: 23.4

---

## Example: Gemma3-4B Vision processing (896x896x24bit uncompressed BMP only)

```
Initialized Gemma-3 language model with the following configuration:
Embed Dim: 2560, Layers: 34, Heads: 8, KV Heads: 4, Head Dim: 256, Shared KV layers: 0
FFN Dim: 10240, Rope Base: 1000000.0, Seq Len: 4096, Vocab: 262208
Yarn Scale: 1.00, eps: 0.000001, rope_scale: 1.0, sliding_window: 1024
Gemma3-clip init
Initializing vision memory buffers...
Initialized Gemma-3 vision model with the following configuration:
Image size: 896, Proj Dim: 2560, Patch size: 14
Embed Dim: 1152, FFN Dim: 4304, Layers: 27, Heads: 16, eps: 0.000001
Proj Scale Factor: 4
Welcome to interactive chat. Type '/exit' to quit.
You: Describe the image.
--- Prompt Processing at pos: 0 (Matrix Mode) 278 tokens ---
Processing image...........................done
--- Multimodal Prompt Processing at pos: 0 (278 tokens) ---
--- Multimodal Prompt Processing Complete ---
--- Prompt Processing Complete 278 tokens, time: 157226 msec ---

--- Generation Start (Max 8192 new tokens) ---
Here's a detailed description of the image:

**Overview:**

The image presents a breathtaking landscape of a lush, green valley with a serene lake nestled within. It’s a classic, idyllic scene of nature’s beauty.

**Key Elements:**

*   **Sky:** The sky is a brilliant, clear blue, dotted with fluffy white cumulus clouds. They are evenly distributed and add a sense of space and openness to the scene.
*   **Mountains/Hills:**  Two prominent, forested hills flank the lake. They are densely covered in vibrant green trees, creating a strong contrast with the blue sky and water. The hills rise smoothly into the distance.
*   **Lake:** The lake itself is a striking teal or emerald green color. It appears calm, with gentle ripples on the surface reflecting the sky and surrounding trees. The water’s color is very deep, hinting at considerable depth.
*   **Vegetation:** The trees are lush and full, suggesting a healthy and thriving ecosystem. The sheer volume of greenery dominates the view, emphasizing the natural abundance of the area.

**Composition and Atmosphere:**
...
```
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

