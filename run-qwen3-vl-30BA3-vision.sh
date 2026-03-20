#!/bin/sh

# example prompt:
# /img australia-512x336.bmp
# Describe the image

./tie -m Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf -t 12 -c 4096 -p Qwen3-VL-30B-A3B-Instruct-mmproj-bf16.gguf
