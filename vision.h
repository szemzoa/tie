#ifndef __VISION_H__
#define __VISION_H__

struct TIEContext;

extern int load_bmp_clip(const char *filename, struct TIEContext *ctx);

extern void vision_mrope_cache_init(MRopeCacheType *cache, int num_patches_h, int num_patches_w, int head_dim,
				    float rope_base);

extern MemType *process_image_vision(struct TIEContext *ctx);

extern void vision_create_embeddings_gemma3(struct TIEContext *ctx);
extern void vision_create_embeddings_qwen3vl(struct TIEContext *ctx);

extern void vision_transformer_layer_gemma3(struct TIEContext *ctx, int layer_idx);
extern void vision_transformer_layer_qwen3vl(struct TIEContext *ctx, int layer_idx);

#endif