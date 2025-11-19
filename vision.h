#ifndef __VISION_H__
#define __VISION_H__

struct TIEContext;

extern int load_bmp_clip(const char *filename, struct TIEContext *ctx);

extern void vision_mrope_cache_init(MRopeCacheType *cache, int num_patches_h, int num_patches_w, int head_dim,
				    float rope_base);

extern MemType *process_image_vision_gemma3(struct TIEContext *ctx);
extern MemType *process_image_vision_qwen3vl(struct TIEContext *ctx);

#endif