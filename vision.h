#ifndef __VISION_H__
#define __VISION_H__

struct TIEContext;

typedef struct {
	int image_size;	     // e.g. 896
	float image_mean[3]; // e.g. {0.5, 0.5, 0.5}
	float image_std[3];  // e.g. {0.5, 0.5, 0.5}
} clip_vision_meta_t;

extern clip_vision_meta_t clip_vision_meta;

extern int load_bmp_clip(const char *filename, const clip_vision_meta_t *meta, struct TIEContext *ctx);
extern MemType *process_image_vision(struct TIEContext *ctx);

#endif