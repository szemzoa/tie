#ifndef __VISION_H__
#define __VISION_H__

struct TIEContext;

extern int load_bmp_clip(const char *filename, struct TIEContext *ctx);
extern MemType *process_image_vision(struct TIEContext *ctx);

#endif