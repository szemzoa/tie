# Target
TARGET = tie

# Build revision
BUILD_REVISION_H = "build_revision.h"
BUILD_REVISION_D = "BUILD_REVISION"

SRCS = threadpool.c gguf.c maths.c predict.c tokenize.c model.c engine.c main.c

CC_ARGS= -g -O3 -mfma -mavx2 -ffast-math -fno-associative-math -march=native -ffp-contract=fast -Wall -Wno-pointer-sign -Wno-unused-label -funsigned-char $(INCLUDE_DIRS)

INCLUDE_DIRS= -I ./

LIB_DIR=
LIBS= -lm -lpthread

CROSS=
STRIP=$(CROSS)strip
CC=$(CROSS)gcc
HOSTCC=gcc
HOSTSTRIP=strip
DATE=/bin/date
CAT=/bin/cat
ECHO=/bin/echo
WORKDIR=$(/bin/pwd)
MAKE=make

# Objects
EXT_OBJS =
BUILD_OBJS = $(SRCS:.c=.o) 
OBJS = $(BUILD_OBJS) $(EXT_OBJS)

# Compiler flags to generate dependency files.
GENDEPFLAGS = -MD -MP -MF .dep/$(@F).d


all: begin build

begin:
	@echo "---------------------------------------------------------------"
	@echo -n "Compiler version: "
	@$(CC) -v 2>&1 | tail -1


.SILENT:

build: $(TARGET)
#	$(STRIP) $(TARGET)

.SECONDARY : $(TARGET)
.PRECIOUS : $(OBJS)
$(TARGET): $(OBJS)
	echo "  LD    $@"
	$(CC) $^ -o $@ $(LIB_DIR) $(LIBS) $(GENDEPFLAGS)
#	ar -cq $(TARGET).a $^ -o $@ 
 
%.o : %.c
	echo "  CC    $<"
	$(CC) $(CC_ARGS) -c $< -o $@ $(GENDEPFLAGS) 

clean:
	echo "RM  $(OBJS)"
	rm -f $(OBJS)
	echo "  CC    $<"
	rm -f $(TARGET)
	rm -f .deps
	rm -f .dep/*.d

-include $(shell mkdir .dep 2>/dev/null) $(wildcard .dep/*)
