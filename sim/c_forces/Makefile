INCLUDES = -I../../src/rebound/src
CFLAGS   = -std=c99 -fPIC -O3 -march=native -fno-math-errno -freciprocal-math -fno-trapping-math -fno-signaling-nans -Wall $(INCLUDES)

LDFLAGS = -shared	
# LDLIBS = -lrebound

BUILD_DIR := ./build

SRCS := $(wildcard *.c)
OBJS := $(SRCS:%.c=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

librebcforces.so: $(OBJS)
	$(CC) $(CFLAGS) -shared $(OBJS) $(LDFLAGS) -o $@ $(LDLIBS)

-include $(DEPS) $(TESTDEPS)

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -c -o $@ $<

.PHONY: clean
clean:
	rm -rf *.so $(BUILD_DIR)

