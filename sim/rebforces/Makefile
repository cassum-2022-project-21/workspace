INCLUDES = -isystem ../../src/rebound/src
# CFLAGS = -g -std=c99 -fPIC -O0 -Wall $(INCLUDES)
CFLAGS   = -g -std=c99 -fPIC -O3 -march=native -fno-math-errno -freciprocal-math -fno-trapping-math -fno-signaling-nans -Wpedantic $(INCLUDES)
# CFLAGS = -g -std=c99 -fPIC -Wall $(INCLUDES)

LDFLAGS = -shared -L../../src/rebound
LDLIBS = -lrebound

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

