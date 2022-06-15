INCLUDES = -I../../src/rebound/src
CFLAGS   = -fPIC -O3 -Wall $(INCLUDES)

LDFLAGS = -shared	
# LDLIBS = -lrebound

BUILD_DIR := ./build

.PHONY: default
default: $(BUILD_DIR)/librebcforces.so


SRCS := $(wildcard *.c)
OBJS := $(SRCS:%.c=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

-include $(DEPS) $(TESTDEPS)

$(BUILD_DIR)/%.o: %.c
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -c -o $@ $<

$(BUILD_DIR)/librebcforces.so: $(OBJS)
	$(CC) $(CFLAGS) -shared $(OBJS) $(LDFLAGS) -o $@ $(LDLIBS)

.PHONY: clean
clean:
	rm -f $(BUILD_DIR)

