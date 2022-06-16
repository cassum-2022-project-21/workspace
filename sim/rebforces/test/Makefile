INCLUDES = -I../../../src/rebound/src
CFLAGS   = -g -Wall $(INCLUDES)

LDFLAGS = -L.. -L../../../src/rebound
LDLIBS = -lrebound -lrebcforces

test: test.c

.PHONY: clean
clean:
	rm -rf *.o test
