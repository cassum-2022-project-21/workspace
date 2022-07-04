INCLUDES = -I../../../src/rebound/src
CFLAGS   = -fno-PIE -g -O3 -std=c99 -march=native -fno-math-errno -freciprocal-math -fno-trapping-math -fno-signaling-nans -Wall $(INCLUDES)

LDFLAGS = -fno-PIE -L.. -L../../../src/rebound
LDLIBS = -lrebound -lrebcforces -lm

.PHONY: clean
clean:
	rm -rf *.o test test-keplerian-no-drag test-scaled-keplerian-drag
