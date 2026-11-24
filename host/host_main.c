/* Runs bench_predict() on every row of a float32 test file and prints one label per line. */
#include <stdio.h>
#include <stdlib.h>

#include "model_info.h"

int bench_predict(const float *x);

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s X_test.bin\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        perror(argv[1]);
        return 1;
    }
    float x[BENCH_N_FEATURES];
    while (fread(x, sizeof(float), BENCH_N_FEATURES, f) == BENCH_N_FEATURES) {
        printf("%d\n", bench_predict(x));
    }
    fclose(f);
    return 0;
}
