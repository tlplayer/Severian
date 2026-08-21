#include <stdio.h>
#include <stdlib.h>

void __sev_coverage_hit(const char *key) {
    const char *path = getenv("SEV_COVERAGE_FILE");
    if (path == NULL) return;
    FILE *file = fopen(path, "a");
    if (file == NULL) return;
    fputs(key, file);
    fputc('\n', file);
    fclose(file);
}
