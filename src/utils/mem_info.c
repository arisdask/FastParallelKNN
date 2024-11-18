#include "../../include/utils/mem_info.h"

size_t get_usable_memory() {
    FILE *file = fopen("/proc/meminfo", "r");
    if (file == NULL) {
        fprintf(stderr, "get_usable_memory: Could not open /proc/meminfo");
        return 0;
    }

    char line[256];
    size_t mem_available = 0;
    
    while (fgets(line, sizeof(line), file)) {
        if (sscanf(line, "MemAvailable: %zu kB", &mem_available) == 1) {
            break;
        }
    }
    
    fclose(file);

    // [result]*1024 to convert from kB to bytes
    return MEMORY_USAGE_RATIO * mem_available * 1024;
}
