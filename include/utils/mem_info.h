#ifndef MEM_INFO_H
#define MEM_INFO_H

#include <stdlib.h>
#include <stdio.h>

#define MEMORY_USAGE_RATIO 0.5

/**
 * Estimate the amount of usable memory available on the system (Linux specific).
 * 
 * @return  Size of usable memory in bytes.
 */
size_t get_usable_memory(void);

#endif // MEM_INFO_H