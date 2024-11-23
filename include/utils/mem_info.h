#ifndef MEM_INFO_H
#define MEM_INFO_H

#include <stdlib.h>
#include <stdio.h>

// Margin scalar of the available memory for safety.
#define MEMORY_USAGE_RATIO 0.3

// - If USE_CONST_MEMORY == 0: `get_usable_memory` returns the usable memory
// based on the available memory value inside the /proc/meminfo file in linux (Ubuntu distr).
// run this in shell: `cat /proc/meminfo` , to open this file. 
// - If USE_CONST_MEMORY == 1: `get_usable_memory` always returns a constant prediction of the usable
// memory, `USABLE_MEM_PREDICTION`, based on the computer in which the project will run.
#define USE_CONST_MEMORY 1

// - 3000000kB = 3GB is a safe but still efficient idea when you run the approximate parallel functions (for no more than 8 threads), 
//   if your system has `16GB` of total RAM.
// - In case the program crashes try to reduce `USABLE_MEM_PREDICTION` to a value based on your machines specs.
// - The value should be in `kB`
#define USABLE_MEM_PREDICTION 3000000


/**
 * Estimate the amount of usable memory available on the system.
 * 
 * @return  Size of usable memory in bytes.
 */
size_t get_usable_memory(void);

#endif // MEM_INFO_H