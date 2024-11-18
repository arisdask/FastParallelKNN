# Compiler and Flags
CC = gcc
CFLAGS = -Wall -O3 -g
LDFLAGS = -lopenblas -lgsl -lgslcblas -lm
# -lopenblas: for OpenBLAS library, install with: sudo apt-get install libopenblas-dev
# -lgsl:      for GNU Scientific Library (GSL), install with: sudo apt-get install libgsl-dev
# -lgslcblas: for CBLAS interface for GSL, install the -lgsl above
# -lm: 		  for <math.h>, provided by the GNU C Library

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build

# Create a list of source files
EXACT_SRC = $(SRC_DIR)/exact/knn_exact_serial.c
UTILS_SRC = $(SRC_DIR)/utils/data_io.c $(SRC_DIR)/utils/distance.c $(SRC_DIR)/utils/mem_info.c
MAIN_SRC = $(SRC_DIR)/main.c
SRC = $(EXACT_SRC) $(UTILS_SRC) $(MAIN_SRC)

# Create a list of object files (matching the source file structure)
EXACT_OBJ = $(BUILD_DIR)/exact/knn_exact_serial.o
UTILS_OBJ = $(BUILD_DIR)/utils/data_io.o $(BUILD_DIR)/utils/distance.o $(BUILD_DIR)/utils/mem_info.o
MAIN_OBJ = $(BUILD_DIR)/main.o
OBJ = $(EXACT_OBJ) $(UTILS_OBJ) $(MAIN_OBJ)

# Output executable
EXEC = knn_project

# Include directories
INCLUDES = -I$(INCLUDE_DIR)/exact -I$(INCLUDE_DIR)/utils

# Default target to build the project
all: $(EXEC)

# Rule to build the final executable
$(EXEC): $(OBJ)
	@echo "Linking object files to create executable: $(EXEC)"
	$(CC) -o $@ $^ $(LDFLAGS) $$(pkg-config --cflags --libs hdf5)
# $(pkg-config --cflags --libs hdf5) --> it's in case adding -lhdf5 in LDFLAGS does not work

# Compile exact .c files into .o files
$(BUILD_DIR)/exact/%.o: $(SRC_DIR)/exact/%.c | $(BUILD_DIR)/exact
	@echo "Compiling $<"
	$(CC) $(CFLAGS) -c $< -o $@ $(INCLUDES)

# Compile utils .c files into .o files
$(BUILD_DIR)/utils/%.o: $(SRC_DIR)/utils/%.c | $(BUILD_DIR)/utils
	@echo "Compiling $<"
	$(CC) $(CFLAGS) -c $< -o $@ $(INCLUDES)

# Compile main.c into main.o
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.c | $(BUILD_DIR)
	@echo "Compiling $<"
	$(CC) $(CFLAGS) -c $< -o $@ $(INCLUDES)

# Create necessary directories
$(BUILD_DIR)/exact:
	@mkdir -p $@
	@echo "Created directory: $@"

$(BUILD_DIR)/utils:
	@mkdir -p $@
	@echo "Created directory: $@"

$(BUILD_DIR):
	@mkdir -p $@
	@echo "Created directory: $@"

# Clean rule to remove build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR) $(EXEC)

# Phony targets (these don't correspond to real files)
.PHONY: all clean run
