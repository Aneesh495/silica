.PHONY: all build test bench clean check-dashes loc

BUILD_DIR ?= build

all: build

build:
	@cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	@cmake --build $(BUILD_DIR) --parallel

test: build
	@cd $(BUILD_DIR) && ctest --output-on-failure || ./$(BUILD_DIR)/silica_tests

bench: build
	@./$(BUILD_DIR)/silica_bench

check-dashes:
	@python3 scripts/verify_dashes.py

loc:
	@python3 scripts/count_loc.py

clean:
	@rm -rf $(BUILD_DIR)
