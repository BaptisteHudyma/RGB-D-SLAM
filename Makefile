SRC_DIR=$(dir $(realpath $(firstword $(MAKEFILE_LIST))))
BUILD_DIR=$(SRC_DIR)build

FOLDER_NAME=$(shell basename $(SRC_DIR))
PROJECT_NAME=RGB-D-SLAM

DATA_SOURCE=$(SRC_DIR)data
DATA_CAPE_SOURCE=$(DATA_SOURCE)CAPE
DATA_TUM_SOURCE=$(DATA_SOURCE)TUM

SRC_FOLDER=$(SRC_DIR)src

TEST_SOURCES := $(basename $(shell find $(SRC_DIR) -name 'test_*.cpp'))
TEST_EXEC := $(shell basename -a $(TEST_SOURCES))

build-code:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake $(SRC_DIR)
	cd $(BUILD_DIR) && make -j
	@echo " --- ok: $*"

clean:
	rm -rf $(BUILD_DIR)/*

all: build-code

test_%:
	$(BUILD_DIR)/$@
	@echo "run test $@"

run-test : build-code $(TEST_EXEC)
	@echo "--- All tests ok: $@"
	
run-tum:
	$(BUILD_DIR)/slam_TUM $(ARGS)

run-cape:
	$(BUILD_DIR)/slam_CAPE $(ARGS)


format-hook:
	cp .pre-commit .git/hooks/pre-commit

format-verify:
	@which clang-format > /dev/null \
		|| (echo; echo Install clang / clang-format to verify format!)
	@find $(SRC_FOLDER) -iname '*.h' -o -iname '*.cpp' -o -iname '*.hpp' -o -iname '*.c' | xargs clang-format --style=file --dry-run -Werror
	@if ! grep -IUr "$$(printf '\r')" src; then true; else echo 'You are using CRLF (\\r\\n) in a POSIX project :('; false; fi
	# format is ok :)

format:
	find $(SRC_FOLDER) -iname '*.h' -o -iname '*.cpp' -o -iname '*.hpp' -o -iname '*.c' | xargs clang-format --style=file -i
	@if ! grep -IUr "$$(printf '\r')" src; then true; else echo 'You are using CRLF (\\r\\n) in a POSIX project :('; false; fi
	# format is ok :)
