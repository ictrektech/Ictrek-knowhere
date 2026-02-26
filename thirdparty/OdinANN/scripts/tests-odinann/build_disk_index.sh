# !/bin/bash

TEST_DIR="/home/ictrek/workspace-docker/xuejin/dataset/bigann/test"
if [ "$#" -gt 0 ]; then
    TEST_DIR=$1
fi

MAX_DEGREE=96 # 控制 Vamana 图中节点的最大连接数
SEARCH_SIZE=128 # 构建阶段的候选列表大小（值越大，图质量越高，但构建时间延长）
MAX_MEMORY_LOAD_INDEX=16
MAX_MEMORY_BUILD_INDEX=30
INDEX_BUILD_THREADS=$(nproc)
DISTANCE_METRIC="l2"
SPLIT_INDEX_FILE=0

if [[ "$TEST_DIR" == *"bigann"* ]]; then
    # bigann dataset parameters
    DATA_TYPE="uint8"
    DATA_FILE="${TEST_DIR}/bigann_learn.bin"
    INDEX_PREFIX="${TEST_DIR}/100m"
elif [[ "$TEST_DIR" == *"gist"* ]]; then
    # gist dataset parameters
    DATA_TYPE="float"
    DATA_FILE="${TEST_DIR}/gist_learn.bin"
    INDEX_PREFIX="${TEST_DIR}/1m"
elif [[ "$TEST_DIR" == *"cohere"* ]]; then
    # cohere dataset parameters
    DATA_TYPE="float"
    DATA_FILE="${TEST_DIR}/cohere_learn.bin"
    INDEX_PREFIX="${TEST_DIR}/1m"
fi

test_exe_path="/home/ictrek/workspace-docker/xuejin/OdinANN/build/tests"

set -x
${test_exe_path}/build_disk_index ${DATA_TYPE} ${DATA_FILE} ${INDEX_PREFIX} ${MAX_DEGREE} ${SEARCH_SIZE} ${MAX_MEMORY_LOAD_INDEX} ${MAX_MEMORY_BUILD_INDEX} ${INDEX_BUILD_THREADS} ${DISTANCE_METRIC} ${SPLIT_INDEX_FILE}