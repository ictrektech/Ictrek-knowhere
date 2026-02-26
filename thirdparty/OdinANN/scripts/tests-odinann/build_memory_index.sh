# !/bin/bash

TEST_DIR="/home/ictrek/workspace-docker/xuejin/dataset/bigann/test"
if [ "$#" -gt 0 ]; then
    TEST_DIR=$1
fi

SAMPLE_RATE=0.1
MAX_DEGREE=96 # 控制 Vamana 图中节点的最大连接数
SEARCH_SIZE=128 # 构建阶段的候选列表大小（值越大，图质量越高，但构建时间延长）
ALPHA=1.2
INDEX_BUILD_THREADS=$(nproc)
DISTANCE_METRIC="l2"
SPLIT_INDEX_FILE=0
#dynamic index against static index, static meaning just read after load&build.
DYNAMIC_INDEX=0

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
${test_exe_path}/utils/gen_random_slice ${DATA_TYPE} ${DATA_FILE} ${INDEX_PREFIX}_SAMPLE_RATE_${SAMPLE_RATE} ${SAMPLE_RATE}
${test_exe_path}/build_memory_index ${DATA_TYPE} ${INDEX_PREFIX}_SAMPLE_RATE_${SAMPLE_RATE}_data.bin ${INDEX_PREFIX}_SAMPLE_RATE_${SAMPLE_RATE}_ids.bin ${INDEX_PREFIX}_mem.index ${DYNAMIC_INDEX} ${SPLIT_INDEX_FILE} ${MAX_DEGREE} ${SEARCH_SIZE} ${ALPHA} ${INDEX_BUILD_THREADS} ${DISTANCE_METRIC}