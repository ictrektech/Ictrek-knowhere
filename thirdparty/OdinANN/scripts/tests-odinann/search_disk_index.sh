# !/bin/bash

TEST_DIR="/home/ictrek/workspace-docker/xuejin/dataset/bigann/test"
if [ "$#" -gt 0 ]; then
    TEST_DIR=$1
fi

SEARCH_THREADS=$(nproc)
CONCCURENT_REQS=32 # beam width
TOPK=100 # 10 or 100
DISTANCE_METRIC="l2" # "l2" or "cosine"
# 0: beam search; 1: page search; 2: pipe search
SEARCH_MODE=2
# 内存图搜索过程中同时保留的最多候选节点数，等于0则不使用内存图
L_MEMORY_INDEX=0
# 磁盘图搜索过程中同时保留的最多候选节点数，可设置多个值作为一组测试，需大于等于TOPK
L_ONDISK_INDEX="100 150 200" # "50 100 150" for topk==10, "100 150 200" for topk==100

if [[ "$TEST_DIR" == *"bigann"* ]]; then
    # bigann dataset parameters
    DATA_TYPE="uint8"
    DATA_FILE="${TEST_DIR}/bigann_learn.bin"
    QUERY_FILE="${TEST_DIR}/bigann_query.bin"
    INDEX_PREFIX="${TEST_DIR}/100m"
    TRUTH_FILE="${TEST_DIR}/100m_gt${TOPK}.bin"
elif [[ "$TEST_DIR" == *"gist"* ]]; then
    # gist dataset parameters
    DATA_TYPE="float"
    DATA_FILE="${TEST_DIR}/gist_learn.bin"
    QUERY_FILE="${TEST_DIR}/gist_query.bin"
    INDEX_PREFIX="${TEST_DIR}/1m"
    TRUTH_FILE="${TEST_DIR}/gist_gt${TOPK}.bin"
elif [[ "$TEST_DIR" == *"cohere"* ]]; then
    # cohere dataset parameters
    DATA_TYPE="float"
    DATA_FILE="${TEST_DIR}/cohere_learn.bin"
    QUERY_FILE="${TEST_DIR}/cohere_query.bin"
    INDEX_PREFIX="${TEST_DIR}/1m"
    TRUTH_FILE="${TEST_DIR}/cohere_gt${TOPK}.bin"
fi

test_exe_path="/home/ictrek/workspace-docker/xuejin/OdinANN/build/tests"

set -x
if [ ! -f ${TRUTH_FILE} ]; then
    ${test_exe_path}/utils/compute_groundtruth ${DATA_TYPE} ${DATA_FILE} ${QUERY_FILE} ${TOPK} ${TRUTH_FILE}
fi
${test_exe_path}/search_disk_index ${DATA_TYPE} ${INDEX_PREFIX} ${SEARCH_THREADS} ${CONCCURENT_REQS} ${QUERY_FILE} ${TRUTH_FILE} ${TOPK} ${DISTANCE_METRIC} ${SEARCH_MODE} ${L_MEMORY_INDEX} ${L_ONDISK_INDEX}