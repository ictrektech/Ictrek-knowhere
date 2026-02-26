FROM docker.m.daocloud.io/ubuntu:22.04

# 声明架构变量
ARG TARGETARCH
ARG TARGETVARIANT

# 声明代理构建参数
ARG http_proxy=""
ARG https_proxy="http://192.168.1.86:7890"


RUN sed -i 's@http://archive.ubuntu.com@http://mirrors.aliyun.com@g' /etc/apt/sources.list && \
    sed -i 's@http://ports.ubuntu.com@http://mirrors.aliyun.com@g' /etc/apt/sources.list && \
    sed -i 's@http://security.ubuntu.com@http://mirrors.aliyun.com@g' /etc/apt/sources.list

RUN apt update
RUN apt install -y software-properties-common wget
RUN add-apt-repository -y ppa:git-core/ppa
RUN apt update

# 根据架构选择数学库
RUN if [ "$(uname -m)" = "aarch64" ] || [ "$TARGETARCH" = "arm64" ]; then \
        echo "检测到ARM64架构，安装OpenBLAS..." && \
        DEBIAN_FRONTEND=noninteractive apt install -y \
            git make cmake g++ libaio-dev libgoogle-perftools-dev libunwind-dev clang-format \
            libboost-dev libboost-program-options-dev libcpprest-dev python3.12 libboost-all-dev \
            libjemalloc-dev libopenblas-dev libopenblas-openmp-dev && \
        cd /opt && \
        wget https://developer.arm.com/-/cdn-downloads/permalink/Arm-Performance-Libraries/Version_25.07/arm-performance-libraries_25.07_deb_gcc.tar && \
        tar xf arm-performance-libraries_25.07_deb_gcc.tar && \
        ./arm-performance-libraries_25.07_deb/arm-performance-libraries_25.07_deb.sh --accept && \
        rm -rf arm-performance-libraries_25.07_deb_gcc.tar arm-performance-libraries_25.07_deb; \
    elif [ "$(uname -m)" = "x86_64" ] || [ "$TARGETARCH" = "amd64" ]; then \
        echo "检测到x86_64架构，安装Intel MKL..." && \
        # 对于x86架构，添加Intel MKL仓库
        apt install -y gnupg && \
        wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
        echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
        apt update && \
        DEBIAN_FRONTEND=noninteractive apt install -y \
            git make cmake g++ libaio-dev libgoogle-perftools-dev libunwind-dev clang-format \
            libboost-dev libboost-program-options-dev libcpprest-dev python3.12 libboost-all-dev \
            libjemalloc-dev intel-oneapi-mkl-devel; \
    else \
        echo "未知架构: $(uname -m), 安装OpenBLAS作为默认..." && \
        DEBIAN_FRONTEND=noninteractive apt install -y \
            git make cmake g++ libaio-dev libgoogle-perftools-dev libunwind-dev clang-format \
            libboost-dev libboost-program-options-dev libcpprest-dev python3.12 libboost-all-dev \
            libjemalloc-dev libopenblas-dev libopenblas-openmp-dev; \
    fi


WORKDIR /root/
COPY . /root/PipeANN/
WORKDIR /root/PipeANN/

RUN ./build.sh
