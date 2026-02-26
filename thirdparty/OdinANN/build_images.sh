#- build
PROXY="http://192.168.1.202:8889"


if [ -n "$PROXY" ]; then
    echo "使用代理构建: $PROXY"
    docker build \
        --build-arg http_proxy="${PROXY}" \
        --build-arg https_proxy="${PROXY}" \
        -t registry.interesting.com:80/interesting/pipeann:latest \
        -f Dockerfile . \
        --no-cache
else
    echo "不使用代理构建"
    docker build \
        -t registry.interesting.com:80/interesting/pipeann:latest \
        -f Dockerfile . \
        --no-cache
fi
