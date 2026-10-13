#!/usr/bin/env bash
# Milvus 镜像国内拉取备用脚本（Plan B）
# 用途：当 registry-mirrors 全部失效时，从华为云 SWR 拉取后改名为 compose 中声明的镜像名
# 用法：bash scripts/pull-mirror.sh
set -euo pipefail

MILVUS_TAG="v2.5.4"
MINIO_TAG="RELEASE.2024-12-18T13-15-44Z"

# 华为云 SWR 上的 docker.io 同步镜像前缀
SWR_PREFIX="swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io"

pull_and_tag() {
  local repo="$1"            # 例如 milvusdb/milvus
  local tag="$2"             # 例如 v2.5.4
  local target="${repo}:${tag}"
  local source="${SWR_PREFIX}/${repo}:${tag}"

  echo ">>> 拉取 ${source}"
  if docker pull "${source}"; then
    docker tag "${source}" "${target}"
    echo ">>> 已重命名为 ${target}"
  else
    echo "!!! 从 SWR 拉取失败，请手动确认该 tag 是否已同步：${source}"
    return 1
  fi
}

echo "=== Milvus 学习环境镜像拉取 ==="
pull_and_tag "milvusdb/milvus" "${MILVUS_TAG}"
pull_and_tag "minio/minio" "${MINIO_TAG}"

echo
echo "=== 完成，当前本地镜像 ==="
docker images | grep -E "milvus|minio" || true

echo
echo "下一步：docker compose up -d"
