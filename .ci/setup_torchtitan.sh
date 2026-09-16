# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shared torchtitan setup for CI scripts.
# Usage: source .ci/setup_torchtitan.sh

TORCHTITAN_BRANCH="main"
TORCHTITAN_REPOSITORY="https://gitcode.com/GitHub_Trending/to/torchtitan.git"

_install_torchft() {
    if ! command -v protoc >/dev/null 2>&1 || [[ ! -f /usr/include/google/protobuf/timestamp.proto ]]; then
        DEBIAN_FRONTEND=noninteractive apt-get update -qq
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends protobuf-compiler libprotobuf-dev
    fi

    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local torchft_requirement
    torchft_requirement="$(python3 -c 'import sys, tomllib; req = tomllib.load(open(sys.argv[1], "rb"))["project"]["optional-dependencies"]["torchft"][0]; print(req.replace("https://github.com/meta-pytorch/torchft.git", "https://gitee.com/mirrors_pytorch/torchft.git"))' "${script_dir}/../pyproject.toml")"

    local cargo_home
    cargo_home="$(mktemp -d "${TMPDIR:-/tmp}/torchtitan-npu-cargo.XXXXXX")"
    cat >"${cargo_home}/config.toml" <<'EOF'
[source.crates-io]
replace-with = "ustc"
[source.ustc]
registry = "sparse+https://mirrors.ustc.edu.cn/crates.io-index/"
EOF
    # Maturin's Rust bootstrap uses its own Cargo home under XDG_CACHE_HOME.
    mkdir -p "${cargo_home}/puccinialin/cargo"
    cp "${cargo_home}/config.toml" "${cargo_home}/puccinialin/cargo/config.toml"

    RUSTUP_UPDATE_ROOT="https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup" \
        RUSTUP_DIST_SERVER="https://mirrors.tuna.tsinghua.edu.cn/rustup" \
        XDG_CACHE_HOME="${cargo_home}" \
        CARGO_HOME="${cargo_home}" \
        CARGO_REGISTRIES_CRATES_IO_INDEX="sparse+https://mirrors.ustc.edu.cn/crates.io-index/" \
        CARGO_NET_RETRY=2 CARGO_HTTP_TIMEOUT=30 \
        python3 -m pip install --verbose --no-deps "${torchft_requirement}"
    python3 -m pip install \
        'opentelemetry-api>=1.39.0' \
        'opentelemetry-sdk>=1.39.0' \
        'opentelemetry-exporter-otlp-proto-http>=1.39.0'
}

_setup_torchtitan() {
    local target_dir="${1:-/tmp/torchtitan}"
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local requirements_file="${script_dir}/../requirements.txt"
    local requirement_pattern='^[[:space:]]*torchtitan==[0-9][0-9A-Za-z.+-]*[[:space:]]*$'
    local torchtitan_requirement
    torchtitan_requirement="$(grep -E "${requirement_pattern}" "${requirements_file}")"
    local torchtitan_version="${torchtitan_requirement##*==}"
    torchtitan_version="${torchtitan_version//[[:space:]]/}"
    local torchtitan_commit="v${torchtitan_version}"

    echo "Preparing torchtitan at ${torchtitan_commit}..."

    echo "Cloning torchtitan source..."
    mkdir -p "$(dirname "$target_dir")"
    git clone --branch "$TORCHTITAN_BRANCH" \
        "$TORCHTITAN_REPOSITORY" "$target_dir"

    git -C "$target_dir" checkout "$torchtitan_commit"

    _install_torchft
}
