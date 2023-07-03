#!/bin/bash

set -e

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
git_root="$(git rev-parse --show-toplevel)"

function package() {
    local build_debug=$1
    local name=$2
    $git_root/recipes/build.sh $name -r && \
    if [ $build_debug -eq 1 ]; then
        $git_root/recipes/build.sh $name -r -d
    fi
}

echo start! \
&& package 1 fmt \
&& package 1 spdlog \
&& package 0 boost \
&& package 1 catch2 \
&& package 0 expected-lite \
&& package 1 zlib \
&& package 0 openssl \
&& package 0 range-v3 \
&& package 1 libiconv \
&& package 0 nlohmann-json \
&& package 1 libuv \
&& package 1 usockets \
&& package 0 uwebsockets \
&& package 0 concurrent-queue \
&& package 0 threadpool \
&& echo done!
