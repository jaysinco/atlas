#!/bin/bash

set -e

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
git_root="$(git rev-parse --show-toplevel)"

case "$OSTYPE" in
    linux*)   os=linux ;;
    msys*)    os=windows ;;
esac

function pkga() {
    local build_debug=$1
    local name=$2
    $git_root/deps/build-recipe.sh $name -r && \
    if [ $build_debug -eq 1 ]; then
        $git_root/deps/build-recipe.sh $name -r -d
    fi
}

function pkgl() {
    if [ $os = "linux" ]; then
        pkga $*
    fi
}

function pkgw() {
    if [ $os = "windows" ]; then
        pkga $*
    fi
}

echo start! \
&& pkgw 0 jom \
&& pkgw 0 nasm \
&& pkgw 0 strawberryperl \
&& pkga 1 fmt \
&& pkga 1 spdlog \
&& pkga 0 boost \
&& pkga 1 catch2 \
&& pkga 0 expected-lite \
&& pkga 1 zlib \
&& pkga 0 openssl \
&& pkga 0 range-v3 \
&& pkga 1 libiconv \
&& pkga 0 nlohmann-json \
&& pkga 1 libuv \
&& pkga 1 usockets \
&& pkga 0 uwebsockets \
&& pkga 0 concurrent-queue \
&& pkga 0 threadpool \
&& echo done!
