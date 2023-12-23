#!/bin/bash

set -e

# flags

do_clean=0
do_arch=x86_64
do_build_debug=0
do_preprocess=0
do_zip=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -h)
            echo
            echo "Usage: `basename "$0"` [options]"
            echo
            echo "Build Options:"
            echo "  -c         clean before build"
            echo "  -a ARCH    target arch, default 'x86_64'"
            echo "  -d         build debug version"
            echo "  -p         preprocess code before build"
            echo "  -z         zip binary after build"
            echo "  -h         print command line options"
            echo
            exit 0
            ;;
        -c) do_clean=1 && shift ;;
        -a) do_arch=$2 && shift && shift ;;
        -d) do_build_debug=1 && shift ;;
        -p) do_preprocess=1 && shift ;;
        -z) do_zip=1 && shift ;;
         *) echo "unknown argument: $1" && exit 1 ;;
    esac
done

# build

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
git_root="$(git rev-parse --show-toplevel)"

build_type=Release
if [ $do_build_debug -eq 1 ]; then
    build_type=Debug
fi

arch=`arch`
source_folder=$git_root/src
tuple_name=$do_arch-${build_type,,}
build_folder=$git_root/out/$tuple_name
binary_folder=$git_root/bin/$tuple_name
tc_toolchain_dir=$git_root/../cpptools/toolchain/$do_arch

source $tc_toolchain_dir/env.sh
export PATH="$PATH:$TC_ROOT_DIR/toolchain/$arch/$arch-unknown-linux-gnu/$arch-unknown-linux-gnu/sysroot/usr/bin"

function clean_build() {
    rm -rf $git_root/out
    rm -rf $git_root/bin
}

function preprocess_code() {
    find $source_folder -iname *.h -or -iname *.cpp | xargs clang-format -i \
    && find $source_folder -iname *.h -or -iname *.cpp | xargs clang-tidy \
        --quiet --warnings-as-errors="*" -p $build_folder
}

function cmake_build() {
    mkdir -p $build_folder \
    && \
    pushd $build_folder \
    && \
    cmake $git_root -G "Ninja" \
        -DCMAKE_TOOLCHAIN_FILE=$TC_CMAKE_TOOLCHAIN \
        -DCMAKE_BUILD_TYPE=$build_type \
        -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$binary_folder \
    && \
    cp $build_folder/compile_commands.json $build_folder/.. \
    && \
    cmake --build . --parallel=`nproc`
}

function zip_binary() {
    tar -czf \
        $git_root/bin/$tuple_name.tar.gz \
        -C $git_root/bin \
        $tuple_name
}


if [ $do_clean -eq 1 ]; then
    clean_build
fi \
&& \
if [ $do_preprocess -eq 1 ]; then
    preprocess_code
fi \
&& \
cmake_build \
&& \
if [ $do_zip -eq 1 ]; then
    zip_binary
fi \
&& \
echo done!
