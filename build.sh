#!/bin/bash

set -e

# flags

do_clean=0
do_arch=`uname -m`
do_build_debug=0
do_build_driver=0
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
            echo "  -a ARCH    target arch, default '$do_arch'"
            echo "  -d         build debug version"
            echo "  -k         build linux device driver"
            echo "  -p         preprocess code before build"
            echo "  -z         zip binary after build"
            echo "  -h         print command line options"
            echo
            exit 0
            ;;
        -c) do_clean=1 && shift ;;
        -a) do_arch=$2 && shift && shift ;;
        -d) do_build_debug=1 && shift ;;
        -k) do_build_driver=1 && shift ;;
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

arch=`uname -m`
case "$OSTYPE" in
    linux*)   os=linux ;;
    msys*)    os=windows ;;
esac

source_folder=$git_root/src
tuple_name=$do_arch-${build_type,,}
build_folder=$git_root/out/$tuple_name
binary_folder=$git_root/bin/$tuple_name
log_folder=$binary_folder/logs
temp_folder=$binary_folder/temp
tc_toolchain_dir=$git_root/../cpptools/toolchain/$os/$arch/$do_arch
driver_src_folder=$source_folder/ldd

source $tc_toolchain_dir/env.sh

function clean_build() {
    rm -rf $build_folder
    rm -rf $binary_folder
}

function preprocess_code() {
    find $source_folder -iname *.h -or -iname *.cpp | xargs clang-format -i \
    && find $source_folder -iname *.h -or -iname *.cpp | xargs clang-tidy \
        --quiet --warnings-as-errors="*" -p $build_folder
}

function cmake_build() {
    if [ $TC_CROSS_COMPILE -eq 1 ]; then
        tc_opt="-DCMAKE_TOOLCHAIN_FILE=$TC_CMAKE_TOOLCHAIN"
    else
        tc_compiler_c=$([ "$os" == "linux" ] && echo "gcc" || echo "cl")
        tc_compiler_cxx=$([ "$os" == "linux" ] && echo "g++" || echo "cl")
        tc_opt="-DCMAKE_C_COMPILER=$tc_compiler_c -DCMAKE_CXX_COMPILER=$tc_compiler_cxx"
    fi \
    && \
    mkdir -p \
        $build_folder \
        $log_folder \
        $temp_folder \
    && \
    pushd $build_folder \
    && \
    cmake $git_root -G "Ninja" \
        -DCMAKE_BUILD_TYPE=$build_type \
        -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=$binary_folder \
        $tc_opt \
        -DTC_CROSS_COMPILE=$TC_CROSS_COMPILE \
        -DTC_INSTALL_DIR=$TC_INSTALL_DIR \
        -DTC_THIRDPARTY=$TC_THIRDPARTY \
    && \
    cp $build_folder/compile_commands.json $build_folder/.. \
    && \
    cmake --build . --parallel=`nproc`
}

function linux_driver_build() {
    if [ "$os" == "linux" -a $TC_CROSS_COMPILE -eq 0 ]; then
        bear \
            --output $build_folder/../compile_commands.json \
            --append \
            -- \
            make -C $driver_src_folder \
        && \
        mv $driver_src_folder/*.ko $binary_folder/
    fi
}

function zip_binary() {
    if [ "$os" == "linux" ]; then
        tar -czf \
            $git_root/bin/$tuple_name.tar.gz \
            -C $git_root/bin \
            $tuple_name
    else
        pushd $git_root/bin
        zip -rq \
            $git_root/bin/$tuple_name.zip \
            $tuple_name
        popd
    fi
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
if [ $do_build_driver -eq 1 ]; then
    linux_driver_build
fi \
&& \
if [ $do_zip -eq 1 ]; then
    zip_binary
fi \
&& \
echo done!
