#!/bin/bash

set -e

# flags

do_clean=0
do_arch=`uname -m`
do_build_debug=0
do_preprocess=0
do_zip=0

do_build_none=0
do_build_core=0
do_build_ldd=0
do_build_flapp=0

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
            echo "  -p         preprocess code before build"
            echo "  -z         zip binary after build"
            echo "  -h         print command line options"
            echo
            echo "Build Targets:"
            echo "  none       empty target"
            echo "  core       cpp backend"
            echo "  ldd        linux device driver"
            echo "  flapp      flutter mock app"
            echo
            exit 0
            ;;
        -c) do_clean=1 && shift ;;
        -a) do_arch=$2 && shift && shift ;;
        -d) do_build_debug=1 && shift ;;
        -p) do_preprocess=1 && shift ;;
        -z) do_zip=1 && shift ;;
      none) do_build_none=1 && shift ;;
      core) do_build_core=1 && shift ;;
       ldd) do_build_ldd=1 && shift ;;
     flapp) do_build_flapp=1 && shift ;;
         *) echo "unknown argument: $1" && exit 1 ;;
    esac
done

if [ $do_build_none -eq 0 -a $do_build_core -eq 0 -a $do_build_ldd -eq 0 -a $do_build_flapp -eq 0 ]; then
    do_build_core=1
    if [[ $(type -P "flutter") ]]; then
        do_build_flapp=0
    fi
fi

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
ldd_src_folder=$source_folder/ldd
flapp_src_folder=$source_folder/flapp

source $tc_toolchain_dir/env.sh

function clean_build() {
    rm -rf $git_root/out
    rm -rf $git_root/bin
    rm -rf $flapp_src_folder/build
    rm -rf $flapp_src_folder/.idea
    rm -rf $flapp_src_folder/*.iml
    if [ $do_build_flapp -eq 1 ]; then
        rm -rf $flapp_src_folder/.dart_tool
    fi
}

function preprocess_code() {
    find $source_folder -iname *.h -or -iname *.cpp | xargs clang-format -i \
    && find $source_folder -iname *.h -or -iname *.cpp | xargs clang-tidy \
        --quiet --warnings-as-errors="*" -p $build_folder
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
    cmake $git_root -G "Ninja" -Wno-dev \
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
            make -C $ldd_src_folder \
        && \
        mv $ldd_src_folder/*.ko $binary_folder/
    fi
}

function flutter_build() {
    if [ $TC_CROSS_COMPILE -eq 0 ]; then
        export PUB_HOSTED_URL=https://pub.flutter-io.cn
        export FLUTTER_STORAGE_BASE_URL=https://storage.flutter-io.cn
        bundle_dir=$([ "$os" == "linux" ] && \
            echo "$flapp_src_folder/build/$os/x64/${build_type,,}/bundle" || \
            echo "$flapp_src_folder/build/$os/x64/runner/$build_type")
        fl_opt=$([ "$os" == "linux" ] && echo "--target-platform=linux-x64" || echo "")

        if [ ! -d $flapp_src_folder ]; then
            flutter --no-version-check create \
                --template=app \
                --platforms=linux,windows \
                --project-name=flapp \
                $flapp_src_folder
        fi \
        && \
        pushd $flapp_src_folder \
        && \
        flutter --no-version-check pub get \
        && \
        flutter --no-version-check build $os \
            --${build_type,,} \
            --no-pub \
            $fl_opt \
        && \
        rsync -r $bundle_dir/* $binary_folder \
        && \
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
if [ $do_build_core -eq 1 ]; then
    cmake_build
fi \
&& \
if [ $do_build_ldd -eq 1 ]; then
    linux_driver_build
fi \
&& \
if [ $do_build_flapp -eq 1 ]; then
    flutter_build
fi \
&& \
if [ $do_zip -eq 1 ]; then
    zip_binary
fi \
&& \
echo done!
