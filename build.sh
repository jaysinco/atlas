#!/bin/bash

set -e

# flags

do_clean=0
do_preprocess=0
do_build_release=0
do_build_nil=0
do_build_core=0
do_build_gui=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -h)
            echo
            echo "Usage: `basename "$0"` [options] [targets]"
            echo
            echo "Build Options:"
            echo "  -c      clean build"
            echo "  -p      preprocess code"
            echo "  -r      build release version (default: debug)"
            echo "  -h      print command line options"
            echo
            echo "Build Targets:"
            echo "  nil     empty target"
            echo "  core    c++ backend"
            echo "  gui     graphical ui"
            echo
            exit 0
            ;;
        -c) do_clean=1 && shift ;;
        -p) do_preprocess=1 && shift ;;
        -r) do_build_release=1 && shift ;;
       nil) do_build_nil=1 && shift ;;
      core) do_build_core=1 && shift ;;
       gui) do_build_gui=1 && shift ;;
         *) echo "Unknown option: $1" && exit 1 ;;
    esac
done

if [ $do_build_nil -eq 0 -a $do_build_core -eq 0 -a $do_build_gui -eq 0 ]; then
    do_build_core=1
    do_build_gui=1
fi

# build

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
git_root="$(git rev-parse --show-toplevel)"

case "$OSTYPE" in
    linux*)   os=linux ;;
    msys*)    os=windows ;;
esac

case `arch` in
    x86_64)   arch=x64 ;;
    aarch64)  arch=aarch64 ;;
esac

export PYTHONPATH=$git_root/deps

build_type=Debug
if [ $do_build_release -eq 1 ]; then
    build_type=Release
fi

conan_profile=$git_root/configs/conan.$os.$arch.profile
source_folder=$git_root/src
build_folder=$git_root/out
binary_folder=$git_root/bin

core_source_folder=$source_folder/core
core_build_folder=$build_folder/core/${build_type,,}

function clean_build() {
    rm -rf $build_folder
    rm -rf $binary_folder
}

function preprocess_code() {
    find $core_source_folder -iname *.h -or -iname *.cpp | xargs clang-format -i \
    && find $core_source_folder -iname *.h -or -iname *.cpp | xargs \
        clang-tidy --quiet --warnings-as-errors="*" -p $core_build_folder
}

function build_core() {
    conan install $git_root \
        --install-folder=$core_build_folder \
        --profile=$conan_profile \
        --profile:build=$conan_profile \
        --settings=build_type=$build_type \
        --build=never \
    && \
    conan build $git_root \
        --install-folder=$core_build_folder
}

function build_flutter() {
    pro_name=$1
    pro_source_folder=$source_folder/$pro_name
    pro_build_folder_abs=$build_folder/$pro_name
    mkdir -p $pro_build_folder_abs
    pro_build_folder=`realpath $pro_build_folder_abs --relative-to=$pro_source_folder`

    pushd $pro_source_folder \
    && \
    flutter pub get \
    && \
    flutter config --build-dir=$pro_build_folder \
    && \
    flutter build $os --${build_type,,}
}

function build_gui() {
    build_flutter hello_flutter
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
    build_core
fi \
&& \
if [ $do_build_gui -eq 1 ]; then
    build_gui
fi \
&& \
echo done!
