#!/bin/bash

set -e

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
git_root="$(git rev-parse --show-toplevel)"

do_arch=`uname -m`
build_type=Release
tuple_name=$do_arch-${build_type,,}
binary_folder=$git_root/bin/$tuple_name

module="hello"
device="hello"
mode="666"

sudo insmod $binary_folder/$module.ko
sudo chmod $mode /dev/${device}0
