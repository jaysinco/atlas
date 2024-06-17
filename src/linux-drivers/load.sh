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

major=$(awk "\$2==\"$module\" {print \$1}" /proc/devices)
sudo rm -f /dev/${device}0
sudo mknod /dev/${device}0 c $major 0
sudo chmod $mode /dev/${device}0
