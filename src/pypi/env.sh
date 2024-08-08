#!/bin/bash

set -e

# flags

do_clean=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -h)
            echo
            echo "Usage: `basename "$0"` [options]"
            echo
            echo "Build Options:"
            echo "  -c         clean before setup env"
            echo "  -h         print command line options"
            echo
            exit 0
            ;;
        -c) do_clean=1 && shift ;;
         *) echo "unknown argument: $1" && exit 1 ;;
    esac
done

# env

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
git_root="$(git rev-parse --show-toplevel)"
venv_name=myenv
venv_path=$script_dir/$venv_name

case "$OSTYPE" in
    linux*)   os=linux ;;
    msys*)    os=windows ;;
esac

pushd $script_dir \
&& \
if [ $do_clean -eq 1 ]; then
    rm -rf $venv_path
fi \
&& \
if [ ! -d "$venv_path" ]; then
    torch_type=$([ "$os" == "linux" ] && echo "cpu" || echo "cu118")
    python -m venv $venv_name \
    && \
    source $venv_path/bin/activate \
    && \
    pip3 install \
        -i https://mirrors.aliyun.com/pypi/simple \
        -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html \
        torch==2.3.1+$torch_type \
        torchvision==0.18.1+$torch_type \
        torchaudio==2.3.1+$torch_type
else
    source $venv_path/bin/activate
fi \
&& \
pip3 install \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r $script_dir/requirements.txt \
&& \
if [ "$os" == "linux" ]; then
    /usr/bin/fish -C "source $venv_path/bin/activate.fish"
else
    /usr/bin/bash
fi
