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

case "$OSTYPE" in
    linux*)   os=linux ;;
    msys*)    os=windows ;;
esac

py_ver=3.13.2
py_exe=$([ "$os" == "linux" ] && echo "~/.pyenv/versions/$py_ver/bin/python" || echo "python")
venv_name=.pyenv
venv_dir=$script_dir/$venv_name
venv_script_name=$([ "$os" == "linux" ] && echo "bin" || echo "Scripts")
venv_script_dir=$venv_dir/$venv_script_name
torch_type=$([ "$os" == "linux" ] && echo "cpu" || echo "cu118")

pushd $script_dir \
&& \
if [ $do_clean -eq 1 ]; then
    rm -rf $venv_dir
fi \
&& \
if [ ! -d "$venv_dir" ]; then
    $py_exe -m venv $venv_name \
    && \
    source $venv_script_dir/activate \
    && \
    pip3 install \
        -i https://download.pytorch.org/whl/$torch_type \
        torch==2.6 \
        torchvision==0.21 \
        torchaudio==2.6
else
    source $venv_script_dir/activate
fi \
&& \
pip3 install \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r $script_dir/pydeps.txt \
&& \
if [ "$os" == "linux" ]; then
    /usr/bin/fish -C "source $venv_script_dir/activate.fish"
else
    /usr/bin/bash --init-file <(echo ". \"$HOME/.bashrc\"; VIRTUAL_ENV_DISABLE_PROMPT=1; source \"$venv_script_dir/activate\"")
fi
