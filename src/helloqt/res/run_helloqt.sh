#!/bin/bash

set -e

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

export QT_QPA_PLATFORM=eglfs
export QT_QPA_EGLFS_FB=/dev/fb0
export QT_QPA_EGLFS_INTEGRATION=none
# export QT_QPA_EGLFS_INTEGRATION=eglfs_x11
export QT_QPA_EGLFS_KMS_CONFIG=/home/odroid/gbm.json
export QT_QPA_EGLFS_DEBUG=1
export QT_DEBUG_PLUGINS=1
export LD_LIBRARY_PATH=$script_dir

DISPLAY=:0.0 $script_dir/helloqt
