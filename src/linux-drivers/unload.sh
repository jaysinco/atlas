#!/bin/bash

set -e

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
git_root="$(git rev-parse --show-toplevel)"

module="hello"
device="hello"

sudo rmmod $module
sudo rm -f /dev/${device}0
