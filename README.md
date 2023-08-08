# Compile Env
* ninja
* cmake
* python3
    ```
    pip3 install conan==1.52 -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```
* flutter
    ```
    export PUB_HOSTED_URL=https://pub.flutter-io.cn
    export FLUTTER_STORAGE_BASE_URL=https://storage.flutter-io.cn
    git clone -b 3.10.6 git@gitee.com:jaysinco/flutter.git
    flutter doctor
    flutter create --template=app --platforms=windows,linux --project-name=hello_flutter flutter
    ```
* linux (ubuntu 20.04)
    * cat /etc/apt/sources.list
        ```
        deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse
        deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
        deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse
        deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-security main restricted universe multiverse
        deb https://launchpad.proxy.ustclug.org/ubuntu-toolchain-r/test/ubuntu focal main
        ```
    * install packages
        ```
        apt-get install gcc-11 g++-11
        update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 60 --slave /usr/bin/g++ g++ /usr/bin/g++-11
        apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
        apt-get install python3-pip ninja-build git git-lfs
        snap install cmake
        ```
* windows
    * visual studio 2019
    * msys2
        ```
        pacman --noconfirm -S base-devel binutils gcc
        ```
