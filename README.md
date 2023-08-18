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
* linux (devos)
* windows
    * visual studio 2019
    * msys2
        ```
        pacman --noconfirm -S base-devel binutils gcc
        ```
