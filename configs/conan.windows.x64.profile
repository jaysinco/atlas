[settings]
os=Windows
os_build=Windows
arch=x86_64
arch_build=x86_64
# Visual Studio 2019
compiler=Visual Studio
compiler.version=16
compiler.runtime=MD
build_type=Release
[options]
[build_requires]
[env]
[conf]
tools.cmake.cmaketoolchain:generator=Ninja
tools.microsoft.bash:subsystem=msys2
tools.microsoft.bash:path=C:/msys64/usr/bin/bash.exe
