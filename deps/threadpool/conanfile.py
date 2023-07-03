import sys, os
from myconanfile import MyConanFile
from conans import ConanFile, tools
from conan.tools.files import copy


class ThreadpoolConan(MyConanFile):
    name = "threadpool"
    version = "3.3.0"
    homepage = "https://github.com/bshoshany/thread-pool"
    description = "A fast, lightweight, and easy-to-use C++17 thread pool library"
    license = "MIT"

    settings = "os", "arch", "compiler", "build_type"

    def source(self):
        srcFile = self._src_abspath(f"thread-pool-{self.version}.tar.gz")
        tools.unzip(srcFile, destination=self.source_folder, strip_root=True)

    def package_id(self):
        self.info.clear()

    def package(self):
        copy(self, "LICENSE.txt", dst=os.path.join(
            self.package_folder, "licenses"), src=self.source_folder)
        for file in ["BS_thread_pool.hpp", "BS_thread_pool_light.hpp"]:
            copy(self, file, src=self.source_folder,
                dst=os.path.join(self.package_folder, "include", "bshoshany"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "threadpool")
        self.cpp_info.set_property("cmake_target_name", "threadpool")
        self.cpp_info.set_property("pkg_config_name", "threadpool")
        if self.settings.os in ["Linux", "FreeBSD"]:
            self.cpp_info.system_libs = ["pthread"]
