import sys, os
from myconanfile import MyConanFile
from conans import ConanFile, tools
from conan.tools.files import copy


class ConcurrentqueueConan(MyConanFile):
    name = "concurrent-queue"
    version = "1.0.3"
    homepage = "https://github.com/cameron314/concurrentqueue"
    description = "A fast multi-producer, multi-consumer lock-free concurrent queue for C++11"
    license = "BSD-2-Clause"

    settings = "os", "arch", "compiler", "build_type"

    def source(self):
        srcFile = self._src_abspath(f"{self.name}-{self.version}.tar.gz")
        tools.unzip(srcFile, destination=self.source_folder, strip_root=True)

    def package_id(self):
        self.info.clear()

    def package(self):
        copy(self, "LICENSE.md", dst=os.path.join(
            self.package_folder, "licenses"), src=self.source_folder)
        for file in ["blockingconcurrentqueue.h", "concurrentqueue.h", "lightweightsemaphore.h"]:
            copy(self, file, src=self.source_folder,
                dst=os.path.join(self.package_folder, "include", "moodycamel"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "concurrent-queue")
        self.cpp_info.set_property("cmake_target_name", "concurrent-queue")
        self.cpp_info.set_property("pkg_config_name", "concurrent-queue")
        if self.settings.os in ["Linux", "FreeBSD"]:
            self.cpp_info.system_libs = ["pthread"]
