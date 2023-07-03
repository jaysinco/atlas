import sys, os
from myconanfile import MyConanFile
from conans import ConanFile, tools
from conan.tools.files import copy


class ExpectedLiteConan(MyConanFile):
    name = "expected-lite"
    version = "0.5.0"
    homepage = "https://github.com/martinmoene/expected-lite"
    description = "Expected objects in C++11 and later in a single-file header-only library"
    license = "BSL-1.0"

    settings = "os", "arch", "compiler", "build_type"

    def source(self):
        srcFile = self._src_abspath(f"{self.name}-{self.version}.tar.gz")
        tools.unzip(srcFile, destination=self.source_folder, strip_root=True)

    def package_id(self):
        self.info.clear()

    def package(self):
        copy(self, "LICENSE.txt", dst=os.path.join(
            self.package_folder, "licenses"), src=self.source_folder)
        copy(self, "*.hpp", dst=os.path.join(self.package_folder, "include"),
             src=os.path.join(self.source_folder, "include"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "expected-lite")
        self.cpp_info.set_property(
            "cmake_target_name", "nonstd::expected-lite")
        self.cpp_info.set_property("pkg_config_name", "expected-lite")
