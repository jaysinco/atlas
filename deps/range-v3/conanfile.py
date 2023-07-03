import sys, os
from myconanfile import MyConanFile
from conans import ConanFile, tools
from conan.tools.microsoft import msvc_runtime_flag, is_msvc
from conan.tools.files import copy


class Rangev3Conan(MyConanFile):
    name = "range-v3"
    version = "0.12.0"
    homepage = "https://github.com/ericniebler/range-v3"
    description = "Experimental range library for C++11/14/17"
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
        copy(self, "*", dst=os.path.join(self.package_folder, "include"),
             src=os.path.join(self.source_folder, "include"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "range-v3")
        self.cpp_info.set_property("cmake_target_name", "range-v3::range-v3")
        self.cpp_info.set_property("pkg_config_name", "range-v3")
        if is_msvc(self):
            self.cpp_info.cxxflags = ["/permissive-"]
