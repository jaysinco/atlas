import sys, os
from myconanfile import MyConanFile
from conans import ConanFile, tools
from conan.tools.cmake import CMakeToolchain, CMake
from conan.tools.files import collect_libs, copy, rmdir


class OpenclHeadersConan(MyConanFile):
    name = "opencl-headers"
    version = "2023.04.17"
    homepage = "https://github.com/KhronosGroup/OpenCL-Headers"
    description = "Khronos OpenCL-Headers"
    license = "Apache-2.0"

    settings = "os", "arch", "compiler", "build_type"

    def source(self):
        headersFile = self._src_abspath(f"OpenCL-Headers-{self.version}.tar.gz")
        tools.unzip(headersFile, destination=self.source_folder, strip_root=False)
        clhppFile = self._src_abspath(f"OpenCL-CLHPP-{self.version}.tar.gz")
        tools.unzip(clhppFile, destination=self.source_folder, strip_root=False)

    def package_id(self):
        self.info.clear()

    def package(self):
        copy(self, "LICENSE", dst=os.path.join(self.package_folder, "licenses"),
             src=os.path.join(self.source_folder, f"OpenCL-Headers-{self.version}"))
        copy(self, "*.h", src=os.path.join(self.source_folder, f"OpenCL-Headers-{self.version}", "CL"),
            dst=os.path.join(self.package_folder, "include", "CL"))
        copy(self, "*.hpp", src=os.path.join(self.source_folder, f"OpenCL-CLHPP-{self.version}", "include", "CL"),
            dst=os.path.join(self.package_folder, "include", "CL"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "opencl-headers")
        self.cpp_info.set_property("cmake_target_name", "opencl-headers")
        self.cpp_info.set_property("pkg_config_name", "opencl-headers")
