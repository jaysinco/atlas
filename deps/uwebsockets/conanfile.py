import sys, os
from myconanfile import MyConanFile
from conans import ConanFile, tools
from conan.tools.cmake import CMakeToolchain, CMake
from conan.tools.files import collect_libs, copy, rmdir


class UwebsocketsConan(MyConanFile):
    name = "uwebsockets"
    version = "20.14.0"
    homepage = "https://github.com/uNetworking/uWebSockets"
    description = "Simple, secure & standards compliant web server for the most demanding of applications"
    license = "Apache-2.0"

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "with_zlib": [True, False],
    }
    default_options = {
        "with_zlib": True,
    }

    def source(self):
        srcFile = self._src_abspath(f"{self.name}-{self.version}.tar.gz")
        tools.unzip(srcFile, destination=self.source_folder, strip_root=True)

    def requirements(self):
        if self.options.with_zlib:
            self.requires(self._ref_pkg("zlib/1.2.12"))
        self.requires(self._ref_pkg("usockets/0.8.1"))

    def package_id(self):
        self.info.clear()

    def package(self):
        copy(self, "LICENSE", dst=os.path.join(
            self.package_folder, "licenses"), src=self.source_folder)
        copy(self, "*.h",
            src=os.path.join(self.source_folder, "src"),
            dst=os.path.join(self.package_folder, "include", "uWebSockets"),
            keep_path=False)

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "uWebSockets")
        self.cpp_info.set_property("cmake_target_name", "uWebSockets")
        self.cpp_info.set_property("pkg_config_name", "uWebSockets")
        if not self.options.with_zlib:
            self.cpp_info.defines.append("UWS_NO_ZLIB")
