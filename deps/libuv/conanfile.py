import sys, os
from myconanfile import MyConanFile
from conans import ConanFile, tools
from conan.tools.cmake import CMakeToolchain, CMake
from conan.tools.files import collect_libs, copy, rmdir


class LibuvConan(MyConanFile):
    name = "libuv"
    version = "1.44.2"
    homepage = "https://libuv.org"
    description = "A multi-platform support library with a focus on asynchronous I/O"
    license = "MIT"

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
    }

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            del self.options.fPIC

    def source(self):
        srcFile = self._src_abspath(f"{self.name}-{self.version}.tar.gz")
        tools.unzip(srcFile, destination=self.source_folder, strip_root=True)
        self._patch_sources(self._dirname(__file__), [
            "0001-fix-cmake-static.patch",
        ])

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["LIBUV_BUILD_TESTS"] = False
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        copy(self, "LICENSE", dst=os.path.join(
            self.package_folder, "licenses"), src=self.source_folder)
        cmake = CMake(self)
        cmake.install()
        rmdir(self, os.path.join(self.package_folder, "lib", "pkgconfig"))
        rmdir(self, os.path.join(self.package_folder, "lib", "cmake"))
        rmdir(self, os.path.join(self.package_folder, "share"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "libuv")
        lib_name = "uv" if self.options.shared else "uv_a"
        self.cpp_info.set_property("cmake_target_name", lib_name)
        self.cpp_info.set_property("pkg_config_name", "libuv" if self.options.shared else "libuv-static")
        self.cpp_info.libs = [lib_name]
        if self.options.shared:
            self.cpp_info.defines = ["USING_UV_SHARED=1"]
        if self.settings.os == "Windows":
            self.cpp_info.system_libs.extend(["iphlpapi", "psapi", "userenv", "ws2_32"])
        elif self.settings.os in ["Linux", "FreeBSD"]:
            self.cpp_info.system_libs.extend(["dl", "pthread", "rt"])
