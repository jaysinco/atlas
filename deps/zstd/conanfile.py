import sys, os
from myconanfile import MyConanFile
from conans import ConanFile, tools
from conan.tools.cmake import CMakeToolchain, CMake
from conan.tools.files import collect_libs, copy, rmdir


class ZstdConan(MyConanFile):
    name = "zstd"
    version = "1.5.2"
    homepage = "https://github.com/facebook/zstd"
    description = "Zstandard - Fast real-time compression algorithm"
    license = "BSD-3-Clause"

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "threading": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "threading": True,
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

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["ZSTD_BUILD_PROGRAMS"] = False
        tc.variables["ZSTD_BUILD_STATIC"] = not self.options.shared
        tc.variables["ZSTD_BUILD_SHARED"] = self.options.shared
        tc.variables["ZSTD_MULTITHREAD_SUPPORT"] = self.options.threading
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure(build_script_folder=self._cmakelists_folder)
        cmake.build()

    def package(self):
        copy(self, "LICENSE", dst=os.path.join(
            self.package_folder, "licenses"), src=self.source_folder)
        cmake = CMake(self)
        cmake.install()
        rmdir(self, os.path.join(self.package_folder, "lib", "pkgconfig"))
        rmdir(self, os.path.join(self.package_folder, "lib", "cmake"))

    def package_info(self):
        zstd_cmake = "libzstd_shared" if self.options.shared else "libzstd_static"
        self.cpp_info.set_property("cmake_file_name", "zstd")
        self.cpp_info.set_property("cmake_target_name", "zstd::{}".format(zstd_cmake))
        self.cpp_info.set_property("pkg_config_name", "libzstd")
        self.cpp_info.libs = collect_libs(self, folder="lib")
        if self.settings.os in ["Linux", "FreeBSD"]:
            self.cpp_info.system_libs.append("pthread")

    @property
    def _cmakelists_folder(self):
        subfolder = os.path.join("build", "cmake")
        return os.path.join(self.source_folder, subfolder)