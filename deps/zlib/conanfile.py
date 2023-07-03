import sys, os
from myconanfile import MyConanFile
from conans import ConanFile, tools
from conan.tools.cmake import CMakeToolchain, CMake, CMakeDeps
from conan.tools.files import collect_libs, copy, rmdir, rm


class ZlibConan(MyConanFile):
    name = "zlib"
    version = "1.2.12"
    homepage = "https://zlib.net"
    description = "A Massively Spiffy Yet Delicately Unobtrusive Compression Library"
    license = "Zlib"

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
            "0001-fix-cmake-install-error.patch",
        ])

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["SKIP_INSTALL_ALL"] = False
        tc.variables["SKIP_INSTALL_LIBRARIES"] = False
        tc.variables["SKIP_INSTALL_HEADERS"] = False
        tc.variables["SKIP_INSTALL_FILES"] = True
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
        if self.settings.os in ["Linux", "FreeBSD"]:
            mask = "*.so*"
            if self.options.shared:
                mask = "*.a"
            tools.remove_files_by_mask(os.path.join(self.package_folder, "lib"), mask)

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "ZLIB")
        self.cpp_info.set_property("cmake_target_name", "ZLIB::ZLIB")
        self.cpp_info.set_property("pkg_config_name", "zlib")
        if self.settings.os == "Windows":
            libname = "zlib"
            if not self.options.shared:
                libname += "static"
            if self.settings.build_type == "Debug":
                libname += "d"
        elif self.settings.os in ["Linux", "FreeBSD"]:
            libname = "z"
        self.cpp_info.libs = [libname]
