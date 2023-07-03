import sys, os
from myconanfile import MyConanFile
from conans import ConanFile, tools
from conan.tools.cmake import CMakeToolchain, CMake
from conan.tools.files import collect_libs, copy, rmdir


class FmtConan(MyConanFile):
    name = "fmt"
    version = "8.1.1"
    homepage = "https://github.com/fmtlib/fmt"
    description = "A safe and fast alternative to printf and IOStreams"
    license = "MIT"

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_fmt_alias": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "with_fmt_alias": False,
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
        tc.variables["FMT_DOC"] = False
        tc.variables["FMT_TEST"] = False
        tc.variables["FMT_INSTALL"] = True
        tc.variables["FMT_LIB_DIR"] = "lib"
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package_id(self):
        del self.info.options.with_fmt_alias

    def package(self):
        copy(self, "LICENSE.rst", dst=os.path.join(
            self.package_folder, "licenses"), src=self.source_folder)
        cmake = CMake(self)
        cmake.install()
        rmdir(self, os.path.join(self.package_folder, "lib", "pkgconfig"))
        rmdir(self, os.path.join(self.package_folder, "lib", "cmake"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "fmt")
        self.cpp_info.set_property("cmake_target_name", "fmt::fmt")
        self.cpp_info.set_property("pkg_config_name", "fmt")
        self.cpp_info.libs = collect_libs(self, folder="lib")
        if self.settings.os in ["Linux", "FreeBSD"]:
            self.cpp_info.system_libs.extend(["m"])
        if self.options.with_fmt_alias:
            self.cpp_info.defines.append("FMT_STRING_ALIAS=1")
        if self.options.shared:
            self.cpp_info.defines.append("FMT_SHARED")
