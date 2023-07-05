import sys, os
from myconanfile import MyConanFile
from conans import ConanFile, tools
from conan.tools.cmake import CMakeToolchain, CMake
from conan.tools.files import collect_libs, copy, rmdir


class StackWalkerConan(MyConanFile):
    name = "stackwalker"
    version = "2023.06.24"
    homepage = "https://github.com/JochenKalmbach/StackWalker"
    description = "Walking the callstack in windows applications"
    license = "BSD-2-Clause"

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
    }
    default_options = {
        "shared": False,
    }

    def configure(self):
        if self.settings.os != "Windows":
            raise ConanInvalidConfiguration("Only Windows supported")

    def source(self):
        srcFile = self._src_abspath(f"{self.name}-v{self.version}.tar.gz")
        tools.unzip(srcFile, destination=self.source_folder, strip_root=True)
        self._patch_sources(self._dirname(__file__), [
            "0001-fix-cmake-install-error.patch",
        ])

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["StackWalker_DISABLE_TESTS"] = False
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

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "stackwalker")
        self.cpp_info.set_property("cmake_target_name", "stackwalker")
        self.cpp_info.libs = collect_libs(self, folder="lib")
