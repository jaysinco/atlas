import sys, os
from myconanfile import MyConanFile
from conans import ConanFile, tools
from conan.tools.cmake import CMakeToolchain, CMake
from conan.tools.files import collect_libs, copy, rmdir


class Catch2Conan(MyConanFile):
    name = "catch2"
    version = "2.13.9"
    homepage = "https://github.com/catchorg/Catch2"
    description = "A modern, C++-native, header-only, framework for unit-tests, TDD and BDD"
    license = "BSL-1.0"

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "fPIC": [True, False],
        "with_main": [True, False],
        "with_benchmark": [True, False],
        "with_prefix": [True, False],
        "default_reporter": "ANY",
    }
    default_options = {
        "fPIC": True,
        "with_main": True,
        "with_benchmark": True,
        "with_prefix": False,
        "default_reporter": None,
    }

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if not self.options.with_main:
            del self.options.fPIC
            del self.options.with_benchmark

    def source(self):
        srcFile = self._src_abspath(f"{self.name}-{self.version}.tar.gz")
        tools.unzip(srcFile, destination=self.source_folder, strip_root=True)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["BUILD_TESTING"] = False
        tc.variables["CATCH_INSTALL_DOCS"] = False
        tc.variables["CATCH_INSTALL_HELPERS"] = True
        tc.variables["CATCH_BUILD_STATIC_LIBRARY"] = self.options.with_main
        tc.variables["CATCH_CONFIG_ENABLE_BENCHMARKING"] = self.options.get_safe(
            "with_benchmark", False)
        tc.variables["CATCH_CONFIG_PREFIX_ALL"] = self.options.with_prefix
        if self.options.default_reporter:
            tc.variables["CATCH_CONFIG_DEFAULT_REPORTER"] = self._default_reporter_str
        tc.generate()

    def build(self):
        if self.options.with_main:
            cmake = CMake(self)
            cmake.configure()
            cmake.build()

    def package_id(self):
        if not self.options.with_main:
            self.info.clear()

    def package(self):
        copy(self, "LICENSE.txt", dst=os.path.join(
            self.package_folder, "licenses"), src=self.source_folder)
        cmake = CMake(self)
        cmake.install()
        rmdir(self, os.path.join(self.package_folder, "share"))
        rmdir(self, os.path.join(self.package_folder, "lib", "cmake"))
        for cmake_file in ["ParseAndAddCatchTests.cmake", "Catch.cmake", "CatchAddTests.cmake"]:
            copy(self, cmake_file,
                 src=os.path.join(self.source_folder, "contrib"),
                 dst=os.path.join(self.package_folder, "lib", "cmake", "Catch2"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "Catch2")
        self.cpp_info.set_property("cmake_target_name", "Catch2::Catch2{}".format(
            "WithMain" if self.options.with_main else ""))
        self.cpp_info.set_property("pkg_config_name", "catch2".format(
            "-with-main" if self.options.with_main else ""))
        if self.options.with_main:
            self.cpp_info.libs = collect_libs(self, folder="lib")
        if self.options.get_safe("with_benchmark", False):
            self.cpp_info.defines.append("CATCH_CONFIG_ENABLE_BENCHMARKING")
        if self.options.with_prefix:
            self.cpp_info.defines.append("CATCH_CONFIG_PREFIX_ALL")
        if self.options.default_reporter:
            self.cpp_info.defines.append("CATCH_CONFIG_DEFAULT_REPORTER={}".format(
                self._default_reporter_str))

    @property
    def _default_reporter_str(self):
        return '"{}"'.format(str(self.options.default_reporter).strip('"'))
