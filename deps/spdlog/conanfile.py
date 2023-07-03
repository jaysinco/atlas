import sys, os
from myconanfile import MyConanFile
from conans import ConanFile, tools
from conan.tools.cmake import CMakeToolchain, CMake, CMakeDeps
from conan.tools.files import collect_libs, copy, rmdir


class SpdlogConan(MyConanFile):
    name = "spdlog"
    version = "1.10.0"
    homepage = "https://github.com/gabime/spdlog"
    description = "Fast C++ logging library"
    license = "MIT"

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "no_exceptions": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "no_exceptions": False,
    }

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            del self.options.fPIC

    def requirements(self):
        self.requires(self._ref_pkg("fmt/8.1.1"))

    def source(self):
        srcFile = self._src_abspath(f"{self.name}-{self.version}.tar.gz")
        tools.unzip(srcFile, destination=self.source_folder, strip_root=True)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["SPDLOG_BUILD_EXAMPLE"] = False
        tc.variables["SPDLOG_BUILD_EXAMPLE_HO"] = False
        tc.variables["SPDLOG_BUILD_TESTS"] = False
        tc.variables["SPDLOG_BUILD_TESTS_HO"] = False
        tc.variables["SPDLOG_BUILD_BENCH"] = False
        tc.variables["SPDLOG_FMT_EXTERNAL"] = True
        tc.variables["SPDLOG_FMT_EXTERNAL_HO"] = False
        tc.variables["SPDLOG_BUILD_SHARED"] = self.options.shared
        tc.variables["SPDLOG_WCHAR_SUPPORT"] = False
        tc.variables["SPDLOG_WCHAR_FILENAMES"] = False
        tc.variables["SPDLOG_INSTALL"] = True
        tc.variables["SPDLOG_NO_EXCEPTIONS"] = self.options.no_exceptions
        tc.generate()
        cmake_deps = CMakeDeps(self)
        cmake_deps.generate()

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

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "spdlog")
        self.cpp_info.set_property("cmake_target_name", "spdlog::spdlog")
        self.cpp_info.set_property("pkg_config_name", "spdlog")
        self.cpp_info.requires = ["fmt::fmt"]
        self.cpp_info.libs = collect_libs(self, folder="lib")
        self.cpp_info.defines.append("SPDLOG_COMPILED_LIB")
        self.cpp_info.defines.append("SPDLOG_FMT_EXTERNAL")
        if self.options.no_exceptions:
            self.cpp_info.defines.append("SPDLOG_NO_EXCEPTIONS")
        if self.settings.os in ["Linux", "FreeBSD"]:
            self.cpp_info.system_libs.extend(["pthread"])
