import sys, os
from myconanfile import MyConanFile
from conans import ConanFile, tools
from conan.tools.cmake import CMakeToolchain, CMake, CMakeDeps
from conan.tools.files import collect_libs, copy, rmdir, rm


class LibcprConan(MyConanFile):
    name = "libcpr"
    version = "1.9.2"
    homepage = "https://docs.libcpr.org/"
    description = "C++ Requests: Curl for People, a spiritual port of Python Requests"
    license = "MIT"

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "signal": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "signal": True,
    }

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            del self.options.fPIC

    def requirements(self):
        self.requires(self._ref_pkg("libcurl/7.85.0"))

    def source(self):
        srcFile = self._src_abspath(f"cpr-{self.version}.tar.gz")
        tools.unzip(srcFile, destination=self.source_folder, strip_root=True)
        self._patch_sources(self._dirname(__file__), [
            "0001-fix-cmake-find-openssl.patch",
        ])

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["CPR_FORCE_USE_SYSTEM_CURL"] = True
        tc.variables["CPR_BUILD_TESTS"] = False
        tc.variables["CPR_GENERATE_COVERAGE"] = False
        tc.variables["CPR_USE_SYSTEM_GTEST"] = False
        tc.variables["CPR_CURL_NOSIGNAL"] = not self.options.signal
        tc.variables["CPR_FORCE_DARWINSSL_BACKEND"] = False
        tc.variables["CPR_FORCE_OPENSSL_BACKEND"] = True
        tc.variables["CPR_FORCE_WINSSL_BACKEND"] = False
        tc.variables["CMAKE_USE_OPENSSL"] = True
        tc.variables["CPR_ENABLE_SSL"] = True
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
        rmdir(self, os.path.join(self.package_folder, "lib", "cmake"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "cpr")
        self.cpp_info.set_property("cmake_target_name", "cpr::cpr")
        self.cpp_info.set_property("pkg_config_name", "cpr")
        self.cpp_info.libs = ["cpr"]
        if self.settings.os in ["Linux", "FreeBSD"]:
            self.cpp_info.system_libs.append("m")