import sys, os
from myconanfile import MyConanFile
from conans import ConanFile, tools
from conan.tools.cmake import CMakeToolchain, CMake, CMakeDeps
from conan.tools.files import collect_libs, copy, rmdir


class LibcurlConan(MyConanFile):
    name = "libcurl"
    version = "7.85.0"
    homepage = "https://curl.se"
    description = "A command line tool and library for transferring data with URLs"
    license = "curl"

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

    def requirements(self):
        self.requires(self._ref_pkg("openssl/1.1.1q"))
        self.requires(self._ref_pkg("zlib/1.2.12"))
        self.requires(self._ref_pkg("zstd/1.5.2"))

    def source(self):
        srcFile = self._src_abspath(f"curl-{self.version}.tar.gz")
        tools.unzip(srcFile, destination=self.source_folder, strip_root=True)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["BUILD_TESTING"] = False
        tc.variables["BUILD_CURL_EXE"] = True
        tc.variables["CURL_DISABLE_LDAP"] = True
        tc.variables["BUILD_SHARED_LIBS"] = self.options.shared
        tc.variables["CURL_STATICLIB"] = not self.options.shared
        tc.variables["CMAKE_DEBUG_POSTFIX"] = ""
        tc.variables["CURL_USE_SCHANNEL"] = False
        tc.variables["CURL_USE_OPENSSL"] = True
        tc.variables["CURL_USE_WOLFSSL"] = False
        tc.variables["USE_NGHTTP2"] = False
        tc.variables["CURL_ZLIB"] = True
        tc.variables["CURL_BROTLI"] = False
        tc.variables["CURL_ZSTD"] = True
        tc.variables["CURL_USE_LIBSSH2"] = False
        tc.variables["ENABLE_ARES"] = False
        tc.variables["CURL_DISABLE_PROXY"] = False
        tc.variables["USE_LIBRTMP"] = False
        tc.variables["USE_LIBIDN2"] = False
        tc.variables["CURL_DISABLE_RTSP"] = False
        tc.variables["CURL_DISABLE_CRYPTO_AUTH"] = False
        tc.variables["CURL_DISABLE_VERBOSE_STRINGS"] = False
        tc.variables["NTLM_WB_ENABLED"] = True
        tc.cache_variables["CURL_CA_BUNDLE"] = "none"
        tc.cache_variables["CURL_CA_PATH"] = "none"
        tc.generate()
        cmake_deps = CMakeDeps(self)
        cmake_deps.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        copy(self, "COPYING", dst=os.path.join(
            self.package_folder, "licenses"), src=self.source_folder)
        pemname = "cacert.pem"
        pempath = self._src_abspath(pemname)
        copy(self, pemname, src=os.path.dirname(pempath),
            dst=os.path.join(self.package_folder, "res"))
        cmake = CMake(self)
        cmake.install()
        rmdir(self, os.path.join(self.package_folder, "lib", "pkgconfig"))
        rmdir(self, os.path.join(self.package_folder, "lib", "cmake"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "CURL")
        self.cpp_info.set_property("cmake_target_name", "CURL::libcurl")
        self.cpp_info.set_property("pkg_config_name", "libcurl")
        self.cpp_info.libs = collect_libs(self, folder="lib")
        if self.settings.os in ["Linux", "FreeBSD"]:
            self.cpp_info.system_libs.extend(["rt", "pthread"])
        elif self.settings.os == "Windows":
            self.cpp_info.system_libs = ["ws2_32"]
        if not self.options.shared:
            self.cpp_info.defines.append("CURL_STATICLIB=1")