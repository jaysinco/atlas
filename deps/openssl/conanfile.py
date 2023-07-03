import sys, os
from myconanfile import MyConanFile
from conans import ConanFile, tools
from conan.tools.files import collect_libs, copy, rmdir
from conan.tools.microsoft import msvc_runtime_flag, is_msvc
from conan.tools.build import build_jobs
import fnmatch


class OpenSSLConan(MyConanFile):
    name = "openssl"
    version = "1.1.1q"
    homepage = "https://github.com/openssl/openssl"
    description = "A toolkit for the Transport Layer Security (TLS) and Secure Sockets Layer (SSL) protocols"
    license = "OpenSSL"

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "no_threads": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "no_threads": False,
    }

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            del self.options.fPIC

    def requirements(self):
        self.requires(self._ref_pkg("zlib/1.2.12"))

    def build_requirements(self):
        if self._build_on_windows():
            self.build_requires(self._ref_pkg("strawberryperl/5.30.0.1"))
            self.build_requires(self._ref_pkg("nasm/2.15.05"))

    def source(self):
        srcFile = self._src_abspath(f"{self.name}-{self.version}.tar.gz")
        tools.unzip(srcFile, destination=self.source_folder, strip_root=True)

    def package_id(self):
        del self.info.settings.build_type

    def package(self):
        copy(self, "LICENSE", dst=os.path.join(
            self.package_folder, "licenses"), src=self.source_folder)
        with tools.vcvars(self) if is_msvc(self) else tools.no_op():
            self._make()
            self._make_install()

        for root, _, files in os.walk(self.package_folder):
            for filename in files:
                if fnmatch.fnmatch(filename, "*.pdb"):
                    os.unlink(os.path.join(self.package_folder, root, filename))

        rmdir(self, os.path.join(self.package_folder, "lib", "pkgconfig"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "OpenSSL")
        self.cpp_info.set_property("pkg_config_name", "openssl")
        self.cpp_info.components["crypto"].set_property("cmake_target_name", "OpenSSL::Crypto")
        self.cpp_info.components["crypto"].set_property("pkg_config_name", "libcrypto")
        self.cpp_info.components["ssl"].set_property("cmake_target_name", "OpenSSL::SSL")
        self.cpp_info.components["ssl"].set_property("pkg_config_name", "libssl")
        if is_msvc(self):
            libsuffix = "" if self.settings.build_type == "Debug" else ""
            self.cpp_info.components["ssl"].libs = ["libssl" + libsuffix]
            self.cpp_info.components["crypto"].libs = ["libcrypto" + libsuffix]
        else:
            self.cpp_info.components["ssl"].libs = ["ssl"]
            self.cpp_info.components["crypto"].libs = ["crypto"]

        self.cpp_info.components["ssl"].requires = ["crypto"]
        self.cpp_info.components["crypto"].requires = ["zlib::zlib"]

        if self.settings.os == "Windows":
            self.cpp_info.components["crypto"].system_libs.extend(
                ["crypt32", "ws2_32", "advapi32", "user32", "bcrypt"])
        elif self.settings.os in ["Linux", "FreeBSD"]:
            self.cpp_info.components["crypto"].system_libs.extend(["dl", "rt"])
            self.cpp_info.components["ssl"].system_libs.append("dl")
            if not self.options.no_threads:
                self.cpp_info.components["crypto"].system_libs.append("pthread")
                self.cpp_info.components["ssl"].system_libs.append("pthread")

    def _make_install(self):
        install_cmd = f"{self._make_exe()} install_sw"
        self.output.info(install_cmd)
        self.run(command=install_cmd, cwd=self.source_folder)

    def _make(self):
        config_cmd = f'perl ./Configure {self._config_flags()}'
        self.output.info(config_cmd)
        self.run(command=config_cmd, cwd=self.source_folder)

        build_cmd = f"{self._make_exe()} {self._make_flags()}"
        self.output.info(build_cmd)
        self.run(command=build_cmd, cwd=self.source_folder)

    @property
    def _target(self):
        if is_msvc(self):
            return "VC-WIN64A"
        if self.settings.compiler == "gcc":
            return "linux-x86_64"
        if self.settings.compiler == "clang":
            return "linux-x86_64-clang"
        return None

    def _make_exe(self):
        if is_msvc(self):
            return "nmake"
        else:
            return "make"

    def _make_flags(self):
        flags = []
        if not is_msvc(self):
            flags.append(f"-j{build_jobs(self)}")
        return " ".join(flags)

    def _config_flags(self):
        flags = [
            '"%s"' % self._target,
            "shared" if self.options.shared else "no-shared",
            "--prefix=\"%s\"" % self.package_folder,
            "--openssldir=\"%s\"" % os.path.join(self.package_folder, "res"),
            "no-unit-test",
            "no-threads" if self.options.no_threads else "threads"
        ]
        flags.append("no-tests")
        flags.append("--debug" if self.settings.build_type == "Debug" else "--release")
        if self.settings.os == "Linux":
            flags.append("-fPIC" if self.options.get_safe("fPIC", True) else "no-pic")

        zlib_info = self.deps_cpp_info["zlib"]
        include_path = zlib_info.include_paths[0]
        if self.settings.os == "Windows":
            lib_path = "%s/%s.lib" % (zlib_info.lib_paths[0], zlib_info.libs[0])
        else:
            lib_path = zlib_info.lib_paths[0]
        include_path = self._normalize_path(include_path)
        lib_path = self._normalize_path(lib_path)
        if self.options["zlib"].shared:
            flags.append("zlib-dynamic")
        else:
            flags.append("zlib")
        flags.extend([
            '--with-zlib-include="%s"' % include_path,
            '--with-zlib-lib="%s"' % lib_path,
        ])
        return " ".join(flags)
