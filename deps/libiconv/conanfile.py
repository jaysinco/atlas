import sys
import os
from myconanfile import MyConanFile
from conans import ConanFile, tools
from conan.tools.env import VirtualBuildEnv
from conan.tools.microsoft import unix_path, is_msvc
from conan.tools.files import collect_libs, copy, rmdir, rm, rename
from conan.tools.gnu import Autotools, AutotoolsToolchain
from conan.tools.scm import Version


class LibiconvConan(MyConanFile):
    name = "libiconv"
    version = "1.17"
    homepage = "https://www.gnu.org/software/libiconv/"
    description = "Convert text to and from Unicode"
    license = "LGPL-2.1"

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

    def build_requirements(self):
        if self._settings_build.os == "Windows":
            self.win_bash = True

    def generate(self):
        def requires_fs_flag():
            return (self.settings.compiler == "Visual Studio" and Version(self.settings.compiler.version) >= "12") or \
                    (self.settings.compiler == "msvc" and Version(self.settings.compiler.version) >= "180")

        tc = AutotoolsToolchain(self)
        if requires_fs_flag():
            tc.extra_cflags.append("-FS")

        env = tc.environment()

        if is_msvc(self) or self._is_clang_cl:
            cc, lib, link = self._msvc_tools
            build_aux_path = os.path.join(self.source_folder, "build-aux")
            lt_compile = unix_path(self, os.path.join(build_aux_path, "compile"))
            lt_ar = unix_path(self, os.path.join(build_aux_path, "ar-lib"))
            env.define("CC", f"{lt_compile} {cc} -nologo")
            env.define("CXX", f"{lt_compile} {cc} -nologo")
            env.define("LD", f"{link}")
            env.define("STRIP", ":")
            env.define("AR", f"{lt_ar} {lib}")
            env.define("RANLIB", ":")
            env.define("NM", "dumpbin -symbols")
            env.define("win32_target", "_WIN32_WINNT_VISTA")

        tc.generate(env)

        env = VirtualBuildEnv(self)
        env.generate()

    def source(self):
        srcFile = self._src_abspath(f"{self.name}-{self.version}.tar.gz")
        tools.unzip(srcFile, destination=self.source_folder, strip_root=True)

    def build(self):
        if self._settings_build.os == "Windows":
            self.win_bash = True

        autotools = Autotools(self)
        autotools.configure()
        autotools.make()

    def package(self):
        if self._settings_build.os == "Windows":
            self.win_bash = True

        copy(self, "COPYING.LIB", self.source_folder, os.path.join(self.package_folder, "licenses"))
        autotools = Autotools(self)
        autotools.install(args=[f"DESTDIR={unix_path(self, self.package_folder)}"])

        rm(self, "*.la", os.path.join(self.package_folder, "lib"))
        rmdir(self, os.path.join(self.package_folder, "share"))

        if (is_msvc(self) or self._is_clang_cl) and self.options.shared:
            for import_lib in ["iconv", "charset"]:
                rename(self, os.path.join(self.package_folder, "lib", "{}.dll.lib".format(import_lib)),
                             os.path.join(self.package_folder, "lib", "{}.lib".format(import_lib)))

    def package_info(self):
        self.cpp_info.set_property("cmake_find_mode", "both")
        self.cpp_info.set_property("cmake_file_name", "Iconv")
        self.cpp_info.set_property("cmake_target_name", "Iconv::Iconv")
        self.cpp_info.libs = ["iconv", "charset"]

    @property
    def _settings_build(self):
        return getattr(self, "settings_build", self.settings)

    @property
    def _is_clang_cl(self):
        return (self.settings.compiler == "clang" and self.settings.os == "Windows") \
               or self.settings.get_safe("compiler.toolset") == "ClangCL"

    @property
    def _msvc_tools(self):
        return ("clang-cl", "llvm-lib", "lld-link") if self._is_clang_cl else ("cl", "lib", "link")
