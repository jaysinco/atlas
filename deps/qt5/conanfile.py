import sys, os
from myconanfile import MyConanFile
from conans import ConanFile, tools
from conan.tools.files import collect_libs, copy, rmdir
from conan.tools.microsoft import msvc_runtime_flag, is_msvc
from conan.tools.build import build_jobs


class QtConan(MyConanFile):
    name = "qt5"
    version = "5.15.6"
    homepage = "https://download.qt.io/official_releases/qt/"
    description = "Qt is a cross-platform framework for graphical user interfaces"
    license = "LGPL-3.0"

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
    }
    default_options = {
        "shared": True,
        "fPIC": True,
    }

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            del self.options.fPIC

    def build_requirements(self):
        if self._build_on_windows():
            self.build_requires(self._ref_pkg("jom/1.1.3"))

    def source(self):
        self._get_source("qtbase")
        self._get_source("qttools")

    def package_id(self):
        del self.info.settings.build_type

    def package(self):
        self._configure("qtbase")
        self._build_and_install("qtbase")

        with tools.environment_append({
            "LLVM_INSTALL_DIR": "C:/Program Files/LLVM"
        }) if tools.os_info.is_windows else tools.no_op():
            self._run_qmake("qttools")
            self._build_and_install("qttools")

        # self._build_doc_and_install("qtbase")
        # self._build_doc_and_install("qttools")

        tools.remove_files_by_mask(os.path.join(
            self.package_folder, "lib"), "*.pdb*")
        tools.remove_files_by_mask(os.path.join(
            self.package_folder, "bin"), "*.pdb")

        copy(self, "LICENSE*",
             dst=os.path.join(self.package_folder, "licenses"),
             src=os.path.join(self.source_folder, "qtbase"))

    def package_info(self):
        self.cpp_info.set_property("cmake_find_mode", "none")

    def _configure(self, name):
        with tools.vcvars(self) if is_msvc(self) else tools.no_op():
            config_cmd = "{} {}".format(self._configure_exe, self._config_flags)
            self.run(command=config_cmd,
                    cwd=os.path.join(self.source_folder, name))

    def _build_and_install(self, name):
        with tools.vcvars(self) if is_msvc(self) else tools.no_op():
            build_cmd = "{} {}".format(self._make_exe, self._build_flags)
            self.run(command=build_cmd,
                    cwd=os.path.join(self.source_folder, name))

            install_cmd = "{} install".format(self._make_exe)
            self.run(command=install_cmd,
                    cwd=os.path.join(self.source_folder, name))

    def _build_doc_and_install(self, name):
        with tools.vcvars(self) if is_msvc(self) else tools.no_op():
            build_cmd = "{} docs".format(self._make_exe)
            self.run(command=build_cmd,
                    cwd=os.path.join(self.source_folder, name))

            install_cmd = "{} install_qch_docs".format(self._make_exe)
            self.run(command=install_cmd,
                    cwd=os.path.join(self.source_folder, name))

    def _run_qmake(self, name):
        with tools.vcvars(self) if is_msvc(self) else tools.no_op():
            qmake_cmd = os.path.join(self.package_folder, "bin", "qmake")
            self.run(command=qmake_cmd,
                    cwd=os.path.join(self.source_folder, name))

    def _get_source(self, name):
        srcFile = self._src_abspath(f"{name}-{self.version}.tar.xz")
        tools.unzip(srcFile, destination=os.path.join(
            self.source_folder, name), strip_root=True)

    def _xplatform(self):
        if self.settings.os == "Linux":
            if self.settings.compiler == "gcc":
                return {"x86": "linux-g++-32",
                        "armv6": "linux-arm-gnueabi-g++",
                        "armv7": "linux-arm-gnueabi-g++",
                        "armv7hf": "linux-arm-gnueabi-g++",
                        "armv8": "linux-aarch64-gnu-g++"}.get(str(self.settings.arch), "linux-g++")
            elif self.settings.compiler == "clang":
                if self.settings.arch == "x86":
                    return "linux-clang-libc++-32" if self.settings.compiler.libcxx == "libc++" else "linux-clang-32"
                elif self.settings.arch == "x86_64":
                    return "linux-clang-libc++" if self.settings.compiler.libcxx == "libc++" else "linux-clang"

        elif self.settings.os == "Windows":
            return {
                "Visual Studio": "win32-msvc",
                "msvc": "win32-msvc",
                "gcc": "win32-g++",
                "clang": "win32-clang-g++",
            }.get(str(self.settings.compiler))

        return None

    @property
    def _configure_exe(self):
        return "configure.bat" if tools.os_info.is_windows else "./configure"

    @property
    def _config_flags(self):
        flags = []
        flags.append("--prefix={}".format(self.package_folder))
        flags.append("--nomake=examples")
        flags.append("--nomake=tests")
        flags.append("-confirm-license")
        flags.append("-opensource")
        if not self.options.shared:
            flags.append("-static")
            if is_msvc(self) and "MT" in msvc_runtime_flag(self):
                flags.append("-static-runtime")
        else:
            flags.append("-shared")
        if self.settings.build_type == "Debug":
            flags.append("-debug")
        else:
            flags.append("-release")
        if self.settings.get_safe("compiler.libcxx") == "libstdc++":
            flags.append("-D_GLIBCXX_USE_CXX11_ABI=0")
        elif self.settings.get_safe("compiler.libcxx") == "libstdc++11":
            flags.append("-D_GLIBCXX_USE_CXX11_ABI=1")
        xplatform_val = self._xplatform()
        if xplatform_val:
            flags.append(f"--platform={xplatform_val}")
        else:
            self.output.warn("host not supported!")
        flags.append("--opengl=desktop")
        flags.append("--c++std=c++17")
        return " ".join(flags)

    @property
    def _make_exe(self):
        return "jom" if tools.os_info.is_windows else "make"

    @property
    def _build_flags(self):
        flags = []
        flags.append(f"-j{build_jobs(self)}")
        return " ".join(flags)