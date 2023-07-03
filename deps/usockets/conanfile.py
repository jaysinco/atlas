import sys, os
from myconanfile import MyConanFile
from conans import ConanFile, tools
from conan.tools.cmake import CMakeToolchain, CMake
from conan.tools.files import collect_libs, copy, rmdir, chdir
from conans import MSBuild, AutoToolsBuildEnvironment

class UsocketsConan(MyConanFile):
    name = "usockets"
    version = "0.8.1"
    homepage = "https://github.com/uNetworking/uSockets"
    description = "Miniscule cross-platform eventing, networking & crypto for async applications"
    license = "Apache-2.0"

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "fPIC": [True, False],
        "eventloop": ["libuv", "boost"],
    }
    default_options = {
        "fPIC": True,
        "eventloop": "libuv",
    }

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        pass

    def requirements(self):
        self.requires(self._ref_pkg("openssl/1.1.1q"))
        if self.options.eventloop == "libuv":
            self.requires(self._ref_pkg("libuv/1.44.2"))
        elif self.options.eventloop == "boost":
            self.requires(self._ref_pkg("boost/1.79.0"))

    def source(self):
        srcFile = self._src_abspath(f"{self.name}-{self.version}.tar.gz")
        tools.unzip(srcFile, destination=self.source_folder, strip_root=True)
        self._patch_sources(self._dirname(__file__), [
            "0001-fix-makefile-flags.patch",
            "0002-fix-vcxproj-static.patch",
            "0003-fix-vcxproj-use-openssl.patch",
        ])

    def _build_msvc(self):
        with chdir(self, os.path.join(self.source_folder)):
            msbuild = MSBuild(self)
            msbuild.build(project_file="uSockets.vcxproj", platforms={"x86": "Win32"})

    def _build_configure(self):
        autotools = AutoToolsBuildEnvironment(self)
        autotools.fpic = self.options.get_safe("fPIC", False)
        with chdir(self, self.source_folder):
            args = []
            args.append("WITH_OPENSSL=1")
            if self.options.eventloop == "libuv":
                args.append("WITH_LIBUV=1")
            elif self.options.eventloop == "boost":
                args.append("WITH_ASIO=1")
            args.extend(f"{key}={value}" for key, value in autotools.vars.items())
            autotools.make(target="default", args=args)

    def build(self):
        if self.settings.compiler == "Visual Studio":
            self._build_msvc()
        else:
            self._build_configure()

    def package(self):
        copy(self, "LICENSE", dst=os.path.join(self.package_folder, "licenses"),
            src=self.source_folder)
        copy(self, "*.h", dst=os.path.join(self.package_folder, "include"),
            src=os.path.join(self.source_folder, "src"), keep_path=True)
        copy(self, "*.a", dst=os.path.join(self.package_folder, "lib"),
            src=self.source_folder, keep_path=False)
        copy(self, "*.lib", dst=os.path.join(self.package_folder, "lib"),
            src=self.source_folder, keep_path=False)
        rmdir(self, os.path.join(self.package_folder, "include", "internal"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "uSockets")
        self.cpp_info.set_property("cmake_target_name", "uSockets::uSockets")
        self.cpp_info.set_property("pkg_config_name", "uSockets")
        self.cpp_info.libs = ["uSockets"]
        self.cpp_info.defines.append("LIBUS_USE_OPENSSL")
        if self.options.eventloop == "libuv":
            self.cpp_info.defines.append("LIBUS_USE_LIBUV")
        elif self.options.eventloop == "boost":
            self.cpp_info.defines.append("LIBUS_USE_ASIO")
