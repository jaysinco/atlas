from myconanfile import MyConanFile
from conan.tools.cmake import CMakeToolchain, CMake, CMakeDeps
import os
import sys
import platform

class AtlasConan(MyConanFile):
    name = "atlas"
    version = "0.1.0"
    homepage = "https://github.com/jaysinco/atlas"
    description = "c++ main repo"
    license = "MIT"

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "spdlog:no_exceptions": False,
    }

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            del self.options.fPIC

    def requirements(self):
        self.requires(self._ref_pkg("spdlog/1.10.0"))
        self.requires(self._ref_pkg("boost/1.79.0"))
        self.requires(self._ref_pkg("expected-lite/0.5.0"))
        self.requires(self._ref_pkg("catch2/2.13.9"))
        self.requires(self._ref_pkg("range-v3/0.12.0"))
        self.requires(self._ref_pkg("libiconv/1.17"))
        self.requires(self._ref_pkg("nlohmann-json/3.11.2"))
        self.requires(self._ref_pkg("uwebsockets/20.14.0"))
        self.requires(self._ref_pkg("concurrent-queue/1.0.3"))
        self.requires(self._ref_pkg("threadpool/3.3.0"))
        self.requires(self._ref_pkg("qt5/5.15.6"))

    def layout(self):
        build_folder = "out"
        build_type = str(self.settings.build_type)
        self.folders.source = ""
        self.folders.build = os.path.join(build_folder, build_type)
        self.folders.generators = os.path.join(
            self.folders.build, "generators")

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG"] = os.path.join(self.source_folder, "bin")
        tc.variables["CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE"] = os.path.join(self.source_folder, "bin", "Release")
        tc.variables["CMAKE_PREFIX_PATH"] = self._cmake_path()
        self._setup_pkg_root(tc)
        tc.generate()
        cmake_deps = CMakeDeps(self)
        cmake_deps.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def _setup_pkg_root(self, tc):
        for pkg in self.deps_cpp_info._dependencies:
            root = "{}_ROOT".format(pkg)
            tc.variables[root] = self.deps_cpp_info[pkg].cpp_info.rootpath

    def _cmake_path(self):
        prefix_path = []
        cmake_dir = {
            "qt5" : "lib/cmake",
        }
        for pkg in cmake_dir:
            prefix_path.append(self._normalize_path(
                os.path.join(self.deps_cpp_info[pkg].cpp_info.rootpath, cmake_dir[pkg])))

        return "%s;${CMAKE_PREFIX_PATH}" % (";".join(prefix_path))
