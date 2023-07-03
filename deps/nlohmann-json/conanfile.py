import sys, os
from myconanfile import MyConanFile
from conans import ConanFile, tools
from conan.tools.cmake import CMakeToolchain, CMake
from conan.tools.files import collect_libs, copy, rmdir


class NlohmannJsonConan(MyConanFile):
    name = "nlohmann-json"
    version = "3.11.2"
    homepage = "https://github.com/nlohmann/json"
    description = "JSON for Modern C++ parser and generator."
    license = "MIT"

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "multiple_headers": [True, False]
    }
    default_options = {
        "multiple_headers": False
    }

    def source(self):
        srcFile = self._src_abspath(f"{self.name}-{self.version}.tar.gz")
        tools.unzip(srcFile, destination=self.source_folder, strip_root=True)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["JSON_BuildTests"] = False
        tc.variables["JSON_MultipleHeaders"] = self.options.multiple_headers
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package_id(self):
        self.info.clear()

    def package(self):
        copy(self, "LICENSE.MIT", dst=os.path.join(
            self.package_folder, "licenses"), src=self.source_folder)
        cmake = CMake(self)
        cmake.install()
        rmdir(self, os.path.join(self.package_folder, "share"))
        try:
            os.remove(os.path.join(self.package_folder, "nlohmann_json.natvis"))
        except FileNotFoundError:
            pass

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "nlohmann-json")
        self.cpp_info.set_property("cmake_target_name", "nlohmann-json")
        self.cpp_info.set_property("pkg_config_name", "nlohmann-json")
