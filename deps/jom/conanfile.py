import sys, os
from myconanfile import MyConanFile
from conans import ConanFile, tools
from conan.tools.files import rmdir, rm, rename, copy
from conan.tools.microsoft import is_msvc
from conans.errors import ConanInvalidConfiguration


class JomConan(MyConanFile):
    name = "jom"
    version = "1.1.3"
    homepage = "http://wiki.qt.io/Jom"
    description = "jom is a clone of nmake to support the execution of multiple independent commands in parallel"
    license = "GPL-3.0"

    settings = "os", "arch", "compiler", "build_type"

    def configure(self):
        if self.settings.os != "Windows":
            raise ConanInvalidConfiguration("Only Windows supported")

    def package_id(self):
        del self.info.settings.compiler
        del self.info.settings.build_type

    def source(self):
        srcFile = self._src_abspath(f"jom-{self.version}.zip")
        tools.unzip(srcFile, destination=self.source_folder, strip_root=False)

    def package(self):
        copy(self, "*.exe", dst=os.path.join(
            self.package_folder, "bin"), src=self.source_folder)

    def package_info(self):
        bin_path = os.path.join(self.package_folder, "bin")
        self.output.info("Appending PATH environment variable: %s" % bin_path)
        self.env_info.PATH.append(bin_path)
