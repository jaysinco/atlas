import sys, os
from myconanfile import MyConanFile
from conans import ConanFile, tools
from conan.tools.files import rmdir, rm, rename, copy
from conan.tools.microsoft import is_msvc
from conans.errors import ConanInvalidConfiguration


class NasmConan(MyConanFile):
    name = "nasm"
    version = "2.15.05"
    homepage = "http://www.nasm.us"
    description = "The Netwide Assembler, NASM, is an 80x86 and x86-64 assembler"
    license = "BSD-2-Clause"

    settings = "os", "arch", "compiler", "build_type"

    def configure(self):
        if self.settings.os != "Windows":
            raise ConanInvalidConfiguration("Only Windows supported")

    def package_id(self):
        del self.info.settings.compiler
        del self.info.settings.build_type

    def source(self):
        srcFile = self._src_abspath(f"nasm-{self.version}-win64.zip")
        tools.unzip(srcFile, destination=self.source_folder, strip_root=True)

    def package(self):
        copy(self, "LICENSE", dst=os.path.join(
            self.package_folder, "licenses"), src=self.source_folder)
        copy(self, "*.exe", dst=os.path.join(
            self.package_folder, "bin"), src=self.source_folder)

    def package_info(self):
        bin_path = os.path.join(self.package_folder, "bin")
        self.output.info("Appending PATH environment variable: %s" % bin_path)
        self.env_info.PATH.append(bin_path)
