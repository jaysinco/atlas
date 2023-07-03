import sys, os
from myconanfile import MyConanFile
from conans import ConanFile, tools
from conan.tools.files import rmdir, rm, rename, copy
from conan.tools.microsoft import is_msvc
from conans.errors import ConanInvalidConfiguration


class StrawberryperlConan(MyConanFile):
    name = "strawberryperl"
    version = "5.30.0.1"
    homepage = "http://strawberryperl.com"
    description = "Strawbery Perl for Windows. Useful as build_require"
    license = "GNU Public License or the Artistic License"

    settings = "os", "arch", "compiler", "build_type"

    def configure(self):
        if self.settings.os != "Windows":
            raise ConanInvalidConfiguration("Only windows supported for Strawberry Perl.")

    def package_id(self):
        del self.info.settings.compiler
        del self.info.settings.build_type

    def package(self):
        srcFile = self._src_abspath(f"strawberry-perl-{self.version}-64bit-portable.zip")
        tools.unzip(srcFile, destination=self.package_folder, strip_root=False)

        rm(self, "*.*", self.package_folder)
        rmdir(self, os.path.join(self.package_folder, "c"))
        rmdir(self, os.path.join(self.package_folder, "cpan"))
        rmdir(self, os.path.join(self.package_folder, "data"))
        rmdir(self, os.path.join(self.package_folder, "licenses"))
        rmdir(self, os.path.join(self.package_folder, "win32"))
        rename(self,
            src=os.path.join(self.package_folder, "perl", "bin"),
            dst=os.path.join(self.package_folder, "bin"))
        rename(self,
            src=os.path.join(self.package_folder, "perl", "lib"),
            dst=os.path.join(self.package_folder, "lib"))
        rename(self,
            src=os.path.join(self.package_folder, "perl", "vendor"),
            dst=os.path.join(self.package_folder, "vendor"))
        rename(self,
            src=os.path.join(self.package_folder, "perl", "site"),
            dst=os.path.join(self.package_folder, "site"))
        rmdir(self, os.path.join(self.package_folder, "perl"))

    def package_info(self):
        self.cpp_info.libdirs = []
        self.cpp_info.includedirs = []

        bin_path = os.path.join(self.package_folder, "bin")
        self.output.info("Appending PATH environment variable: %s" % bin_path)
        self.env_info.PATH.append(bin_path)
