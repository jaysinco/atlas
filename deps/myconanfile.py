from conans import ConanFile, tools
from conan.tools.files import collect_libs, copy, rmdir
from typing import List
import os

class MyConanFile(ConanFile):
    url = "https://github.com/JaySinco/Prototyping"

    def export(self):
        copy(self, "myconanfile.py", dst=self.export_folder, src=self._dirname(__file__))

    def layout(self):
        build_folder = "out"
        build_type = str(self.settings.build_type)
        self.folders.source = "src"
        self.folders.build = os.path.join(build_folder, build_type)
        self.folders.generators = os.path.join(
            self.folders.build, "generators")

    def _src_abspath(self, filename: str):
        return os.path.join(tools.get_env("MY_SOURCE_REPO"), filename)

    def _ref_pkg(self, pkgname: str):
        return f"{pkgname}@jaysinco/stable"

    def _dirname(self, file: str):
        return os.path.dirname(os.path.abspath(file))

    def _patch_sources(self, dirname: str, patches: List[str]):
        for pat in patches:
            tools.patch(self.source_folder, os.path.join(dirname, "patches", pat))

    def _build_on_windows(self):
        settings_build = getattr(self, "settings_build", self.settings)
        return settings_build.os == "Windows"

    def _normalize_path(self, path):
        if self.settings.os == "Windows":
            return path.replace("\\", "/")
        else:
            return path