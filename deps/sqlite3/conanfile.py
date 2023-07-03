import sys, os
from myconanfile import MyConanFile
from conans import ConanFile, tools
from conan.tools.cmake import CMakeToolchain, CMake, CMakeDeps
from conan.tools.files import collect_libs, copy, rmdir


class Sqlite3Conan(MyConanFile):
    name = "sqlite3"
    version = "3.39.4"
    homepage = "https://www.sqlite.org"
    description = "Self-contained, serverless, in-process SQL database engine."
    license = "Unlicense"

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "threadsafe": [0, 1, 2],
        "enable_column_metadata": [True, False],
        "enable_dbstat_vtab": [True, False],
        "enable_explain_comments": [True, False],
        "enable_fts3": [True, False],
        "enable_fts3_parenthesis": [True, False],
        "enable_fts4": [True, False],
        "enable_fts5": [True, False],
        "enable_json1": [True, False],
        "enable_soundex": [True, False],
        "enable_preupdate_hook": [True, False],
        "enable_rtree": [True, False],
        "use_alloca": [True, False],
        "omit_load_extension": [True, False],
        "omit_deprecated": [True, False],
        "enable_math_functions": [True, False],
        "enable_unlock_notify": [True, False],
        "enable_default_secure_delete": [True, False],
        "disable_gethostuuid": [True, False],
        "max_column": "ANY",
        "max_variable_number": "ANY",
        "max_blob_size": "ANY",
        "build_executable": [True, False],
        "enable_default_vfs": [True, False],
        "enable_dbpage_vtab": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "threadsafe": 1,
        "enable_column_metadata": True,
        "enable_dbstat_vtab": False,
        "enable_explain_comments": False,
        "enable_fts3": False,
        "enable_fts3_parenthesis": False,
        "enable_fts4": False,
        "enable_fts5": False,
        "enable_json1": False,
        "enable_soundex": False,
        "enable_preupdate_hook": False,
        "enable_rtree": True,
        "use_alloca": False,
        "omit_load_extension": False,
        "omit_deprecated": False,
        "enable_math_functions": True,
        "enable_unlock_notify": True,
        "enable_default_secure_delete": False,
        "disable_gethostuuid": False,
        "max_column": None,
        "max_variable_number": None,
        "max_blob_size": None,
        "build_executable": True,
        "enable_default_vfs": True,
        "enable_dbpage_vtab": False,
    }

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            del self.options.fPIC

    def source(self):
        srcFile = self._src_abspath(f"sqlite-{self.version}.zip")
        tools.unzip(srcFile, destination=self.source_folder, strip_root=True)
        copy(self, "CMakeLists.txt", src=self._dirname(__file__),
            dst=self.source_folder)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["SQLITE3_SRC_DIR"] = self.source_folder.replace("\\", "/")
        tc.variables["SQLITE3_VERSION"] = self.version
        tc.variables["SQLITE3_BUILD_EXECUTABLE"] = self.options.build_executable
        tc.variables["THREADSAFE"] = self.options.threadsafe
        tc.variables["ENABLE_COLUMN_METADATA"] = self.options.enable_column_metadata
        tc.variables["ENABLE_DBSTAT_VTAB"] = self.options.enable_dbstat_vtab
        tc.variables["ENABLE_EXPLAIN_COMMENTS"] = self.options.enable_explain_comments
        tc.variables["ENABLE_FTS3"] = self.options.enable_fts3
        tc.variables["ENABLE_FTS3_PARENTHESIS"] = self.options.enable_fts3_parenthesis
        tc.variables["ENABLE_FTS4"] = self.options.enable_fts4
        tc.variables["ENABLE_FTS5"] = self.options.enable_fts5
        tc.variables["ENABLE_JSON1"] = self.options.enable_json1
        tc.variables["ENABLE_PREUPDATE_HOOK"] = self.options.enable_preupdate_hook
        tc.variables["ENABLE_SOUNDEX"] = self.options.enable_soundex
        tc.variables["ENABLE_RTREE"] = self.options.enable_rtree
        tc.variables["ENABLE_UNLOCK_NOTIFY"] = self.options.enable_unlock_notify
        tc.variables["ENABLE_DEFAULT_SECURE_DELETE"] = self.options.enable_default_secure_delete
        tc.variables["USE_ALLOCA"] = self.options.use_alloca
        tc.variables["OMIT_LOAD_EXTENSION"] = self.options.omit_load_extension
        tc.variables["OMIT_DEPRECATED"] = self.options.omit_deprecated
        tc.variables["ENABLE_MATH_FUNCTIONS"] = self.options.enable_math_functions
        tc.variables["HAVE_FDATASYNC"] = True
        tc.variables["HAVE_GMTIME_R"] = True
        tc.variables["HAVE_LOCALTIME_R"] = self.settings.os != "Windows"
        tc.variables["HAVE_POSIX_FALLOCATE"] = not (self.settings.os in ["Windows", "Android"])
        tc.variables["HAVE_STRERROR_R"] = True
        tc.variables["HAVE_USLEEP"] = True
        tc.variables["DISABLE_GETHOSTUUID"] = self.options.disable_gethostuuid
        if self.options.max_column:
            tc.variables["MAX_COLUMN"] = self.options.max_column
        if self.options.max_variable_number:
            tc.variables["MAX_VARIABLE_NUMBER"] = self.options.max_variable_number
        if self.options.max_blob_size:
            tc.variables["MAX_BLOB_SIZE"] = self.options.max_blob_size
        tc.variables["DISABLE_DEFAULT_VFS"] = not self.options.enable_default_vfs
        tc.variables["ENABLE_DBPAGE_VTAB"] = self.options.enable_dbpage_vtab
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "SQLite3")
        self.cpp_info.set_property("cmake_target_name", "SQLite::SQLite3")
        self.cpp_info.set_property("pkg_config_name", "sqlite3")
        self.cpp_info.libs = ["sqlite3"]
        if self.options.omit_load_extension:
            self.cpp_info.defines.append("SQLITE_OMIT_LOAD_EXTENSION")
        if self.settings.os in ["Linux", "FreeBSD"]:
            if self.options.threadsafe:
                self.cpp_info.system_libs.append("pthread")
            if not self.options.omit_load_extension:
                self.cpp_info.system_libs.append("dl")
            if self.options.enable_fts5 or self.options.get_safe("enable_math_functions"):
                self.cpp_info.system_libs.append("m")
        elif self.settings.os == "Windows":
            if self.options.shared:
                self.cpp_info.defines.append("SQLITE_API=__declspec(dllimport)")