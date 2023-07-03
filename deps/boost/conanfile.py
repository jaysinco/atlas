import sys, os
from myconanfile import MyConanFile
from conans import ConanFile, tools
from conan.tools.files import collect_libs, copy, rmdir
from conan.tools.microsoft import msvc_runtime_flag, is_msvc
from conan.tools.build import build_jobs
from conan.errors import ConanException
import yaml


class BoostConan(MyConanFile):
    name = "boost"
    version = "1.79.0"
    homepage = "https://www.boost.org"
    description = "Boost provides free peer-reviewed portable C++ source libraries"
    license = "BSL-1.0"

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
    }

    _cached_dependencies = None

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            del self.options.fPIC

    def export(self):
        super().export()
        copy(self, self._dependency_filename, dst=self.export_folder, src=self._dirname(__file__))

    def source(self):
        srcFile = self._src_abspath(f"{self.name}-{self.version}.tar.gz")
        tools.unzip(srcFile, destination=self.source_folder, strip_root=True)
        with tools.vcvars(self) if is_msvc(self) else tools.no_op():
            bootstrap_cmd = "{} {}".format(
                self._bootstrap_exe, self._bootstrap_flags)
            self.run(command=bootstrap_cmd, cwd=self.source_folder)

    def package_id(self):
        del self.info.settings.build_type

    def package(self):
        with tools.vcvars(self) if is_msvc(self) else tools.no_op():
            install_cmd = "{} {} install --prefix={}".format(
                self._b2_exe, self._build_flags, self.package_folder)
            self.run(command=install_cmd, cwd=self.source_folder)

        copy(self, "LICENSE_1_0.txt", dst=os.path.join(
            self.package_folder, "licenses"), src=self.source_folder)

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "Boost")
        self.cpp_info.filenames["cmake_find_package"] = "Boost"
        self.cpp_info.filenames["cmake_find_package_multi"] = "Boost"
        self.cpp_info.names["cmake_find_package"] = "Boost"
        self.cpp_info.names["cmake_find_package_multi"] = "Boost"

        # - Use 'headers' component for all includes + defines
        # - Use '_libboost' component to attach extra system_libs, ...

        self.cpp_info.components["headers"].libs = []
        self.cpp_info.components["headers"].set_property("cmake_target_name", "Boost::headers")
        self.cpp_info.components["headers"].names["cmake_find_package"] = "headers"
        self.cpp_info.components["headers"].names["cmake_find_package_multi"] = "headers"
        self.cpp_info.components["headers"].names["pkg_config"] = "boost"

        # Boost::boost is an alias of Boost::headers
        self.cpp_info.components["_boost_cmake"].requires = ["headers"]
        self.cpp_info.components["_boost_cmake"].set_property("cmake_target_name", "Boost::boost")
        self.cpp_info.components["_boost_cmake"].names["cmake_find_package"] = "boost"
        self.cpp_info.components["_boost_cmake"].names["cmake_find_package_multi"] = "boost"

        self.cpp_info.components["_libboost"].requires = ["headers"]

        self.cpp_info.components["disable_autolinking"].libs = []
        self.cpp_info.components["disable_autolinking"].set_property("cmake_target_name", "Boost::disable_autolinking")
        self.cpp_info.components["disable_autolinking"].names["cmake_find_package"] = "disable_autolinking"
        self.cpp_info.components["disable_autolinking"].names["cmake_find_package_multi"] = "disable_autolinking"
        self.cpp_info.components["disable_autolinking"].names["pkg_config"] = "boost_disable_autolinking"  # FIXME: disable on pkg_config

        # Even headers needs to know the flags for disabling autolinking ...
        # magic_autolink is an option in the recipe, so if a consumer wants this version of boost,
        # then they should not get autolinking.
        # Note that autolinking can sneak in just by some file #including a header with (eg) boost/atomic.hpp,
        # even if it doesn't use any part that requires linking with libboost_atomic in order to compile.
        # So a boost-header-only library that links to Boost::headers needs to see BOOST_ALL_NO_LIB
        # in order to avoid autolinking to libboost_atomic

        # This define is already imported into all of the _libboost libraries from this recipe anyway,
        # so it would be better to be consistent and ensure ANYTHING using boost (headers or libs) has consistent #defines.

        # Same applies for for BOOST_AUTO_LINK_{layout}:
        # consumer libs that use headers also need to know what is the layout/filename of the libraries.
        #
        # eg, if using the "tagged" naming scheme, and a header triggers an autolink,
        # then that header's autolink request had better be configured to request the "tagged" library name.
        # Otherwise, the linker will be looking for a (eg) "versioned" library name, and there will be a link error.

        # Note that "_libboost" requires "headers" so these defines will be applied to all the libraries too.
        self.cpp_info.components["headers"].requires.append("disable_autolinking")
        if is_msvc(self):
            self.cpp_info.components["disable_autolinking"].defines = ["BOOST_ALL_NO_LIB"]
            self.output.info("Disabled magic autolinking (smart and magic decisions)")

        self.cpp_info.components["dynamic_linking"].libs = []
        self.cpp_info.components["dynamic_linking"].set_property("cmake_target_name", "Boost::dynamic_linking")
        self.cpp_info.components["dynamic_linking"].names["cmake_find_package"] = "dynamic_linking"
        self.cpp_info.components["dynamic_linking"].names["cmake_find_package_multi"] = "dynamic_linking"
        self.cpp_info.components["dynamic_linking"].names["pkg_config"] = "boost_dynamic_linking"  # FIXME: disable on pkg_config
        # A library that only links to Boost::headers can be linked into another library that links a Boost::library,
        # so for this reasons, the header-only library should know the BOOST_ALL_DYN_LINK definition as it will likely
        # change some important part of the boost code and cause linking errors downstream.
        # This is in the same theme as the notes above, re autolinking.
        self.cpp_info.components["headers"].requires.append("dynamic_linking")
        if self._shared:
            # A Boost::dynamic_linking cmake target does only make sense for a shared boost package
            self.cpp_info.components["dynamic_linking"].defines = ["BOOST_ALL_DYN_LINK"]

        libsuffix = ""
        libformatdata = {}
        libformatdata["py_major"] = 3
        libformatdata["py_minor"] = 8

        def add_libprefix(n):
            """ On MSVC, static libraries are built with a 'lib' prefix. Some libraries do not support shared, so are always built as a static library. """
            libprefix = ""
            if is_msvc(self) and (not self._shared or n in self._dependencies["static_only"]):
                libprefix = "lib"
            return libprefix + n

        all_detected_libraries = set(l[:-4] if l.endswith(".dll") else l for l in collect_libs(self))
        all_expected_libraries = set()
        incomplete_components = []

        def filter_transform_module_libraries(names):
            libs = []
            for name in names:
                if name in ("boost_graph_parallel", "boost_mpi", "boost_mpi_python"):
                    continue
                if name in ("boost_stacktrace_windbg", "boost_stacktrace_windbg_cached", "boost_stacktrace_backtrace") and self.settings.os == "Linux":
                    continue
                if name in ("boost_stacktrace_addr2line", "boost_stacktrace_backtrace", "boost_stacktrace_basic") and self.settings.os == "Windows":
                    continue
                if "_numa" in name:
                    continue
                if "_numpy" in name:
                    continue
                if "_python" in name:
                    continue
                new_name = add_libprefix(name.format(**libformatdata)) + libsuffix
                libs.append(new_name)
            return libs

        for module in self._dependencies["dependencies"].keys():

            module_libraries = filter_transform_module_libraries(self._dependencies["libs"][module])

            # Don't create components for modules that should have libraries, but don't have (because of filter)
            if self._dependencies["libs"][module] and not module_libraries:
                continue

            all_expected_libraries = all_expected_libraries.union(module_libraries)
            if set(module_libraries).difference(all_detected_libraries):
                incomplete_components.append(module)

            # Starting v1.69.0 Boost.System is header-only. A stub library is
            # still built for compatibility, but linking to it is no longer
            # necessary.
            # https://www.boost.org/doc/libs/1_75_0/libs/system/doc/html/system.html#changes_in_boost_1_69
            if module == "system":
                module_libraries = []

            self.cpp_info.components[module].libs = module_libraries

            self.cpp_info.components[module].requires = self._dependencies["dependencies"][module] + ["_libboost"]
            self.cpp_info.components[module].set_property("cmake_target_name", "Boost::" + module)
            self.cpp_info.components[module].names["cmake_find_package"] = module
            self.cpp_info.components[module].names["cmake_find_package_multi"] = module
            self.cpp_info.components[module].names["pkg_config"] = f"boost_{module}"

        for incomplete_component in incomplete_components:
            self.output.warn(f"Boost component '{incomplete_component}' is missing libraries. Try building boost with '-o boost:without_{incomplete_component}'. (Option is not guaranteed to exist)")

        non_used = all_detected_libraries.difference(all_expected_libraries)
        if non_used:
            self.output.warn(f"These libraries were built, but were not used in any boost module: {non_used}")

        non_built = all_expected_libraries.difference(all_detected_libraries)
        if non_built:
            raise ConanException(f"These libraries were expected to be built, but were not built: {non_built}")

        if is_msvc(self):
            # https://github.com/conan-community/conan-boost/issues/127#issuecomment-404750974
            self.cpp_info.components["_libboost"].system_libs.append("bcrypt")
        elif self.settings.os == "Linux":
            # https://github.com/conan-community/community/issues/135
            self.cpp_info.components["_libboost"].system_libs.append("rt")
            self.cpp_info.components["_libboost"].system_libs.append("pthread")

    @property
    def _dependency_filename(self):
        return f"dependencies-{self.version}.yml"

    @property
    def _dependencies(self):
        if self._cached_dependencies is None:
            dependencies_filepath = os.path.join(self._dirname(__file__), self._dependency_filename)
            if not os.path.isfile(dependencies_filepath):
                raise ConanException(f"Cannot find {dependencies_filepath}")
            with open(dependencies_filepath, encoding='utf-8') as f:
                self._cached_dependencies = yaml.safe_load(f)
        return self._cached_dependencies

    @property
    def _shared(self):
        return self.options.get_safe("shared", self.default_options["shared"])

    @property
    def _b2_exe(self):
        return "b2.exe" if tools.os_info.is_windows else "./b2"

    @property
    def _bootstrap_exe(self):
        return "bootstrap.bat" if tools.os_info.is_windows else "./bootstrap.sh"

    @property
    def _toolset(self):
        if is_msvc(self):
            return "msvc"
        if self.settings.compiler in ["clang", "gcc"]:
            return str(self.settings.compiler)
        return None

    @property
    def _b2_architecture(self):
        if str(self.settings.arch).startswith("x86"):
            return "x86"
        return None

    @property
    def _bootstrap_flags(self):
        return "--without-libraries=python --with-toolset={}".format(self._toolset)

    @property
    def _b2_address_model(self):
        if str(self.settings.arch) in ("x86_64"):
            return "64"
        return "32"

    @property
    def _gnu_cxx11_abi(self):
        try:
            if str(self.settings.compiler.libcxx) == "libstdc++":
                return "0"
            if str(self.settings.compiler.libcxx) == "libstdc++11":
                return "1"
        except ConanException:
            pass
        return None

    @property
    def _b2_stdlib(self):
        return { "libstdc++11": "libstdc++" }.get(
            str(self.settings.compiler.libcxx),
            str(self.settings.compiler.libcxx)
        )

    @property
    def _b2_cxxflags(self):
        cxx_flags = []
        if self.options.get_safe("fPIC"):
            cxx_flags.append("-fPIC")
        if self.settings.compiler in ("clang", "apple-clang"):
            cxx_flags.append(f"-stdlib={self._b2_stdlib}")
        return " ".join(cxx_flags)

    @property
    def _b2_linkflags(self):
        link_flags = []
        if self.settings.compiler in ("clang", "apple-clang"):
            link_flags.append(f"-stdlib={self._b2_stdlib}")
        return " ".join(link_flags)

    @property
    def _build_flags(self):
        flags = []
        if self.settings.build_type == "Debug":
            flags.append("variant=debug")
        else:
            flags.append("variant=release")
        if is_msvc(self):
            flags.append(
                "runtime-link={}".format('static' if 'MT' in msvc_runtime_flag(self) else 'shared'))
        if self._gnu_cxx11_abi:
            flags.append(f"define=_GLIBCXX_USE_CXX11_ABI={self._gnu_cxx11_abi}")
        flags.append(f"link={'shared' if self.options.shared else 'static'}")
        flags.append(f"architecture={self._b2_architecture}")
        flags.append(f"address-model={self._b2_address_model}")
        flags.append(f"toolset={self._toolset}")
        flags.append("threading=multi")
        flags.append(f'cxxflags="{self._b2_cxxflags}"')
        flags.append(f'linkflags="{self._b2_linkflags}"')
        flags.append(f"-j{build_jobs(self)}")
        flags.append("--abbreviate-paths")
        flags.append("--layout=system")
        flags.append("--debug-configuration")
        flags.append(f"--build-dir={self.build_folder}")
        flags.append("-q")
        return " ".join(flags)
