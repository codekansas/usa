#!/usr/bin/env python
# pylint: disable=import-outside-toplevel
# pylint: disable=import-error
# mypy: ignore-errors

import os
import re
import shutil
import subprocess
import sys
import sysconfig
from multiprocessing import cpu_count
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


def get_arch_list() -> str:
    import torch.cuda

    if torch.cuda.is_available():
        major_num, minor_num = torch.cuda.get_device_capability()
        return f"{major_num}.{minor_num}"
    arch_list: list[str] = []
    for arch in torch.cuda.get_arch_list():
        match = re.match(r"sm_(\d+)", arch)
        assert match, f"Invalid arch list: {torch.cuda.get_arch_list()}"
        arch_list.append(match.group(1))
    assert arch_list, f"Empty arch list: {torch.cuda.get_arch_list()} (did you install the wrong PyTorch version?)"
    return ";".join(".".join([i[:-1], i[-1:]]) for i in arch_list)


class CMakeExtension(Extension):
    """CMake extension.

    This is a subclass of setuptools.Extension that allows specifying the
    location of the CMakeLists.txt file.

    Usage:
        setup(
            name="my_package",
            ext_modules=[CMakeExtension("my_package")],
            cmdclass={"build_ext": CMakeBuild},
        )
    """

    def __init__(self, name: str) -> None:
        super().__init__(name, sources=[])

        self.sourcedir = os.fspath(Path(__file__).parent.resolve() / name)


class CMakeBuild(build_ext):
    """CMake build extension.

    This is a subclass of setuptools.command.build_ext.build_ext that runs
    cmake to build the extension.

    Usage:
        setup(
            name="my_package",
            ext_modules=[CMakeExtension("my_package")],
            cmdclass={"build_ext": CMakeBuild},
        )
    """

    def initialize_options(self) -> None:
        super().initialize_options()

        # Set parallel build.
        self.parallel = cpu_count()

    def build_extensions(self) -> None:
        self.check_extensions_list(self.extensions)
        self._build_extensions_serial()

    def build_extension(self, ext: CMakeExtension) -> None:
        import cmake  # noqa: F401
        import pybind11  # noqa: F401
        import torch._C  # noqa: F401
        from torch.utils.cpp_extension import (  # noqa: F401
            CUDA_HOME,
            include_paths as torch_include_paths,
        )

        if CUDA_HOME is not None and not shutil.which("nvcc"):
            raise RuntimeError("NVCC installation not found")
        if torch.utils.cmake_prefix_path is None:
            raise RuntimeError("CMake prefix path not found")

        cmake_path = os.path.join(cmake.CMAKE_BIN_DIR, "cmake")
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)  # type: ignore[no-untyped-call]
        extdir = ext_fullpath.parent.resolve()
        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Need to copy PyBind flags.
        cmake_cxx_flags: list[str] = []
        for name in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
            val = getattr(torch._C, f"_PYBIND11_{name}")
            if val is not None:
                cmake_cxx_flags += [f'-DPYBIND11_{name}=\\"{val}\\"']

        cmake_cxx_flags += [f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"]

        # Found this necessary for building on Apple M1 machine.
        cmake_cxx_flags += ["-fPIC", "-Wl,-undefined,dynamic_lookup", "-Wno-unused-command-line-argument"]

        # System include paths.
        cmake_include_dirs = [*torch_include_paths(cuda=CUDA_HOME is not None), pybind11.get_include()]
        python_include_path = sysconfig.get_path("include", scheme="posix_prefix")
        if python_include_path is not None:
            cmake_include_dirs += [python_include_path]
        cmake_cxx_flags += [f"-isystem {dir_name}" for dir_name in cmake_include_dirs]

        # Sets paths to various CMake stuff.
        cmake_prefix_path_str = ";".join([torch.utils.cmake_prefix_path, pybind11.get_cmake_dir()])
        cmake_cxx_flags_str = " ".join(cmake_cxx_flags)

        # Gets CMake arguments.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DPYTHON_INCLUDE_DIR={sysconfig.get_path('include')}",
            f"-DPYTHON_LIBRARY={sysconfig.get_path('platlib')}",
            f"-DCMAKE_PREFIX_PATH={cmake_prefix_path_str}",
            f"-DCMAKE_CXX_FLAGS={cmake_cxx_flags_str}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]

        if CUDA_HOME is not None:
            nvcc_path = shutil.which("nvcc")
            assert nvcc_path is not None
            cmake_args += [
                f"-DCUDA_TOOLKIT_ROOT_DIR='{CUDA_HOME}'",
                f"-DTORCH_CUDA_ARCH_LIST='{get_arch_list()}'",
                f"-DCMAKE_CUDA_COMPILER='{Path(nvcc_path).resolve()}'",
            ]

        build_args = []
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        env = os.environ.copy()

        if self.compiler.compiler_type != "msvc":
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    # pylint: disable-next=import-outside-toplevel
                    import ninja  # noqa: F401

                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        else:
            single_config = any(x in cmake_generator for x in ("NMake", "Ninja"))
            contains_arch = any(x in cmake_generator for x in ("ARM", "Win64"))
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]
            if not single_config:
                cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += [f"-DCMAKE_OSX_ARCHITECTURES={';'.join(archs)}"]

        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            if hasattr(self, "parallel") and self.parallel:
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        print(f"Building extension {ext.name} in {build_temp}")
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        def show_and_run(cmd: list[str]) -> None:
            print(" ".join(cmd))
            subprocess.run(cmd, env=env, check=True)

        show_and_run([cmake_path, f"-S{ext.sourcedir}", f"-B{build_temp}"] + cmake_args)
        show_and_run([cmake_path, "--build", f"{build_temp}"] + (["--"] + build_args if build_args else []))

    def copy_extensions_to_source(self) -> None:
        pass

    def run(self) -> None:
        super().run()

        def gen_stubs(ext: Extension) -> None:
            cmd = ["stubgen", "-p", f"{ext.name.replace('/', '.')}", "-o", "."]
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)

        if shutil.which("stubgen") is not None:
            for ext in self.extensions:
                gen_stubs(ext)


with open("README.md", "r", encoding="utf-8") as f:
    long_description: str = f.read()


with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements: list[str] = f.read().splitlines()


with open("requirements-dev.txt", "r", encoding="utf-8") as f:
    requirements_dev: list[str] = f.read().splitlines()


setup(
    name="ml-project",
    version="0.0.1",
    description="Template repository for ML projects",
    author="Benjamin Bolte",
    url="https://github.com/codekansas/ml-project-template",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    setup_requires=["cmake", "mypy", "pybind11", "torch"],
    install_requires=requirements,
    tests_require=requirements_dev,
    extras_require={"dev": requirements_dev},
    ext_modules=[CMakeExtension("project/cpp")],
    cmdclass={"build_ext": CMakeBuild},
    exclude_package_data={
        "project": [
            "cpp/**/*.cpp",
        ],
    },
)
