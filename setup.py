import setuptools
import os

VERSION = 0.1
PACKAGES = [
    "ztpcraft",
    "ztpcraft.toolbox",
    "ztpcraft.aedt_interface",
    "ztpcraft.util",
]

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(CURRENT_DIR, "requirements.txt")) as requirements:
    INSTALL_REQUIRES = requirements.read().splitlines()

setuptools.setup(
    name="ztpcraft",
    version=VERSION,
    description="Tianpu Zhao's personal toolbox",
    url="https://github.com/pacosynthesis/ztpcrafts",
    author="Tianpu Zhao",
    author_email="pacosynthesis@gmail.com",
    license="MIT",
    packages=PACKAGES,
    zip_safe=False,
    install_requires=INSTALL_REQUIRES,
    # extras_require=EXTRA_REQUIRES,
    python_requires=">=3.10",
)


def write_version_py(filename="ztpcraft/version.py"):
    if os.path.exists(filename):
        os.remove(filename)
    versionfile = open(filename, "w")
    try:
        versionfile.write(
            f"# THIS FILE IS GENERATED FROM chencrafts SETUP.PY\n"
            "version = '{VERSION}'"
        )
    finally:
        versionfile.close()


write_version_py()
