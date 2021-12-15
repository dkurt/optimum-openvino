# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
from setuptools import find_namespace_packages, setup

# Ensure we match the version set in src/optimum/version.py
try:
    filepath = "optimum/version.py"
    with open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)


install_requires = [
    "transformers==4.9.1",
    "torch<1.10",
    "openvino-dev==2021.4.2",
    "nncf",
]

setup(
    name="optimum-openvino",
    version=__version__,
    description="OpenVINO backend for Optimum Library (Hugging Face Transformers extension)",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="transformers, quantization, pruning, training, intel, openvino",
    url="https://github.com/dkurt/optimum-openvino",
    author="Intel Corporation",
    author_email="openvino_pushbot@intel.com",
    license="Apache",
    packages=find_namespace_packages(include=["optimum.*"]),
    install_requires=install_requires,
)
