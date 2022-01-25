# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
import os
from setuptools import find_namespace_packages, setup

# Ensure we match the version set in src/optimum/version.py
try:
    filepath = "optimum/version.py"
    with open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)


install_requires = [
    "transformers",
    "torch<1.10",
    "openvino-dev[onnx]",
    "nncf",
    "datasets",
]

# Add patches as data
folder = "optimum/intel/nncf/patches"
data = [os.path.join(folder, name) for name in os.listdir(folder)]

setup(
    name="optimum-openvino",
    version=__version__,
    description="Intel OpenVINO extension for Hugging Face Transformers",
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
    data_files=[("../../optimum/intel/nncf/patches", data)],
)
