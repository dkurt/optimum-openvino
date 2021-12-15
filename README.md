# Optimum OpenVINO

Optimum OpenVINO is an extension for [Optimum](https://github.com/huggingface/optimum) library which brings [Intel OpenVINO](https://github.com/openvinotoolkit/openvino) backend for [Hugging Face Transformers](https://github.com/huggingface/transformers) :hugs:.

This project provides multiple APIs to enable different tools:
* [OpenVINO Runtime](#openvino-runtime)
* [Neural Network Compression Framework (NNCF)](#nncf)
* [Post-Training Optimization Tool (POT)](#pot)

## Install

```
pip install -i https://test.pypi.org/simple/ optimum-openvino
```

## OpenVINO Runtime

This module provides an inference API for Hugging Face models. There are options to use models with PyTorch\*, TensorFlow\* pretrained weights or use native OpenVINO IR format (a pair of files `ov_model.xml` and `ov_model.bin`).

To use OpenVINO backend, import one of the `AutoModel` classes with `OV` prefix. Specify a model name or local path in `from_pretrained` method.

```python
from optimum.intel.openvino import OVAutoModel

# PyTorch trained model with OpenVINO backend
model = OVAutoModel.from_pretrained(<name_or_path>, from_pt=True)

# TensorFlow trained model with OpenVINO backend
model = OVAutoModel.from_pretrained(<name_or_path>, from_tf=True)

# Initialize a model from OpenVINO IR
model = OVAutoModel.from_pretrained(<name_or_path>)
```

## NNCF

TBD

## POT

TBD
