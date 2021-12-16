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

NNCF is used for model training with applying such features like quantization, pruning. To enable NNCF in you training pipeline do the following steps:

1. Import `NNCFAutoConfig`:

```python
from optimum.intel.nncf import NNCFAutoConfig
```

> **NOTE**: `NNCFAutoConfig` must be imported before `transformers` to make magic work

2. Initialize a config from `.json` file:

```python
nncf_config = NNCFAutoConfig.from_json(training_args.nncf_config)
```

3. Pass a config to `Trainer` object. In example,

```python
trainer = QuestionAnsweringTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    eval_examples=eval_examples if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    post_process_function=post_processing_function,
    compute_metrics=compute_metrics,
    nncf_config=nncf_config,
)
```

## POT

TBD
