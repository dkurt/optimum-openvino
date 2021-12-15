# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import torch

from openvino.inference_engine import IECore

from transformers.file_utils import ModelOutput, cached_path, hf_bucket_url
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.utils import logging


logger = logging.get_logger(__name__)

OV_WEIGHTS_NAME = "ov_model.xml"
ie = IECore()


def load_ov_model_from_pytorch(model):
    import io

    buf = io.BytesIO()
    dummy_input_ids = torch.randint(0, 255, (1, 11))
    dummy_mask = torch.randint(0, 255, (1, 11))
    if model.config.model_type == "gpt2":
        if model.config.use_cache:
            raise NotImplementedError("GPT2 model with use_cache=True is not implemented for OpenVINO backend")

        inputs = (dummy_input_ids, None, dummy_mask)
    else:
        inputs = (dummy_input_ids, dummy_mask)

    if model.__class__.__name__.endswith("ForQuestionAnswering"):
        outputs = ["output_s", "output_e"]
    else:
        outputs = ["output"]

    with torch.no_grad():
        torch.onnx.export(
            model, inputs, buf, input_names=["input_ids", "attention_mask"], output_names=outputs, opset_version=11
        )

    net = ie.read_network(buf.getvalue(), b"", init_from_buffer=True)
    return OVPreTrainedModel(net, model.config)


def load_ov_model_from_tf(model):
    import subprocess
    import sys

    model.save("keras_model", signatures=model.serving)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mo",
            "--saved_model_dir=keras_model",
            "--model_name=model",
            "--input",
            "input_ids,attention_mask",
            "--input_shape",
            "[1, 11], [1, 11]",
            "--disable_nhwc_to_nchw",
        ],
        check=True,
    )
    net = ie.read_network("model.xml")
    return OVPreTrainedModel(net, model.config)


def load_ov_model_from_ir(xml_path, bin_path):
    if not xml_path.endswith(".xml"):
        import shutil

        shutil.copyfile(xml_path, xml_path + ".xml")
        xml_path += ".xml"

    net = ie.read_network(xml_path, bin_path)
    return OVPreTrainedModel(net)


class OVPreTrainedModel(GenerationMixin):
    _pt_auto_model = None
    _tf_auto_model = None

    def __init__(self, net, config=None):
        super().__init__()
        self.net = net
        self.exec_net = None
        self.config = config
        self.max_length = 0
        self.ov_config = {}
        self.ov_device = "CPU"
        self.device = torch.device("cpu")

    @classmethod
    def from_pretrained(cls, model_name_or_path, *model_args, **kwargs):
        cache_dir = kwargs.pop("cache_dir", None)
        from_pt = kwargs.pop("from_pt", False)
        from_tf = kwargs.pop("from_tf", False)
        from_ov = kwargs.pop("from_ov", not (from_pt | from_tf))
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        if from_pt:
            model = cls._pt_auto_model.from_pretrained(model_name_or_path, *model_args, **kwargs)
            return load_ov_model_from_pytorch(model)
        elif from_tf:
            model = cls._tf_auto_model.from_pretrained(model_name_or_path, *model_args, **kwargs)
            return load_ov_model_from_tf(model)

        user_agent = {"file_type": "model", "framework": "openvino", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        # Load model
        OV_BIN_NAME = OV_WEIGHTS_NAME.replace(".xml", ".bin")
        if model_name_or_path is not None:
            if os.path.isdir(model_name_or_path):
                if (
                    from_ov
                    and os.path.isfile(os.path.join(model_name_or_path, OV_WEIGHTS_NAME))
                    and os.path.isfile(os.path.join(model_name_or_path, OV_BIN_NAME))
                ):
                    # Load from an OpenVINO IR
                    archive_files = [os.path.join(model_name_or_path, name) for name in [OV_WEIGHTS_NAME, OV_BIN_NAME]]
                else:
                    raise EnvironmentError(
                        f"Error no files named {[OV_WEIGHTS_NAME, OV_BIN_NAME]} found in directory "
                        f"{model_name_or_path} or `from_ov` set to False"
                    )
            # elif os.path.isfile(model_name_or_path) or is_remote_url(model_name_or_path):
            #     archive_file = model_name_or_path
            else:
                names = [OV_WEIGHTS_NAME, OV_BIN_NAME]
                archive_files = [
                    hf_bucket_url(
                        model_name_or_path,
                        filename=name,
                        revision=revision,
                    )
                    for name in names
                ]

            # redirect to the cache, if necessary
            try:
                resolved_archive_files = [
                    cached_path(
                        archive_file,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        user_agent=user_agent,
                    )
                    for archive_file in archive_files
                ]
            except EnvironmentError as err:
                logger.error(err)
                name = model_name_or_path
                msg = (
                    f"Can't load weights for '{name}'. Make sure that:\n\n"
                    f"- '{name}' is a correct model identifier listed on 'https://huggingface.co/models'\n"
                    f"  (make sure '{name}' is not a path to a local directory with something else, in that case)\n\n"
                    f"- or '{name}' is the correct path to a directory containing a file named {OV_WEIGHTS_NAME}.\n\n"
                )
                raise EnvironmentError(msg)

            if resolved_archive_files == archive_files:
                logger.info(f"loading weights file {archive_files}")
            else:
                logger.info(f"loading weights file {archive_files} from cache at {resolved_archive_files}")
        else:
            resolved_archive_files = None

        return load_ov_model_from_ir(*resolved_archive_files)

    def save_pretrained(
        self,
        save_directory,
        **kwargs,
    ):
        """
        Save model in OpenVINO IR format into a directory
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        self.net.serialize(os.path.join(save_directory, OV_WEIGHTS_NAME))

    def to(self, device):
        self.ov_device = device

    def set_config(self, config):
        self.ov_config = config

    def _load_network(self):
        self.exec_net = ie.load_network(self.net, self.ov_device, self.ov_config)

    def __call__(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        use_cache=False,
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        if attention_mask is None:
            attention_mask = np.ones_like(input_ids)

        batch_size, inp_length = input_ids.shape
        if inp_length < self.max_length:
            pad = ((0, 0), (0, self.max_length - inp_length))
            input_ids = np.pad(input_ids, pad)
            attention_mask = np.pad(attention_mask, pad)

        inputs_info = self.net.input_info
        if list(inputs_info["input_ids"].input_data.shape) != list(input_ids.shape):
            shapes = {key: input_ids.shape for key in inputs_info}
            self.net.reshape(shapes)
            self.exec_net = None

        if self.exec_net is None:
            self._load_network()

        outs = self.exec_net.infer(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )

        logits = outs["output"] if "output" in outs else next(iter(outs.values()))

        # Trunc padded values
        if inp_length != logits.shape[1]:
            logits = logits[:, :inp_length]

        if return_dict:
            result = ModelOutput()
            result["logits"] = torch.tensor(logits)
        elif "output_s" in outs and "output_e" in outs:
            result = QuestionAnsweringModelOutput()
            result["start_logits"] = outs["output_s"]
            result["end_logits"] = outs["output_e"]
        else:
            result = [logits]
        return result

    def generate(self, input_ids, *args, **kwargs):
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)

        max_length = kwargs.get("max_length", None)
        self.max_length = max_length if max_length is not None else self.config.max_length
        self.max_length -= 1

        return super().generate(input_ids, *args, **kwargs)
