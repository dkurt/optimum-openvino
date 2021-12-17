from packaging import version

from .nncf_auto import NNCFAutoConfig

import transformers
from transformers import (
    trainer,
    trainer_callback,
    training_args,
    modeling_utils,
)

__all__ = [
    "NNCFAutoConfig",
]


# This code patches Transformers methods for NNCF
# source: https://github.com/openvinotoolkit/nncf/blob/develop/third_party_integration/huggingface_transformers/0001-Modifications-for-NNCF-usage.patch
def replace_code_of_module(module, new_source):
    import ast

    code = compile(ast.parse(new_source), "<string>", "exec")
    exec(code, module.__dict__)


ADD_LINES_BEFORE = 0
ADD_LINES_AFTER = 1
REPLACE = 2


def patch_func(func, rules):
    import inspect
    import textwrap

    # Get function source code
    lines = inspect.getsourcelines(func)
    lines = [line.rstrip("\n") for line in lines[0]]

    for rule in rules:
        pattern, new_text, mode = rule

        idx = lines.index(pattern)
        if mode == ADD_LINES_BEFORE:
            lines.insert(idx, new_text)
        elif mode == ADD_LINES_AFTER:
            lines.insert(idx + 1, new_text)
        elif REPLACE:
            lines[idx] = new_text

    # Restore newline characters
    code = "".join([line + "\n" for line in lines])
    code = textwrap.dedent(code)

    replace_code_of_module(func, code)


patch_func(
    trainer_callback,
    [
        [
            "            self.training_bar.update(state.global_step - self.current_step)",
            """
            if hasattr(state, "curr_loss"):
                self.training_bar.set_postfix(loss=state.curr_loss)
""",
            ADD_LINES_AFTER,
        ],
    ],
)

patch_func(
    training_args,
    [
        [
            "    def __post_init__(self):",
            """
    nncf_config: str = field(default=None,
                             metadata={"help": "NNCF configuration .json file for compression-enabled training"})
""",
            ADD_LINES_BEFORE,
        ],
    ],
)

patch_func(
    modeling_utils,
    [
        [
            "class Conv1D(nn.Module):",
            """
import nncf
@nncf.torch.register_module()""",
            ADD_LINES_BEFORE,
        ],
    ],
)

patches = [
    [
        "from tqdm.auto import tqdm",
        """
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch import create_compressed_model
from optimum.intel.nncf import NNCFAutoConfig
""",
        ADD_LINES_BEFORE,
    ],
    [
        "import torch",
        "from nncf.common.utils.tensorboard import prepare_for_tensorboard",
        ADD_LINES_AFTER,
    ],
    [
        "        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),",
        "        nncf_config: NNCFAutoConfig = None,",
        ADD_LINES_AFTER,
    ],
    [
        "        self.args = args",
        """
        if nncf_config is not None:
            nncf_config.auto_register_extra_structs(args, train_dataset, data_collator)
            # TODO: restore compression state
            # compression_state_file = os.path.join(model_name_or_path, NNCF_PT_STATE_NAME)
            # if os.path.isfile(compression_state_file):
            #     compression_state = torch.load(compression_state_file)
            # else:
            compression_state = None
            self.compression_ctrl, model = create_compressed_model(
                model, nncf_config, compression_state=compression_state
            )
""",
        ADD_LINES_AFTER,
    ],
    [
        "            signature = inspect.signature(self.model.forward)",
        """
            if isinstance(self.model, NNCFNetwork):
                signature = inspect.signature(self.model.get_nncf_wrapped_model().forward)
            else:
                signature = inspect.signature(self.model.forward)
""",
        REPLACE,
    ],
    [
        "            model = nn.parallel.DistributedDataParallel(",
        """
            if self.compression_ctrl is not None:
                self.compression_ctrl.distributed()
""",
        ADD_LINES_BEFORE,
    ],
    [
        "        for epoch in range(epochs_trained, num_train_epochs):",
        """
            if self.compression_ctrl is not None:
                self.compression_ctrl.scheduler.epoch_step()
                print(self.compression_ctrl.statistics().to_str())
""",
        ADD_LINES_AFTER,
    ],
    [
        "                self.current_flos += float(self.floating_point_ops(inputs))",
        "                tr_loss += curr_loss",
        ADD_LINES_BEFORE,
    ],
    [
        "                    optimizer_was_run = True",
        """
                    if self.compression_ctrl is not None:
                        self.compression_ctrl.scheduler.step()
""",
        ADD_LINES_BEFORE,
    ],
    [
        "                    self.state.epoch = epoch + (step + 1) / steps_in_epoch",
        "                    self.state.curr_loss = curr_loss.cpu().detach().item()",
        ADD_LINES_AFTER,
    ],
    [
        "            loss = loss / self.args.gradient_accumulation_steps",
        """
        if self.compression_ctrl is not None:
            compression_loss = self.compression_ctrl.loss()
            loss += compression_loss
""",
        ADD_LINES_AFTER,
    ],
    [
        "            self._total_loss_scalar += tr_loss_scalar",
        """
            if self.compression_ctrl is not None:
                logs["compression_loss"] = self.compression_ctrl.loss().item()
                compression_stats = self.compression_ctrl.statistics()
                for key, value in prepare_for_tensorboard(compression_stats).items():
                    logs["compression/statistics/{0}".format(key)] = value
                print(compression_stats.to_str())
""",
        ADD_LINES_BEFORE,
    ],
    [
        "            if isinstance(unwrap_model(self.model), PreTrainedModel):",
        """
            unwrapped_model = unwrap_model(self.model)
            if isinstance(unwrapped_model, NNCFNetwork):
                is_pretrained = isinstance(unwrapped_model.get_nncf_wrapped_model(), PreTrainedModel)
            else:
                is_pretrained = isinstance(unwrapped_model, PreTrainedModel)
            if is_pretrained:
""",
        REPLACE,
    ],
    [
        "            if isinstance(unwrap_model(self.model), PreTrainedModel):",
        """
            unwrapped_model = unwrap_model(self.model)
            if isinstance(unwrapped_model, NNCFNetwork):
                is_pretrained = isinstance(unwrapped_model.get_nncf_wrapped_model(), PreTrainedModel)
            else:
                is_pretrained = isinstance(unwrapped_model, PreTrainedModel)
            if is_pretrained:
""",
        REPLACE,
    ],
    [
        "                    state_dict = self.model.state_dict()",
        "                    state_dict = unwrapped_model.state_dict()",
        REPLACE,
    ],
    [
        "                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)",
        "                unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)",
        REPLACE,
    ],
    [
        "        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))"
        if version.parse(transformers.__version__) >= version.parse("4.11.0")
        else '        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))',
        """
        path_to_onnx = os.path.join(output_dir, "ov_model.onnx")
        self.compression_ctrl.export_model(path_to_onnx, input_names=["input_ids", "attention_mask"])

        import subprocess

        subprocess.run(
            [sys.executable, "-m", "mo", "--input_model", path_to_onnx, "--output_dir", output_dir], check=True
        )
        os.remove(path_to_onnx)
""",
        ADD_LINES_AFTER,
    ],
]

if version.parse(transformers.__version__) >= version.parse("4.11.0"):
    patches += [
        [
            "                    tr_loss_step = self.training_step(model, inputs)",
            "                curr_loss = tr_loss_step",
            ADD_LINES_AFTER,
        ]
    ]
else:
    patches += [
        [
            "                        tr_loss += self.training_step(model, inputs)",
            "                        curr_loss = self.training_step(model, inputs)",
            REPLACE,
        ],
        [
            "                    tr_loss += self.training_step(model, inputs)",
            "                    curr_loss = self.training_step(model, inputs)",
            REPLACE,
        ],
    ]

patch_func(trainer, patches)
