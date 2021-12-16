from .nncf_auto import NNCFAutoConfig

from transformers import (
    trainer,
    trainer_callback,
    training_args,
    modeling_utils,
)

import os

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


with open(os.path.join(os.path.dirname(__file__), "..", "nncf", "trainer.py"), "rt") as f:
    code = f.read()

replace_code_of_module(trainer, code)

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
