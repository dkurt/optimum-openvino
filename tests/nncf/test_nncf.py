import sys
import unittest
import subprocess
import json

import numpy as np

from optimum.intel.openvino import (
    OVAutoModel,
)


class NNCFBertBaseNERTest(unittest.TestCase):
    def test_quantized_model(self):
        subprocess.run(
            [
                sys.executable,
                "examples/pytorch/token-classification/run_ner.py",
                "--model_name_or_path=bert-base-cased",
                "--dataset_name=conll2003",
                "--output_dir=bert_base_cased_conll_int8",
                "--do_train",
                "--do_eval",
                "--evaluation_strategy=epoch",
                "--nncf_config=nncf_bert_config_conll.json",
                "--num_train_epochs=1",
                "--max_train_samples=1000",
                "--max_eval_samples=100",
            ],
            check=True,
        )

        with open("bert_base_cased_conll_int8/all_results.json", "rt") as f:
            logs = json.loads(f.read())
            self.assertGreaterEqual(logs["eval_accuracy"], 0.937)
            self.assertGreaterEqual(logs["eval_precision"], 0.66)
            self.assertGreaterEqual(logs["eval_recall"], 0.66)

        model = OVAutoModel.from_pretrained("bert_base_cased_conll_int8")
        input_ids = np.random.randint(0, 256, [1, 128])
        attention_mask = np.random.randint(0, 2, [1, 128])

        expected_shape = (1, 128, 9)
        output = model(input_ids, attention_mask=attention_mask)[0]
        self.assertEqual(output.shape, expected_shape)
