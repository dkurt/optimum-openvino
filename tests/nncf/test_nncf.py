import sys
import unittest
import subprocess
import json
from packaging import version

import numpy as np

from optimum.intel.openvino import (
    OVAutoModel,
)
import transformers


class NNCFTests(unittest.TestCase):
    # def test_bert_base_ner(self):
    #     subprocess.run(
    #         [
    #             sys.executable,
    #             "examples/pytorch/token-classification/run_ner.py",
    #             "--model_name_or_path=bert-base-cased",
    #             "--dataset_name=conll2003",
    #             "--output_dir=bert_base_cased_conll_int8",
    #             "--do_train",
    #             "--do_eval",
    #             "--evaluation_strategy=epoch",
    #             "--nncf_config=nncf_bert_config_conll.json",
    #             "--num_train_epochs=1",
    #             "--max_train_samples=1000",
    #             "--max_eval_samples=100",
    #         ],
    #         check=True,
    #     )

    #     with open("bert_base_cased_conll_int8/all_results.json", "rt") as f:
    #         logs = json.loads(f.read())
    #         self.assertGreaterEqual(logs["eval_accuracy"], 0.93)
    #         self.assertGreaterEqual(logs["eval_precision"], 0.66)
    #         self.assertGreaterEqual(logs["eval_recall"], 0.66)

    #     model = OVAutoModel.from_pretrained("bert_base_cased_conll_int8")
    #     input_ids = np.random.randint(0, 256, [1, 128])
    #     attention_mask = np.random.randint(0, 2, [1, 128])

    #     expected_shape = (1, 128, 9)
    #     output = model(input_ids, attention_mask=attention_mask)[0]
    #     self.assertEqual(output.shape, expected_shape)

    def test_wav2vec2_audio_classification(self):
        if version.parse(transformers.__version__) < version.parse("4.15.0"):
            return unittest.skip("Test is supported starts from Transformers 4.15.0")

        subprocess.run(
            [
                sys.executable,
                "examples/pytorch/audio-classification/run_audio_classification.py",
                "--model_name_or_path=anton-l/wav2vec2-base-ft-keyword-spotting",
                "--dataset_name=superb",
                "--dataset_config_name=ks",
                "--output_dir=wav2vec2-base-ft-keyword-spotting",
                "--overwrite_output_dir",
                "--remove_unused_columns=False",
                "--do_eval",
                "--do_train",
                "--nncf_config=nncf_wav2vec2_config.json",
                "--learning_rate=3e-5",
                "--max_length_seconds=1",
                "--attention_mask=False",
                "--warmup_ratio=0.1",
                "--num_train_epochs=1",
                "--max_train_samples=500",
                "--max_eval_samples=500",
                "--per_device_train_batch_size=32",
                "--gradient_accumulation_steps=4",
                "--per_device_eval_batch_size=32",
                "--dataloader_num_workers=4",
                "--logging_strategy=steps",
                "--logging_steps=10",
                "--evaluation_strategy=epoch",
                "--save_strategy=epoch",
                "--load_best_model_at_end=True",
                "--metric_for_best_model=accuracy",
                "--save_total_limit=3",
                "--seed=0",
            ],
            check=True,
        )

        with open("wav2vec2-base-ft-keyword-spotting/all_results.json", "rt") as f:
            logs = json.loads(f.read())
            self.assertGreaterEqual(logs["eval_accuracy"], 0.792)
