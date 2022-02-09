import unittest
import logging

import torch
from transformers import AutoTokenizer

from optimum.intel.openvino import OVAutoModelForQuestionAnswering
from optimum.intel.pot import OVAutoQuantizer

LOGGER = logging.getLogger(__name__)

class OVBertForQuestionAnsweringTest(unittest.TestCase):
    def check_model(self, model, tok):
        context = """
        Soon her eye fell on a little glass box that
        was lying under the table: she opened it, and
        found in it a very small cake, on which the
        words “EAT ME” were beautifully marked in
        currants. “Well, I’ll eat it,” said Alice, “ and if
        it makes me grow larger, I can reach the key ;
        and if it makes me grow smaller, I can creep
        under the door; so either way I’ll get into the
        garden, and I don’t care which happens !”
        """

        question = "Where Alice should go?"

        # For better OpenVINO efficiency it's recommended to use fixed input shape.
        # So pad input_ids up to specific max_length.
        input_ids = tok.encode(
            question + " " + tok.sep_token + " " + context, return_tensors="pt", max_length=128, padding="max_length"
        )

        outputs = model(input_ids)

        start_pos = outputs.start_logits.argmax()
        end_pos = outputs.end_logits.argmax() + 1

        answer_ids = input_ids[0, start_pos:end_pos]
        answer = tok.convert_tokens_to_string(tok.convert_ids_to_tokens(answer_ids))

        LOGGER.info(answer)

        self.assertEqual(answer, "the garden")

    def test_from_pt(self):
        torch.sqrt = lambda x: torch.pow(x, 0.5)
        tok = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        model = OVAutoModelForQuestionAnswering.from_pretrained(
            "bert-large-uncased-whole-word-masking-finetuned-squad", from_pt=True
        )
        
        int8_model_dir = OVAutoQuantizer(model, 'config.yml'
        ).quantize()

        int8_model = OVAutoModelForQuestionAnswering.from_pretrained(int8_model_dir)

        self.check_model(int8_model, tok)

if __name__ == "__main__":
    unittest.main()