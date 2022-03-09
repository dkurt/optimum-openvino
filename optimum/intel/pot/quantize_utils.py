import numpy as np

from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers import AutoTokenizer

from compression.api import DataLoader, Metric

from addict import Dict

from datasets import load_dataset, load_metric

class bcolors:
    """
    Just gives us some colors for the text
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class OVQADataLoader(DataLoader):

    # Required methods
    def __init__(self, config):
        """Constructor
        :param config: data loader specific config
        """
        if not isinstance(config, Dict):
            config = Dict(config)
        super().__init__(config)
        self.task = config.dataset.task.lower()
        self.batch_size = config.dataset.batch_size
        self.max_length = config.dataset.max_length
        self.model_name = config.model_name
        self.calib_size = config.dataset.calib_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.features = []
        self.labels = []
        self.prepare_dataset()
        self.items = np.arange(self.calib_size)

        print(bcolors.UNDERLINE + "\nQuantizing FP32 OpenVINO model to INT8\n" + bcolors.ENDC)

        print(bcolors.OKBLUE + "There are {:,} samples in the calibration dataset ".format(len(self.items)) + \
            bcolors.OKGREEN + bcolors.ENDC)

    def __len__(self):
        """Returns size of the dataset"""
        return len(self.items)

    def get_calib_features(self, dataset):
        return dataset.map(
                self.preprocess_function,
                batched=True
            )

    def preprocess_function(self, examples):
        example_questions = [q.strip() for q in examples["question"]]
        example_answers = examples["answers"]
        tokenized_inputs = self.tokenizer(
            example_questions,
            examples["context"],
            max_length=self.max_length,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        for i in range(len(tokenized_inputs["input_ids"])):
            seq_ids = tokenized_inputs.sequence_ids(i)
            ctx_start_index = 0
            ctx_end_index = 0
            # Get context ids, but not including padding
            for j, x in enumerate(seq_ids):
                if x == 1:
                    ctx_start_index = j
                    break
            for j, x in enumerate(seq_ids[ctx_start_index:]):
                if x != 1:
                    ctx_end_index = j
                break

            for k, offset in enumerate(tokenized_inputs["offset_mapping"][i]):
                # Only include the offset mapping if it is within the valid context window
                if k > ctx_start_index and k < ctx_end_index:
                    tokenized_inputs["offset_mapping"][i] = offset
                else:
                    tokenized_inputs["offset_mapping"][i] = (0, 0)

        tokenized_inputs["id"] = examples["id"]
        return tokenized_inputs

    def __getitem__(self, item):
        """
        Returns annotation, data and metadata at the specified index.
        Possible formats:
        (index, annotation), data
        (index, annotation), data, metadata
        """
        if item >= len(self):
            raise IndexError

        input_ids = self.features[self.items[item]]["input_ids"]
        attention_mask = self.features[self.items[item]]["attention_mask"]

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}       
        label = self.labels[self.items[item]]
        return (item, label), inputs

    def prepare_dataset(self):
        """Prepare dataset"""
        # Dataset loading
        self.dataset = load_dataset(self.task, split="validation")

        calibration_features = self.get_calib_features(self.dataset)

        for i, feature in enumerate(calibration_features):
            self.features.append({'input_ids': feature["input_ids"], 'attention_mask': feature["attention_mask"]})
            self.labels.append({'input_ids': feature["input_ids"], 'answer': feature["answers"], 'id': feature["id"]})

class OVQAAccuracyMetric(Metric):
    # Required methods
    def __init__(self, config):
        super().__init__()
        self.metric_scores = []
        self.task = config.dataset.task.lower()
        self.max_length = config.dataset.max_length
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        self.metric = load_metric(self.task)
        self.metric_name = "f1"
        
    @property
    def value(self):
        """Returns accuracy metric value for the last model output."""
        return {self.metric_name: self.metric_scores[-1]}

    @property
    def avg_value(self):
        """Returns accuracy metric value for all model outputs."""
        return {self.metric_name: np.ravel(self.metric_scores).mean()}

    def postprocess_function(self, output, target):
        result = QuestionAnsweringModelOutput()
        result["start_logits"] = output["output_s"]
        result["end_logits"] = output["output_e"]
        outputs = result

        start_pos = outputs.start_logits.argmax()
        end_pos = outputs.end_logits.argmax() + 1

        target_id = target[0]['id']
        input_ids = target[0]['input_ids']
        target_answers = target[0]['answer']

        answer_ids = np.array(input_ids)[start_pos:end_pos]
        pred_answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(answer_ids))

        predictions = [{'prediction_text': pred_answer, 'id': target_id}]
        references = [{'answers': target_answers, 'id': target_id}]

        return predictions, references

    def update(self, output, target):
        """
        Updates prediction matches.

        :param output: model output
        :param target: annotations
        """
        predictions, references = self.postprocess_function(output, target)
        final_score = self.metric.compute(predictions=predictions, references=references)
        self.metric_scores.append(final_score[self.metric_name])

    def reset(self):
        """
        Resets collected matches
        """
        self.metric_scores = []

    def get_attributes(self):
        """
        Returns a dictionary of metric attributes {metric_name: {attribute_name: value}}.
        Required attributes: 'direction': 'higher-better' or 'higher-worse'
                             'type': metric type
        """
        return {self.metric_name: {"direction": "higher-better", "type": "accuracy"}}

    

    