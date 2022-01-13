import os
import glob
import yaml
import logging

from .quantize_utils import OVQADataLoader, OVQAAccuracyMetric

from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights
from compression.pipeline.initializer import create_pipeline

from addict import Dict

LOGGER = logging.getLogger(__name__)

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

def get_data_loader_cls(config):
    if config.task_type == "QuestionAnswering":
        return OVQADataLoader(config)

def get_metric_cls(config):
    if config.task_type == "QuestionAnswering":
        return OVQAAccuracyMetric(config)

class OVAutoQuantizer():
    def __init__(self, model_name: str, pretrained_model, config_path):
        self.config = Dict(self.parse_config(config_path))
        self.config.model_name = model_name 
        self.pretrained_model =  pretrained_model
        self.ir_path = None
        self.engine_config = Dict({"device": "CPU"})
        self.algorithms = [Dict(self.config.quantization_algorithm)]
        
    def parse_config(self, cfg_path):
        try:
            with open(cfg_path) as stream:
                config = yaml.safe_load(stream)
        except yaml.YAMLError as err:
            LOGGER.error(err)      
        
        return config


    def quantize(self):
        # Generate the OV IR if not already present
        if os.path.isfile(glob.glob(os.path.join(self.config.model.model_ir_path, '*.xml'))[-1]):
            self.ir_path = glob.glob(os.path.join(self.config.model.model_ir_path, '*.xml'))[-1]
            LOGGER.warning(f'Using existing IR {self.ir_path}')
        else:
            pretrained_model.save_pretrained(self.config.model.model_ir_path)

        model_config = Dict({"model_name": self.config.model_name, "model": self.ir_path, "weights": self.ir_path.replace(".xml", ".bin")})

        # Step 1: Load the model.
        model = load_model(model_config)

        # Step 2: Initialize the data loader.
        data_loader = get_data_loader_cls(self.config)
        
        # Step 3 (Optional. Required for AccuracyAwareQuantization): Initialize the metric.
        metric = get_metric_cls(self.config)

        # Step 4: Initialize the engine for metric calculation and statistics collection.
        engine = IEEngine(config=self.engine_config,
                          data_loader=data_loader,
                          metric=metric)

        # Step 5: Create a pipeline of compression algorithms.
        pipeline = create_pipeline(self.algorithms, engine)

        metric_results_FP32 = pipeline.evaluate(model)

        # print metric value
        if metric_results_FP32:
            for name, value in metric_results_FP32.items():
                print(bcolors.OKGREEN + "{: <27s} FP32: {}".format(name, value) + bcolors.ENDC)


        # Step 6: Execute the pipeline.
        compressed_model = pipeline.run(model)

        # Step 7 (Optional): Compress model weights to quantized precision
        #                    in order to reduce the size of final .bin file.
        compress_model_weights(compressed_model)

        # Step 8: Save the compressed model to the desired path.
        save_model(compressed_model, os.path.join(os.path.curdir, 'optimized'))

        # Step 9 (Optional): Evaluate the compressed model. Print the results.
        metric_results_INT8 = pipeline.evaluate(compressed_model)

        print(bcolors.BOLD + "\nFINAL RESULTS" + bcolors.ENDC)

        if metric_results_INT8:
            for name, value in metric_results_INT8.items():
                print(bcolors.OKGREEN + "{: <27s} INT8: {}".format(name, value) + bcolors.ENDC)

        print(bcolors.HEADER + "Saved optimized model to: {}".format(os.path.join(os.path.curdir, 'optimized')) + bcolors.ENDC)

        return os.path.join(os.path.curdir, 'optimized')
