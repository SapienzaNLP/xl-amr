"""
The ``predict`` subcommand allows you to make bulk JSON-to-JSON
or dataset to JSON predictions using a trained model and its
:class:`~allennlp.service.predictors.predictor.Predictor` wrapper.

.. code-block:: bash

    $ allennlp predict -h
    usage: allennlp predict [-h] [--output-file OUTPUT_FILE]
                            [--weights-file WEIGHTS_FILE]
                            [--batch-size BATCH_SIZE] [--silent]
                            [--cuda-device CUDA_DEVICE] [--use-dataset-reader]
                            [-o OVERRIDES] [--predictor PREDICTOR]
                            [--include-package INCLUDE_PACKAGE]
                            archive_file input_file

    Run the specified model against a JSON-lines input file.

    positional arguments:
    archive_file          the archived model to make predictions with
    input_file            path to input file

    optional arguments:
    -h, --help              show this help message and exit
    --output-file OUTPUT_FILE
                            path to output file
    --weights-file WEIGHTS_FILE
                            a path that overrides which weights file to use
    --batch-size BATCH_SIZE The batch size to use for processing
    --silent                do not print output to stdout
    --cuda-device CUDA_DEVICE
                            id of GPU to use (if any)
    --use-dataset-reader    Whether to use the dataset reader of the original
                            model to load Instances
    -o OVERRIDES, --overrides OVERRIDES
                            a JSON structure used to override the experiment
                            configuration
    --predictor PREDICTOR   optionally specify a specific predictor to use
    --include-package INCLUDE_PACKAGE
                            additional packages to include
"""
from typing import List, Iterator, Optional
import argparse
import sys
import json
import os
import torch

from xlamr_stog.commands.subcommand import Subcommand
from xlamr_stog.utils.checks import check_for_gpu, ConfigurationError
from xlamr_stog.utils import lazy_groups_of
from xlamr_stog.predictors.predictor import Predictor, JsonDict
from xlamr_stog.predictors import STOGPredictor
from xlamr_stog.data import Instance

class Predict(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Run the specified model against a JSON-lines input file.'''
        subparser = parser.add_parser(
                name, description=description, help='Use a trained model to make predictions.')

        subparser.add_argument('--archive-file', required=True, type=str, help='the archived model to make predictions with')
        subparser.add_argument('--input-file', type=str, help='path to input file')

        subparser.add_argument('--output-file', type=str, help='path to output file')
        subparser.add_argument('--weights-file',
                               type=str,
                               help='a path that overrides which weights file to use')

        batch_size = subparser.add_mutually_exclusive_group(required=False)
        batch_size.add_argument('--batch-size', type=int, default=1, help='The batch size to use for processing')

        subparser.add_argument('--silent', action='store_true', help='do not print output to stdout')

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')

        subparser.add_argument('--use-dataset-reader',
                               action='store_true',
                               help='Whether to use the dataset reader of the original model to load Instances')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')

        subparser.set_defaults(func=_predict)

        return subparser


def _get_predictor(args: argparse.Namespace) -> Predictor:
    from xlamr_stog.utils.archival import load_archive
    # check_for_gpu(args.cuda_device)
    archive = load_archive(args.archive_file,
                           device=args.cuda_device,
                           weights_file=args.weights_file)
    print("Loaded model weights.")
    return Predictor.from_archive(archive)


class _PredictManager:

    def __init__(self,
                 predictor: Predictor,
                 input_file,
                 output_file: Optional[str],
                 batch_size: int,
                 print_to_console: bool,
                 has_dataset_reader: bool,
                 beam_size: int) -> None:

        self._predictor = predictor
        self._input_file = input_file
        if output_file is not None:
            self._output_file = open(output_file, "w")
            # self._output_file_seq = open(output_file+'.bos', "w")
            self._output_file_seq = None
        else:
            self._output_file = None
            self._output_file_seq=None
        self._batch_size = batch_size
        self._print_to_console = print_to_console
        # self.universal_postags = universal_postags
        # self.source_copy = source_copy

        if has_dataset_reader:
            self._dataset_reader = predictor._dataset_reader # pylint: disable=protected-access
            self._dataset_reader.universal_postags = self._predictor._model.universal_postags
            self._dataset_reader.source_copy = self._predictor._model.generator_source_copy
            self._dataset_reader.translation_mapping = self._predictor._model.translation_mapping
            self._dataset_reader.multilingual = self._predictor._model.multilingual
            self._dataset_reader.extra_check = self._predictor._model.test_config.get('extra_check',False)
        else:
            self._dataset_reader = None

        # TODO: there should be better ways to do this
        if type(predictor) in (STOGPredictor,):
            self.beam_size = beam_size
            self._predictor._model.set_beam_size(self.beam_size)
            self._predictor._model.set_decoder_token_indexers(self._dataset_reader._token_indexers)

    def _predict_json(self, batch_data: List[JsonDict]) -> Iterator[str]:
        if len(batch_data) == 1:
            results = [self._predictor.predict_json(batch_data[0])]
        else:
            results = self._predictor.predict_batch_json(batch_data)
        for output in results:
            yield self._predictor.dump_line(output)

    def _predict_instances(self, batch_data: List[Instance]):# -> Iterator[str]:
        if len(batch_data) == 1:
            pred_results = self._predictor.predict_instance(batch_data[0])
            results = [pred_results[0]]
            encoder_last_state_seq = [pred_results[1]]
        else:
            results, encoder_last_state_seq = self._predictor.predict_batch_instance(batch_data)
        for i, output in enumerate(results):
            yield self._predictor.dump_line(output), encoder_last_state_seq[i]

    def _maybe_print_to_console_and_file(self,
                                         prediction: str,
                                         model_input: str = None) -> None:
        if self._print_to_console:
            if model_input is not None:
                print("input: ", model_input)
            print("prediction: ", prediction)
        if self._output_file is not None:
            self._output_file.write(prediction)

    def _maybe_print_to_console_and_file_seq(self,prediction) -> None:
            if self._output_file_seq is not None:
                self._output_file_seq.write(" ".join([str(x) for x in prediction.tolist()]))
                self._output_file_seq.write("\n")

    def _get_json_data(self) -> Iterator[JsonDict]:
        if self._input_file == "-":
            for line in sys.stdin:
                if not line.isspace():
                    yield self._predictor.load_line(line)
        else:
            with open(self._input_file, "r") as file_input:
                for line in file_input:
                    if not line.isspace():
                        yield self._predictor.load_line(line)

    def _get_instance_data(self) -> Iterator[Instance]:
        if self._input_file == "-":
            raise ConfigurationError("stdin is not an option when using a DatasetReader.")
        elif self._dataset_reader is None:
            raise ConfigurationError("To generate instances directly, pass a DatasetReader.")
        else:
            yield from self._dataset_reader.read(self._input_file)

    def run(self) -> None:
        has_reader = self._dataset_reader is not None
        if has_reader:
            for batch in lazy_groups_of(self._get_instance_data(), self._batch_size):
                for model_input_instance, result in zip(batch, self._predict_instances(batch)):
                    self._maybe_print_to_console_and_file(result[0], str(model_input_instance))
                    self._maybe_print_to_console_and_file_seq(result[1])
        else:
            for batch_json in lazy_groups_of(self._get_json_data(), self._batch_size):
                for model_input_json, result in zip(batch_json, self._predict_json(batch_json)):
                    self._maybe_print_to_console_and_file(result, json.dumps(model_input_json))

        if self._output_file is not None:
            self._output_file.close()


def _predict(args: argparse.Namespace) -> None:
    predictor = _get_predictor(args)

    if args.silent and not args.output_file:
        print("--silent specified without --output-file.")
        print("Exiting early because no output will be created.")
        sys.exit(0)

    manager = _PredictManager(predictor,
                              args.input_file,
                              args.output_file,
                              args.batch_size,
                              not args.silent,
                              args.use_dataset_reader,
                              args.beam_size)
    manager.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Use a trained model to make predictions.')

    parser.add_argument('--archive-file', required=True, type=str, help='the archived model to make predictions with')
    parser.add_argument('--input-file', nargs="+", required=True, type=str, help='path to input file')

    parser.add_argument('--output-file', type=str, help='path to output file')
    parser.add_argument('--weights-file', type=str, help='a path that overrides which weights file to use')

    parser.add_argument('--batch-size', type=int, default=1, help='The batch size to use for processing')

    parser.add_argument('--silent', action='store_true', help='do not print output to stdout')

    parser.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')

    parser.add_argument('--use-dataset-reader',
                           action='store_true',
                           help='Whether to use the dataset reader of the original model to load Instances')

    parser.add_argument('-o', '--overrides',
                           type=str,
                           default="",
                           help='a JSON structure used to override the experiment configuration')

    parser.add_argument('--predictor',
                           type=str,
                           help='optionally specify a specific predictor to use')

    parser.add_argument('--beam-size',
                        type=int,
                        default=1,
                        help="Beam size for seq2seq decoding")

    args = parser.parse_args()

    if args.cuda_device >= 0:
        device = torch.device('cuda:{}'.format(args.cuda_device))
    else:
        device = torch.device('cpu')
    args.cuda_device = device

    if not os.path.exists(os.path.join(args.archive_file, "test_output")):
        os.makedirs(os.path.join(args.archive_file, "test_output"))
    args.input_file = [tuple(args.input_file)]

    _predict(args)

