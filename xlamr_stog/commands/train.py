import os
import re
import yaml
import torch
import json
import warnings
import argparse
from xlamr_stog.utils import logging
from xlamr_stog.utils.params import Params, remove_pretrained_embedding_params
from xlamr_stog import models as Models
from xlamr_stog.data.dataset_builder import dataset_from_params, iterator_from_params
from xlamr_stog.data.vocabulary import Vocabulary
from xlamr_stog.training.trainer import Trainer
from xlamr_stog.utils import environment
from xlamr_stog.utils.checks import ConfigurationError
from xlamr_stog.utils.archival import CONFIG_NAME, _DEFAULT_WEIGHTS, archive_model
from xlamr_stog.commands.evaluate import evaluate
from xlamr_stog.metrics import dump_metrics

logger = logging.init_logger()
if not "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")

def serialize_sets(obj):
    if isinstance(obj, set):
        return list(obj)

    return obj

def create_serialization_dir(params: Params) -> None:
    """
    This function creates the serialization directory if it doesn't exist.  If it already exists
    and is non-empty, then it verifies that we're recovering from a training with an identical configuration.
    Parameters
    ----------
    params: ``Params``
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir: ``str``
        The directory in which to save results and logs.
    recover: ``bool``
        If ``True``, we will try to recover from an existing serialization directory, and crash if
        the directory doesn't exist, or doesn't match the configuration we're given.
    """
    serialization_dir = params['environment']['serialization_dir']
    recover = params['environment']['recover']
    if os.path.exists(serialization_dir) and os.listdir(serialization_dir):
        if not recover:
            raise ConfigurationError(f"Serialization directory ({serialization_dir}) already exists and is "
                                     f"not empty. Specify --recover to recover training from existing output.")

        logger.info(f"Recovering from prior training at {serialization_dir}.")

        recovered_config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(recovered_config_file):
            raise ConfigurationError("The serialization directory already exists but doesn't "
                                     "contain a config.json. You probably gave the wrong directory.")
        else:
            loaded_params = Params.from_file(recovered_config_file)

            if params != loaded_params:
                raise ConfigurationError("Training configuration does not match the configuration we're "
                                         "recovering from.")

            # In the recover mode, we don't need to reload the pre-trained embeddings.
            remove_pretrained_embedding_params(params)
    else:
        if recover:
            raise ConfigurationError(f"--recover specified but serialization_dir ({serialization_dir}) "
                                     "does not exist.  There is nothing to recover from.")
        os.makedirs(serialization_dir, exist_ok=True)
        params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        torch.nn.init.xavier_uniform_(m.weight.data)

def train_model(params: Params, build_vocab_only:bool=False, vocab_dir:str=None):
    """
    Trains the model specified in the given :class:`Params` object, using the data and training
    parameters also specified in that object, and saves the results.
    Parameters
    ----------
    params : ``Params``
        A parameter object specifying an AllenNLP Experiment.
    Returns
    -------
    best_model: ``Model``
        The model with the best epoch weights.
    """
    # Set up the environment.
    environment_params = params['environment']
    environment.set_seed(environment_params)
    if not build_vocab_only:
        create_serialization_dir(params)
    environment.prepare_global_logging(environment_params)
    environment.check_for_gpu(environment_params)
    if environment_params['gpu']:
        device = torch.device('cuda:{}'.format(environment_params['cuda_device']))
        print(torch.cuda.get_device_name(environment_params['cuda_device']))
        environment.occupy_gpu(device)
    else:
        device = torch.device('cpu')
    params['trainer']['device'] = device

    # Load data.
    data_params = params['data']
    dataset = dataset_from_params(data_params,
                                  universal_postags=params["model"].get('universal_postags',False),
                                  generator_source_copy=data_params.get('source_copy', True),
                                  multilingual=params['model'].get('multilingual',False),
                                  extra_check=params['data'].get('extra_check',False))
    train_data = dataset['train']
    dev_data = dataset.get('dev')
    test_data = dataset.get('test')
    train_mappings = dataset.get('train_mappings',None)
    train_replacements = dataset.get('train_replacements',None)

    # Vocabulary and iterator are created here.
    vocab_params = params.get('vocab', {})
    if "fixed_vocab" in vocab_params and vocab_params["fixed_vocab"]:
        vocab = Vocabulary.from_files("data/vocabulary")
    else:
        vocab = Vocabulary.from_instances(instances=train_data, **vocab_params)

    # Initializing the model can have side effect of expanding the vocabulary
    if not build_vocab_only:
        vocab_dir = os.path.join(environment_params['serialization_dir'], "vocabulary")

    vocab.save_to_files(vocab_dir)
    train_iterator, dev_iterater, test_iterater = iterator_from_params(vocab, data_params['iterator'])
    if build_vocab_only:
        return

    if train_mappings is not None and train_replacements is not None:
        with open(os.path.join(environment_params['serialization_dir'],"trns_lex_missing.json"),"w", encoding='utf-8') as outfile:
            json.dump(train_mappings[-1], outfile, indent=4, default=serialize_sets)
        with open(os.path.join(environment_params['serialization_dir'],"trns_lexicalizations.json"),"w", encoding='utf-8') as outfile:
            json.dump(train_mappings[-2], outfile, indent=4, default=serialize_sets)
        with open(os.path.join(environment_params['serialization_dir'],"trns_rep.json"), "w", encoding='utf-8') as outfile:
            json.dump(train_replacements, outfile, indent=4, default=serialize_sets)
    
    # Build the model.
    model_params = params['model']
    model = getattr(Models, model_params['model_type']).from_params(vocab, model_params, environment_params['gpu'], train_mappings, train_replacements)
    logger.info(model)

    # Train
    trainer_params = params['trainer']
    no_grad_regexes = trainer_params['no_grad']
    for name, parameter in model.named_parameters():
        if any(re.search(regex, name) for regex in no_grad_regexes):
            parameter.requires_grad_(False)

    frozen_parameter_names, tunable_parameter_names = \
        environment.get_frozen_and_tunable_parameter_names(model)
    logger.info("Following parameters are Frozen  (without gradient):")
    for name in frozen_parameter_names:
        logger.info(name)
    logger.info("Following parameters are Tunable (with gradient):")
    for name in tunable_parameter_names:
        logger.info(name)

    logger.info("Total nr of parameters Tunable (with gradient):")
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(pytorch_total_params)

    trainer = Trainer.from_params(model, train_data, dev_data, train_iterator, dev_iterater, trainer_params)

    serialization_dir = trainer_params['serialization_dir']
    try:
        metrics = trainer.train()
    except KeyboardInterrupt:
        # if we have completed an epoch, try to create a model archive.
        if os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)):
            logger.info("Training interrupted by the user. Attempting to create "
                         "a model archive using the current best epoch weights.")
            archive_model(serialization_dir)
        raise

    # Now tar up results
    archive_model(serialization_dir)

    logger.info("Loading the best epoch weights.")
    best_model_state_path = os.path.join(serialization_dir, 'best.th')
    best_model_state = torch.load(best_model_state_path)
    best_model = model
    if not isinstance(best_model, torch.nn.DataParallel):
        best_model_state = {re.sub(r'^module\.', '', k):v for k, v in best_model_state.items()}
    best_model.load_state_dict(best_model_state)

    return best_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser('train.py')
    parser.add_argument('params', help='Parameters YAML file.')
    parser.add_argument('--build_vocab_only', action='store_true')
    parser.add_argument('--vocab_dir', default="data/vocabulary")
    args = parser.parse_args()

    params = Params.from_file(args.params)
    logger.info(params)

    train_model(
        params, 
        build_vocab_only=args.build_vocab_only,
        vocab_dir=args.vocab_dir)
