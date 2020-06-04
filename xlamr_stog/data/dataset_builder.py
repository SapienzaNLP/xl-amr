import os
import argparse
import json
import numpy
from tqdm import tqdm
from xlamr_stog.utils.params import Params
from xlamr_stog.utils import logging
from xlamr_stog.data.iterators import BucketIterator, BasicIterator
from xlamr_stog.data.token_indexers import SingleIdTokenIndexer,TokenCharactersIndexer
from xlamr_stog.data.dataset_readers import AbstractMeaningRepresentationDatasetReader

ROOT_TOKEN="<root>"
ROOT_CHAR="<r>"
logger = logging.init_logger()


def load_dataset_reader(dataset_type, universal_postags=False, generator_source_copy=True, multilingual=False, translation_mapping=None, tgt_src_replacements=None, split="test", *args, **kwargs):
    if dataset_type == "AMR":
        dataset_reader = AbstractMeaningRepresentationDatasetReader(
            token_indexers=dict(
                encoder_tokens=SingleIdTokenIndexer(namespace="encoder_token_ids"),
                encoder_characters=TokenCharactersIndexer(namespace="encoder_token_characters"),
                decoder_tokens=SingleIdTokenIndexer(namespace="decoder_token_ids"),
                decoder_characters=TokenCharactersIndexer(namespace="decoder_token_characters")
            ),
            word_splitter=kwargs.get('word_splitter', None),
            bpe_splitter=kwargs.get('bpe_splitter', None),
            bpe_codes=kwargs.get('bpe_codes', None),
            universal_postags=universal_postags,
            source_copy=generator_source_copy,
            multilingual=multilingual,
            extra_check=kwargs.get('extra_check', False),
            translation_mapping=translation_mapping,
            tgt_src_replacements=tgt_src_replacements,
            split=split
        )

    else:
        raise NotImplementedError
    return dataset_reader


def load_dataset(path, dataset_type, universal_postags, generator_source_copy, multilingual, translation_mapping, tgt_src_replacements, split="test", *args, **kwargs):
    dataset_reader=load_dataset_reader(dataset_type, universal_postags, generator_source_copy, multilingual, translation_mapping, tgt_src_replacements, split, *args, **kwargs)
    return dataset_reader.read(path), dataset_reader.translation_mapping, dataset_reader.tgt_src_replacements

import re
def embeddings_dict(emb_path):
    embeddings = dict()

    dec_tokens_to_keep = {re.sub(r'-\d\d$', '',  x.rstrip()) for x in open("vocabulary/decoder_token_ids.txt", encoding='utf-8').readlines()}
    enc_tokens_to_keep = {re.sub(r'-\d\d$', '',  x.rstrip())for x in open("vocabulary/encoder_token_ids.txt", encoding='utf-8').readlines()}

    tokens_to_keep=dec_tokens_to_keep.union(enc_tokens_to_keep)

    with open(emb_path,"r", encoding='utf-8') as infile:
        for line in tqdm(infile):
            fields = line.rstrip().split(' ')
            if fields[0] in tokens_to_keep:
                vector = numpy.asarray(fields[1:], dtype='float32')
                embeddings[fields[0]] = vector
    print("Loaded mEmbeddings")
    return embeddings

def lemmapos2synset2lemmapos(synset_en_lex, lemma2synset_path):
    lp2lp = dict()
    with open(lemma2synset_path,"r", encoding='utf-8') as infile:
        for line in infile:
            fields = line.rstrip().split("\t")
            lemmapos = fields[0].lower()
            if lemmapos not in lp2lp:
                lp2lp[lemmapos]=set()
            for synset in fields[1:]:
                if synset in synset_en_lex:
                    for lex in synset_en_lex[synset]:
                        lp2lp[lemmapos].add(lex)
    return lp2lp

def read_en_lex(lemma2synset_path):
    synset2lp = dict()
    with open(lemma2synset_path, "r", encoding='utf-8') as infile:
        for line in infile:
            fields = line.rstrip().split("\t")
            lemmapos = fields[0].lower()
            for s in fields[1:]:
                if s not in synset2lp:
                    synset2lp[s]=set()
                synset2lp[s].add(lemmapos)
    return synset2lp


def lexicalizations_dict(lexicalizations_path):
    lexicalizations=dict()
    synset2lemmapos_en = read_en_lex(lexicalizations_path.format("en"))
    lexicalizations["it"]=lemmapos2synset2lemmapos(synset2lemmapos_en,lexicalizations_path.format("it"))
    lexicalizations["es"]=lemmapos2synset2lemmapos(synset2lemmapos_en,lexicalizations_path.format("es"))
    lexicalizations["de"]=lemmapos2synset2lemmapos(synset2lemmapos_en,lexicalizations_path.format("de"))
    return lexicalizations


def dataset_from_params(params, universal_postags=False, generator_source_copy=True, multilingual=False, extra_check=False):
    assert universal_postags and multilingual
    train_data = [(l, os.path.join(params['data_dir'], p)) for l,p in params['train_data']]
    dev_data = [(l, os.path.join(params['data_dir'], p)) for l,p in params['dev_data']]
    # test_data = [(l, p) for l,p in params['test_data']]
    # train_data = os.path.join(params['data_dir'], params['train_data'])
    # dev_data = os.path.join(params['data_dir'], params['dev_data'])
    test_data = params['test_data']
    data_type = params['data_type']
    translation_mapping_path = params.get("mult_token_mapping", None)
    translation_emb_path = params.get("mult_token_embeddings", None)
    lexicalizations_path = params.get("mult_lexicalizations_path", None)
    translation_mapping = dict()
    if translation_mapping_path is not None:
        translation_mapping["it"] = json.load(open(translation_mapping_path.format("it"), "r", encoding='utf-8'))
        translation_mapping["es"] = json.load(open(translation_mapping_path.format("es"), "r", encoding='utf-8'))
        translation_mapping["de"] = json.load(open(translation_mapping_path.format("de"), "r", encoding='utf-8'))
    if translation_emb_path is not None:
        mult_emb_dict = embeddings_dict(translation_emb_path)
    else:
        mult_emb_dict = None
    if lexicalizations_path is not None:
        mult_lexicalizations = lexicalizations_dict(lexicalizations_path)
    else:
        mult_lexicalizations=None

    missing_lex = dict()
    missing_lex["it"]=dict()
    missing_lex["es"]=dict()
    missing_lex["de"]=dict()
    replacements = None
    translation_mapping_tuple=(translation_mapping,mult_emb_dict,mult_lexicalizations, missing_lex) #(static_sim_dict, embedings, lexicalizations, missing_lex)

    logger.info("Building train datasets ...")
    train_data, train_mappings, train_replacements = load_dataset(train_data, data_type, universal_postags,
                                                                  generator_source_copy, multilingual,
                                                                  translation_mapping_tuple, replacements,
                                                                  split="train", **params)

    logger.info("Building dev datasets ...")
    dev_data,*_ = load_dataset(dev_data, data_type,universal_postags, generator_source_copy,multilingual,train_mappings,train_replacements, split ="dev", **params)

    if test_data:
        test_data = [(l, os.path.join(params['data_dir'], p)) for l,p in params['test_data']]
        # test_data = os.path.join(params['data_dir'], params['test_data'])
        logger.info("Building test datasets ...")
        test_data, *_ = load_dataset(test_data, data_type, universal_postags, generator_source_copy,multilingual,train_mappings,train_replacements,split="test", **params)

    #logger.info("Building vocabulary ...")
    # build_vocab(fields, train_data)

    return dict(
        train=train_data,
        dev=dev_data,
        test=test_data,
        train_mappings=train_mappings,
        train_replacements=train_replacements
    )


def iterator_from_params(vocab, params):
    # TODO: There are some other options for iterator, I think we consider about it later.
    iter_type = params['iter_type']
    train_batch_size = params['train_batch_size']
    test_batch_size = params['test_batch_size']

    if iter_type == "BucketIterator":
        train_iterator = BucketIterator(
            sorting_keys=list(map(tuple, params.get('sorting_keys', []))),
            batch_size=train_batch_size,
        )
    elif iter_type == "BasicIterator":
        train_iterator = BasicIterator(
            batch_size=train_batch_size
        )
    else:
        raise NotImplementedError

    dev_iterator = BasicIterator(
        batch_size=train_batch_size
    )

    test_iterator = BasicIterator(
        batch_size=test_batch_size
    )

    train_iterator.index_with(vocab)
    dev_iterator.index_with(vocab)
    test_iterator.index_with(vocab)

    return train_iterator, dev_iterator, test_iterator
