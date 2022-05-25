
from typing import Dict, List, Tuple
import logging
import os
import json

from overrides import overrides
# NLTK is so performance orientated (ha ha) that they have lazy imports. Why? Who knows.
from xlamr_stog.data.dataset_readers.amr_parsing.io import AMRIO
from xlamr_stog.data.dataset_readers.amr_parsing.amr import AMR, AMRGraph
from xlamr_stog.utils.file import cached_path
from xlamr_stog.data.dataset_readers.dataset_reader import DatasetReader
from xlamr_stog.data.fields import TextField, SpanField, SequenceLabelField, ListField, MetadataField, Field, AdjacencyField, ArrayField
from xlamr_stog.data.instance import Instance
from xlamr_stog.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from xlamr_stog.data.tokenizers import Token
from xlamr_stog.data.tokenizers.bert_tokenizer import AMRBertTokenizer

from xlamr_stog.utils.checks import ConfigurationError
from xlamr_stog.utils.string import START_SYMBOL, END_SYMBOL


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



@DatasetReader.register("amr_trees")
class AbstractMeaningRepresentationDatasetReader(DatasetReader):
    '''
    Dataset reader for AMR data
    '''
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 word_splitter = None,
                 bpe_codes = None,
                 bpe_splitter=None,
                 universal_postags=False,
                 source_copy = True,
                 multilingual = False,
                 extra_check = False,
                 translation_mapping = None,
                 tgt_src_replacements = None,
                 split="test",
                 lazy: bool = False,
                 skip_first_line: bool = True,
                 evaluation: bool = False
                 ) -> None:

        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        if word_splitter is not None:
            self._word_splitter = AMRBertTokenizer.from_pretrained(word_splitter, do_lower_case=False)
        else:
            self._word_splitter = None

        if bpe_splitter is not None and bpe_codes is not None:
            self._bpe_splitter = None
            # self._bpe_splitter = LaserTokenizer(bpe_splitter, bpe_codes)
            # self.laser_pad_index = self._bpe_splitter.pad_index
        else:
            self._bpe_splitter = None

        self._skip_first_line = skip_first_line
        self._evaluation = evaluation
        self.universal_postags = universal_postags
        self.source_copy = source_copy
        self.multilingual = multilingual
        self.extra_check=extra_check

        # Vector based vocab mapping
        self.translation_mapping = translation_mapping
        self.split = split
        if self.split=="train" and tgt_src_replacements is None:
            self.tgt_src_replacements = dict()
            self.tgt_src_replacements["it"] = dict()
            self.tgt_src_replacements["es"] = dict()
            self.tgt_src_replacements["de"] = dict()
            self.tgt_src_replacements["zh"] = dict()
            self.tgt_src_replacements["ms"] = dict()
        else:
            self.tgt_src_replacements = tgt_src_replacements

        self._number_bert_ids = 0
        self._number_bert_oov_ids = 0
        self._number_non_oov_pos_tags = 0
        self._number_pos_tags = 0

    def report_coverage(self):
        if self._number_bert_ids != 0:
            logger.info('BERT OOV  rate: {0:.4f} ({1}/{2})'.format(
                self._number_bert_oov_ids / self._number_bert_ids,
                self._number_bert_oov_ids, self._number_bert_ids
            ))
        if self._number_non_oov_pos_tags != 0:
            logger.info('POS tag coverage: {0:.4f} ({1}/{2})'.format(
                self._number_non_oov_pos_tags / self._number_pos_tags,
                self._number_non_oov_pos_tags, self._number_pos_tags
            ))

    def set_evaluation(self):
        self._evaluation = True

    @staticmethod
    def convert_postags(lang):
        u_pos = dict()
        with open("data/misc/pos-conversion-table.{}".format(lang), "r") as infile:
            for line in infile:
                fields = line.rstrip().split()
                u_pos[fields[0]] = fields[1]
        return u_pos

    @overrides
    def _read(self, file_path_list):
        print(file_path_list)
        source_copy = self.source_copy
        for lang, file_path in file_path_list:
            try:
                u_pos = self.convert_postags(lang)
            except:
                u_pos = None
            # if `file_path` is a URL, redirect to the cache
            file_path = cached_path(file_path)
            logger.info("Reading instances from lines in file at: %s %s %s", lang, file_path, self.split)
            i = 0
            for amr in AMRIO.read(file_path, lang=lang, universal_postags=self.universal_postags, postag_map=u_pos):
                i+=1

                try:
                    yield self.text_to_instance(amr, lang, source_copy, self.split)
                except Exception as e:
                    if self.split != "test":
                        continue
                    else:
                        raise e
        self.report_coverage()

    @overrides
    def text_to_instance(self, amr:AMR, lang = None, source_copy= True, split="test") -> Instance:
        # pylint: disable=arguments-differ

        fields: Dict[str, Field] = {}

        max_tgt_length = None if self._evaluation else 60

        list_data = amr.graph.get_list_data(
            amr, START_SYMBOL, END_SYMBOL, self._word_splitter, self._bpe_splitter, max_tgt_length,
            split=split,
            language=lang,
            multilingual=self.multilingual,
            extra_check=self.extra_check,
            source_copy= source_copy,
            translation_mapping = self.translation_mapping,
            tgt_src_replacements=self.tgt_src_replacements)


        # These four fields are used for seq2seq model and target side self copy

        fields["src_tokens"] = TextField(
            tokens=[Token(x) for x in list_data["src_tokens"]],
            token_indexers={k: v for k, v in self._token_indexers.items() if 'encoder' in k}
        )

        if list_data['src_token_ids'] is not None:
            fields['src_token_ids'] = ArrayField(list_data['src_token_ids'])
            self._number_bert_ids += len(list_data['src_token_ids'])
            self._number_bert_oov_ids += len([bert_id for bert_id in list_data['src_token_ids'] if bert_id == 100])

        if list_data['src_token_subword_index'] is not None:
            fields['src_token_subword_index'] = ArrayField(list_data['src_token_subword_index'])


        if list_data['bpe_src_token_ids'] is not None:
            fields['bpe_src_token_ids'] = ArrayField(list_data['bpe_src_token_ids'], padding_value=self.laser_pad_index)
            # self._number_laser_ids += len(list_data['bpe_src_token_ids'])
            # self._number_laser_oov_ids += len([bpe_id for bpe_id in list_data['bpe_src_token_ids'] if bpe_id == 3])

        if list_data['bpe_src_token_subword_index'] is not None:
            fields['bpe_src_token_subword_index'] = ArrayField(list_data['bpe_src_token_subword_index'], padding_value=-1)


        fields["src_must_copy_tags"] = SequenceLabelField(
            labels=list_data["src_must_copy_tags"],
            sequence_field=fields["src_tokens"],
            label_namespace="must_copy_tags"
        )

        fields["tgt_tokens"] = TextField(
            tokens=[Token(x) for x in list_data["tgt_tokens"]],
            token_indexers={k: v for k, v in self._token_indexers.items() if 'decoder' in k}
        )

        fields["src_pos_tags"] = SequenceLabelField(
            labels=list_data["src_pos_tags"],
            sequence_field=fields["src_tokens"],
            label_namespace="pos_tags"
        )

        fields["tgt_pos_tags"] = SequenceLabelField(
            labels=list_data["tgt_pos_tags"],
            sequence_field=fields["tgt_tokens"],
            label_namespace="pos_tags"
        )

        self._number_pos_tags += len(list_data['tgt_pos_tags'])
        self._number_non_oov_pos_tags += len(
            [tag for tag in list_data['tgt_pos_tags'] if tag != '@@UNKNOWN@@'])

        fields["tgt_copy_indices"] = SequenceLabelField(
            labels=list_data["tgt_copy_indices"],
            sequence_field=fields["tgt_tokens"],
            label_namespace="coref_tags",
        )

        fields["tgt_copy_mask"] = SequenceLabelField(
            labels=list_data["tgt_copy_mask"],
            sequence_field=fields["tgt_tokens"],
            label_namespace="coref_mask_tags",
        )

        fields["tgt_copy_map"] = AdjacencyField(
            indices=list_data["tgt_copy_map"],
            sequence_field=fields["tgt_tokens"],
            padding_value=0
        )

        # These two fields for source copy
        fields["src_copy_indices"] = SequenceLabelField(
            labels=list_data["src_copy_indices"],
            sequence_field=fields["tgt_tokens"],
            label_namespace="source_copy_target_tags",
        )

        fields["src_copy_map"] = AdjacencyField(
            indices=list_data["src_copy_map"],
            sequence_field=TextField(
                [
                    Token(x) for x in list_data["src_copy_vocab"].get_special_tok_list() + list_data["src_tokens"]
                ],
                None
            ),
            padding_value=0
        )

        # These two fields are used in biaffine parser
        fields["head_tags"] = SequenceLabelField(
            labels=list_data["head_tags"],
            sequence_field=fields["tgt_tokens"],
            label_namespace="head_tags",
            strip_sentence_symbols=True
        )

        fields["head_indices"] = SequenceLabelField(
            labels=list_data["head_indices"],
            sequence_field=fields["tgt_tokens"],
            label_namespace="head_index_tags",
            strip_sentence_symbols=True
        )

        if self._evaluation:
            # Metadata fields, good for debugging
            fields["src_tokens_str"] = MetadataField(
                list_data["src_tokens"]
            )

            fields["tgt_tokens_str"] = MetadataField(
                list_data.get("tgt_tokens", [])
            )

            fields["src_copy_vocab"] = MetadataField(
                list_data["src_copy_vocab"]
            )

            fields["tag_lut"] = MetadataField(
                dict(pos=list_data["pos_tag_lut"])
            )

            fields["source_copy_invalid_ids"] = MetadataField(
                list_data['src_copy_invalid_ids']
            )

            fields["amr"] = MetadataField(
                amr
            )
            fields["lang"] = MetadataField(
                lang
            )

        return Instance(fields)


