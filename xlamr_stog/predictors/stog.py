from overrides import overrides
import re
import json
import sys
from xlamr_stog.utils.registrable import Registrable
from xlamr_stog.utils.checks import ConfigurationError
from xlamr_stog.utils.string import JsonDict, sanitize
from xlamr_stog.utils.src_tgt_match import find_similar_token_vector_test
from xlamr_stog.data import DatasetReader, Instance
from xlamr_stog.predictors.predictor import Predictor
from xlamr_stog.utils.string import START_SYMBOL, END_SYMBOL
from xlamr_stog.data.dataset_readers.amr_parsing.amr import AMRGraph
from xlamr_stog.utils.exception_hook import ExceptionHook

sys.excepthook = ExceptionHook()


@Predictor.register('STOG')
class STOGPredictor(Predictor):
    """
    Predictor for the :class:`~xl-amr.models.xl-amr. model.
    """

    def predict(self, source: str) -> JsonDict:
        return self.predict_json({"source": source})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"source": "..."}``.
        """
        source = json_dict["source"]
        return self._dataset_reader.text_to_instance(source)

    @overrides
    def predict_batch_instance(self, instances):
        outputs = []
        gen_vocab_size = self._model.vocab.get_vocab_size('decoder_token_ids')
        _outputs, encoder_last_state_seq = super(STOGPredictor, self).predict_batch_instance(instances)
        for instance, output in zip(instances, _outputs):
            gold_amr = instance.fields['amr'].metadata
            lang = instance.fields['lang'].metadata

            copy_vocab = instance.fields['src_copy_vocab'].metadata
            node_indexes = output['nodes']
            head_indexes = output['heads']
            head_label_indexes = output['head_labels']
            corefs = output['corefs']

            nodes = []
            head_labels = []
            copy_indicators = []

            for i, index in enumerate(node_indexes):
                # Lookup the node.
                if index >= gen_vocab_size:
                    copy_index = index - gen_vocab_size
                    init_word = copy_vocab.get_token_from_idx(copy_index)
                    #TODO verify
                    word = None
                    if lang!="en":
                        word = find_similar_token_vector_test(lang+"_"+init_word, (self._model.train_replacements[lang],self._model.translation_mapping[0][lang]))
                    if word is None:
                        word = init_word

                    nodes.append(word)
                    copy_indicators.append(1)
                else:
                    nodes.append(self._model.vocab.get_token_from_index(index, 'decoder_token_ids'))
                    copy_indicators.append(0)
                # Lookup the head label.
                head_labels.append(self._model.vocab.get_token_from_index(
                    head_label_indexes[i], 'head_tags'))

            if END_SYMBOL in nodes:
                nodes = nodes[:nodes.index(END_SYMBOL)]
                head_indexes = head_indexes[:len(nodes)]
                head_labels = head_labels[:len(nodes)]
                corefs = corefs[:len(nodes)]

            outputs.append(dict(
                nodes=nodes,
                heads=head_indexes,
                corefs=corefs,
                head_labels=head_labels,
                copy_indicators=copy_indicators,
                gold_amr=gold_amr
            ))

        return outputs, encoder_last_state_seq

    @overrides
    def dump_line(self, output):
        # return ' '.join(output['nodes']) + '\n'
        pred_graph = AMRGraph.from_prediction(output)
        amr = output['gold_amr']
        gold_graph = amr.graph
        amr.graph = pred_graph
        if "# ::save-date" in str(amr):
            string_to_print = str(amr).replace(
                "# ::save-date", "# ::tgt_ref {}\n# ::tgt_pred {}\n# ::save-date".format(
                # "# ::tok-en ", "# ::tgt_ref {}\n# ::tgt_pred {}\n# ::tok-en ".format(
                    " ".join(output["nodes"]),
                    " ".join(gold_graph.get_tgt_tokens()
                             )
                )
            )
        elif "# ::tok-en"  in str(amr):
            string_to_print = str(amr).replace(
                # "# ::save-date", "# ::tgt_ref {}\n# ::tgt_pred {}\n# ::save-date".format(
                "# ::tok-en", "# ::tgt_ref {}\n# ::tgt_pred {}\n# ::tok-en".format(
                    " ".join(output["nodes"]),
                    " ".join(gold_graph.get_tgt_tokens()
                             )
                )
            )
        else:
            string_to_print = str(amr)
        return string_to_print + '\n\n'
