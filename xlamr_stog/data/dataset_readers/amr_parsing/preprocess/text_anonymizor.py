import os
import re
import json
import string
from typing import List, Dict, Set
from collections import defaultdict

from xlamr_stog.data.dataset_readers.amr_parsing.io import AMRIO
from xlamr_stog.data.dataset_readers.amr_parsing.amr import AMR
from tqdm import tqdm

def prev_token_is(index: int, k: int, amr: AMR, pattern: str):
    if index - k >= 0:
        return re.match(pattern, amr.lemmas[index - k])


def next_token_is(index: int, k: int, amr: AMR, pattern: str):
    if index + k < len(amr.lemmas):
        return re.match(pattern, amr.lemmas[index + k])


def is_anonym_type(index: int, amr: AMR, text_map: Dict, types: List) -> bool:
    lemma = amr.lemmas[index]
    # print(lemma)
    if lemma in text_map and type(text_map[lemma])==dict:
        return lemma in text_map and text_map[lemma]['ner'] in types
    else:
        return False


class TextAnonymizor:

    def __init__(self,
                 text_maps: Dict,
                 priority_lists: List,
                 _VNE: str,
                 _LOCEN1: List,
                 _LOCEN2: List,
                 _N: List,
                 _M: List,
                 _R: List,
                 _INVP: List,
                 _INVS: List,
                 exlude_ners: bool
                 ) -> None:
        self._text_maps = text_maps
        self._priority_lists = priority_lists
        self._VNE = _VNE
        self._LOCEN1 = _LOCEN1
        self._LOCEN2 = _LOCEN2
        self._N = _N
        # self._M = _M
        self._M = "'^settembre'"
        self._R = _R
        self._INVP = _INVP
        self._INVS = _INVS
        self.NM_anonym={x.rstrip() for x in open("data/misc/NM_anonym.txt").readlines()}
        self.exlude_ners = exlude_ners

    def __call__(self, amr: AMR) -> Dict:
        anonymization_map = {}
        for anonym_type, (text_map, pos_tag) in self._text_maps.items():
            max_length = len(max(text_map, key=len))
            anonymization_map.update(self._abstract(amr, text_map, max_length, anonym_type, pos_tag))
        return anonymization_map

    def _abstract(self,
                  amr: AMR,
                  text_map: Dict,
                  max_length: int,
                  anonym_type: str,
                  pos_tag: str) -> Dict:
        replaced_spans = {}
        collected_entities = set()
        ignored_spans = self._get_ignored_spans(amr)
        while True:
            done = self._replace_span(
                amr,
                text_map,
                max_length,
                anonym_type,
                pos_tag,
                ignored_spans,
                replaced_spans,
                collected_entities,
            )
            if done:
                break
        ner_counter = defaultdict(int)
        anonymization_map = {}
        for i, lemma in enumerate(amr.lemmas):
            if lemma in replaced_spans:
                if anonym_type == 'quantity':
                    ner = lemma.rsplit('_', 2)[1]
                else:
                    ner = lemma.rsplit('_', 1)[0]
                ner_counter[ner] += 1

                if anonym_type == 'quantity':
                    if ner in ('1', '10', '100', '1000') or not re.search(r"[\./]", ner):
                        anonym_lemma = str(int(ner) * ner_counter[ner])
                    else:
                        anonym_lemma = str(float(ner) * ner_counter[ner])
                else:
                    anonym_lemma = ner + '_' + str(ner_counter[ner])

                amr.lemmas[i] = anonym_lemma
                amr.tokens[i] = anonym_lemma
                anonymization_map[anonym_lemma] = replaced_spans[lemma]
        return anonymization_map

    def _leave_as_is(self,
                     index: int,
                     amr: AMR,
                     text_map: Dict,
                     anonym_type: str) -> bool:
        if anonym_type == 'named-entity':
            if amr.pos_tags[index].startswith('V') and not next_token_is(index, 1, amr, self._VNE):
                return True
            if amr.tokens[index]=="polizia": return True
            if (is_anonym_type(index, amr, text_map, ["LOCATION", "ENTITY"])
                    and next_token_is(index, 0, amr, self._LOCEN1[0]) and (
                        prev_token_is(index, 1, amr, self._LOCEN1[1]) or
                        next_token_is(index, 1, amr, self._LOCEN1[2]))):
                return True
            if next_token_is(index, 0, amr, self._LOCEN2[0]) and prev_token_is(index, 1, amr, self._LOCEN2[1]):
                return True

        if anonym_type == 'ordinal-entity':
            if next_token_is(index, 0, amr, r"^\d+th$") and not prev_token_is(index, 1, amr, self._M):
                return False
            if len(amr.lemmas[index]) == 1 and amr.lemmas[index].isdigit() and (
                    next_token_is(index, 1, amr, self._R[0]) or next_token_is(index, 2, amr, self._R[1])):
                return False
            if index == len(amr.lemmas) - 2 and amr.pos_tags[index + 1] in '.,!?':
                return True
            if prev_token_is(index, 1, amr, self._INVP[0]) or next_token_is(index, 1, amr, self._INVS[0]):
                return True
            if not prev_token_is(index, 2, amr, self._INVP[1]) and next_token_is(index, 1, amr, self._INVS[1]):
                return True
            if next_token_is(index, 1, amr, self._R[1]) and (
                    not next_token_is(index, 3, amr, self._VNE) or prev_token_is(index, 1, amr, r"^ORDINAL")):
                return True

        if anonym_type == 'date-entity':
            if is_anonym_type(index, amr, text_map, ['DATE_ATTRS']) and next_token_is(index, 1, amr, r"^''$"):
                    return True
            if (amr.lemmas[index].isdigit() and len(amr.lemmas[index]) < 4 and (
                        prev_token_is(index, 1, amr, self._INVP[2]) or next_token_is(index, 1, amr, self._INVS[2]))):
                return True
            if amr.lemmas[index].isalpha() and (prev_token_is(index, 1, amr, self._INVP[3]) or next_token_is(index, 1, amr, self._INVS[3])):
                return True

        if anonym_type == 'quantity':
            if len(amr.lemmas[index]) == 1 and prev_token_is(index, 2, amr, self._INVP[4]) and next_token_is(index, 1, amr, self._INVP[4]):
                return True
            if ' '.join(amr.lemmas[index - 2: index + 2]) in self._N[2:4]:
                return True
        else:
            if index == 0 and len(amr.lemmas[index]) == 1 and amr.lemmas[index].isdigit():
                return True

        if anonym_type != 'ordinal-entity':
            if amr.ner_tags[index] == 'ORDINAL' and not next_token_is(index, 0, amr, self._N[1]):
                return True

        if next_token_is(index, 0, amr, self._N[0]) and (
                prev_token_is(index, 1, amr, self._INVP[5]) or next_token_is(index, 1, amr, self._INVS[5])):
            return True

        return False

    def _replace_span(self,
                      amr: AMR,
                      text_map: Dict,
                      max_length: int,
                      anonym_type: str,
                      pos_tag: str,
                      ignored_spans: Set,
                      replaced_spans: Dict,
                      collected_entities: Set) -> bool:
        for length in range(max_length, 0, -1):
            for start in range(len(amr.lemmas) + 1 - length):
                if length == 1 and self._leave_as_is(start, amr, text_map, anonym_type):
                    continue
                span1 = ' '.join(amr.tokens[start:start + length])
                span2 = ' '.join(amr.lemmas[start:start + length])
                ner_tags = amr.ner_tags[start:start + length]
                if anonym_type == 'named-entity' and len(set(ner_tags)) > 1:
                    check_here=True
                if span1 in ignored_spans or span2 in ignored_spans:
                    continue
                if lang_stopwords:
                    if span1 in lang_stopwords or span2 in lang_stopwords: continue

                if (span1 in text_map and type(text_map[span1])==dict) or (span2 in text_map and type(text_map[span2])==dict) or (amr.lang=="zh" and span1.replace(" ","") in text_map and type(text_map[span1.replace(" ","")])==dict) or (amr.lang=="zh" and span2.replace(" ","") in text_map and type(text_map[span2.replace(" ","")])==dict):
                    if amr.lang=="zh":
                        value = text_map.get(span1, None) or text_map.get(span2, None) or text_map.get(span1.replace(" ",""), None) or text_map.get(span2.replace(" ",""), None)
                    else:
                        value = text_map.get(span1, None) or text_map.get(span2, None)
                    if anonym_type == 'named-entity':
                        entity_name = value['lemma'] if 'lemma' in value else value['ops']
                        if entity_name in collected_entities:
                            continue
                        else:
                            collected_entities.add(entity_name)
                    anonym_lemma = value['ner'] + '_' + str(len(replaced_spans))
                    pos_tag = amr.pos_tags[start] if anonym_type == 'quantity' else pos_tag
                    ner = 'NUMBER' if anonym_type == 'quantity' else value['ner']
                    replaced_spans[anonym_lemma] = value
                    amr.replace_span(list(range(start, start + length)), [anonym_lemma], [pos_tag], [ner])
                    return False
                else:

                    if lang2en_bn is not None and  lang2en_span is not None: #continue
                        if anonym_type != "named-entity": continue
                        ner_tags = amr.ner_tags[start:start + length]
                        # if len(set(ner_tags))==1 and ner_tags[0] in {"O","0"}: continue
                        if amr.lang =="zh":
                            span1=span1.replace(" ","")
                            span2=span2.replace(" ","")
                        else:
                            span1=span1.replace(" ","_")
                            span2=span2.replace(" ","_")

                        if span1 in lang2en_span or span2 in lang2en_span or span1.title() in lang2en_span or span2.title() in lang2en_span:
                            if span1 in lang2en_span: candidate_span = span1
                            elif span2 in lang2en_span: candidate_span = span2
                            elif span1.title() in lang2en_span: candidate_span = span1.title()
                            else: candidate_span = span2.title()

                            if candidate_span is not None:
                                en_spans=lang2en_span[candidate_span]
                                value=None
                                for en_span in en_spans:
                                    en_span=en_span.title()
                                    if en_span in text_map and type(text_map[en_span]) == dict:
                                        value = text_map.get(en_span, None)
                                        break
                                if value is None: continue
                                if anonym_type == 'named-entity':
                                    entity_name = value['lemma'] if 'lemma' in value else value['ops']
                                    if entity_name in collected_entities:
                                        continue
                                    else:
                                        collected_entities.add(entity_name)
                                anonym_lemma = value['ner'] + '_' + str(len(replaced_spans))
                                pos_tag = amr.pos_tags[start] if anonym_type == 'quantity' else pos_tag
                                ner = 'NUMBER' if anonym_type == 'quantity' else value['ner']
                                replaced_spans[anonym_lemma] = value
                                amr.replace_span(list(range(start, start + length)), [anonym_lemma], [pos_tag], [ner])
                                return False

                        elif span1 in lang2en_bn or span2 in lang2en_bn or span1.title() in lang2en_bn or span2.title() in lang2en_bn:
                            if anonym_type != "named-entity": continue
                            if span1 in string.punctuation or span2 in string.punctuation: continue
                            if len(set(ner_tags)) == 1 and ner_tags[0] in {"O", "0"}: continue
                            if span1 in lang2en_bn: candidate_nm = span1
                            elif span2 in lang2en_bn: candidate_nm = span2
                            elif span1.title() in lang2en_bn: candidate_nm = span1.title()
                            else: candidate_nm = span2.title()

                            if candidate_nm is not None:
                                en_nm = lang2en_bn[candidate_nm]
                                entity_name = en_nm.replace("_"," ")
                                if entity_name in collected_entities:
                                    continue
                                else:
                                    collected_entities.add(entity_name)
                                ner=[x for x in ner_tags if x not in {"O","0"}]
                                if len(ner)==0:
                                    ner="ENTITY"
                                else:
                                    ner=ner[0]

                                anonym_lemma = ner + '_' + str(len(replaced_spans))
                                pos_tag = amr.pos_tags[start] if anonym_type == 'quantity' else pos_tag

                                replaced_spans[anonym_lemma] = {"type": "named-entity", "span": entity_name, "ops": entity_name, "ner": ner}
                                # replaced_spans[anonym_lemma] = {"type": "named-entity", "span": span1.replace(" ","_"), "ops": entity_name, "ner": ner}
                                amr.replace_span(list(range(start, start + length)), [anonym_lemma], [pos_tag], [ner])
                                return False
                    #inclusive for all languages
                    if amr.lang == "en" and self.exlude_ners: continue
                    if amr.lang == "zh": continue
                    if "_".join(span1.split("_")[:-1]) not in self.NM_anonym and len(set(ner_tags))==1 and ner_tags[0] not in {"O", "0"}:
                        # print("_".join(span1.split("_")[:-1]))
                        cont=False
                        for nmt in self.NM_anonym:
                            if nmt in span1:
                                cont=True

                        if cont: continue
                        entity_name = span1.replace("_"," ")
                        if entity_name in collected_entities:
                            continue
                        else:
                            collected_entities.add(entity_name)
                        if ner_tags[0] in ["EMAIL","PERCENT", "GPE","CARDINAL", "EVENT", "NORP", "QUANTITY", "LAW", "FAC", "PRODUCT","WORK_OF_ART"]:
                            ner_tag_am = "ENTITY"
                        else:
                            ner_tag_am = ner_tags[0]
                        anonym_lemma = ner_tag_am + '_' + str(len(replaced_spans))
                        pos_tag = amr.pos_tags[start] if anonym_type == 'quantity' else pos_tag
                        ner = ner_tag_am #ner_tags[0]
                        # replaced_spans[anonym_lemma] = {"type": "named-entity", "span": span1, "ops": entity_name, "ner": ner}
                        replaced_spans[anonym_lemma] = {"type": "named-entity", "span": entity_name, "ops": entity_name,"ner": ner}
                        amr.replace_span(list(range(start, start + length)), [anonym_lemma], [pos_tag], [ner])
                        return False

        return True

    def _get_ignored_spans(self, amr: AMR) -> Set:
        ignored_spans = set()
        for spans in self._priority_lists:
            for i, span in enumerate(spans):
                tokens = span.split()
                if len(tokens) > 1:
                    if span + ' ' in ' '.join(amr.lemmas) or span + ' ' in ' '.join(amr.tokens):
                        ignored_spans.update(spans[i + 1:])
                        break
                else:
                    if span in amr.lemmas or span in amr.tokens:
                        ignored_spans.update(spans[i + 1:])
                        break
        return ignored_spans

    @classmethod
    def from_json(cls, file_path: str, exlude_ners: bool = True) -> 'TextAnonymizor':
        with open(file_path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            text_maps=d["text_maps"],
            priority_lists=d["priority_lists"],
            _VNE=d["VNE"],
            _LOCEN1=d["LOCEN1"],
            _LOCEN2=d["LOCEN2"],
            _N=d["N"],
            _M=d["M"],
            _R=d["R"],
            _INVP=d["INVP"],
            _INVS=d["INVS"],
            exlude_ners=exlude_ners
        )

from collections import Counter

def load_name_bn_map(json_file):
    en_nm_lang_nm = json.load(open(json_file, "r", encoding='utf-8'))
    lang_nm2en_nm=dict()
    for en_nm in en_nm_lang_nm:
        lang_nm = en_nm_lang_nm[en_nm]
        for nm in lang_nm:
            if nm.lower() in {"morte", "muerte","nunca"}: continue
            if nm not in lang_nm2en_nm:
                lang_nm2en_nm[nm]=set()
            lang_nm2en_nm[nm].add(en_nm)


    return lang_nm2en_nm

def load_name_bn_wiki_map(file):
    lang_nm2en_nm = dict()

    with open(file, "r", encoding='utf-8') as infile:
        for line in infile:
            if line.startswith("#"): continue
            fields=line.rstrip().split()
            lang_nm = fields[2].replace("-","_")

            if lang_nm in lang_nm2en_nm: continue
            if lang_nm.islower(): continue
            if ":EN:" in fields[1]:
                en_wiki = fields[1].split(":EN:")[-1]
            elif "#n#1" in fields[1]:
                en_wiki = fields[1][:-4]
            else:
                continue

            # if lang_nm.lower() in {"Secondo","uno","due","tre","lettere","trattati","trattato", "presidente"}:continue
            lang_nm2en_nm[lang_nm]=en_wiki

    return lang_nm2en_nm

def load_name_span_map(json_file, lang):
    en_span_en_lang = json.load(open(json_file, "r", encoding='utf-8'))
    lang_nm2en_span=dict()
    for span in en_span_en_lang:
        lang_nm = en_span_en_lang[span][lang]
        for nm in lang_nm:
            if nm.islower(): continue
            if nm.lower() in {"morte", "muerte","natura","stati","padre","polizia","nunca"}: continue
            if nm not in lang_nm2en_span:
                lang_nm2en_span[nm]=set()
            lang_nm2en_span[nm].add(span)

    return lang_nm2en_span

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("text_anonymizor.py")

    parser.add_argument('--amr_file', nargs="+", required=True, help="File to anonymize.")
    parser.add_argument('--util_dir')
    parser.add_argument('--lang')
    parser.add_argument('--exclude_ners', action="store_true", help="consider NER tags for entities not found in training.")
    args = parser.parse_args()


    if args.lang=="en":
        text_anonymizor = TextAnonymizor.from_json(os.path.join(args.util_dir,
            "text_anonymization_rules.json"))
        lang_stopwords=None
        lang2en_span=None
        lang2en_bn=None

    else:
        text_anonymizor = TextAnonymizor.from_json(os.path.join(args.util_dir,"text_anonymization_en-{}.json".format(args.lang)))
        lang_stopwords = set([x.rstrip() for x in open("data/cross-lingual-babelnet_mappings/stopwords_{}.txt".format(args.lang))])

        lang2en_span=load_name_span_map("data/cross-lingual-babelnet_mappings/name_span_en_{}_map_amr_bn.json".format(args.lang), args.lang)
        lang2en_bn=load_name_bn_wiki_map("data/cross-lingual-babelnet_mappings/namedEntity_wiki_synsets.{}.tsv".format(args.lang.upper()))

    for amr_file in args.amr_file:
        with open(amr_file + ".recategorize{}".format("_noner" if args.exclude_ners else ""), "w", encoding="utf-8") as f:
            for amr in tqdm(AMRIO.read(amr_file, lang=args.lang)):

                amr.abstract_map = text_anonymizor(amr)
                f.write(str(amr) + "\n\n")
