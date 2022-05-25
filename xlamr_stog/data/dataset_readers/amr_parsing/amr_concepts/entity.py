import re
from collections import defaultdict, Counter

import nltk


STEMMER = nltk.stem.SnowballStemmer('english').stem


def strip_lemma(lemma):
    # Remove twitter '@'.
    if len(lemma) and lemma[0] == '@':
        lemma = lemma[1:]
    # Remove '-$' suffix.
    if len(lemma) and lemma[-1] == '$':
        lemma = lemma[:-1]
    return lemma


def strip_span(span, tokens):
    start = 0
    while start < len(span):
        token = tokens[span[start]]
        if not re.search(r'^(in|of|at|-|,)$', token):
            break
        start += 1
    end = len(span) - 1
    while end > start:
        token = tokens[span[end]]
        if not re.search(r'^(in|of|at|-|,)$', token):
            break
        end -= 1
    return span[start: end + 1]


def rephrase_ops(ops):
    ret = []
    joined_ops = ' '.join(map(str, ops))
    if joined_ops == '"United" "States"':
        ret.append('"America"')
    elif joined_ops == '"World" "War" "II"':
        ret.append('"WWII"')
    elif joined_ops == '"Republican" "National" "Convention"':
        ret.append('"RNC"')
    elif joined_ops == '"Grand" "Old" "Party"':
        ret.append('"GOP"')
    elif joined_ops == '"United" "Nations"':
        ret.append('"U.N."')
    return ret


def tokenize_ops(ops):
    ret = []
    for op in ops:
        if not isinstance(op, str):
            ret += [op]
            continue
        if re.search(r'^".*"$', op):
            op = op[1:-1]
        ret += re.split(r"(-|'s|n't|')", op)
    return ret


def group_indexes_to_spans(indexes, amr):
    indexes = list(indexes)
    indexes.sort()
    spans = []
    last_index = None
    for idx in indexes:
        if last_index is None or idx - last_index > 2:
            spans.append([])
        elif idx - last_index == 2:
            if re.search(r"(,|'s|of|'|-|in)", amr.tokens[idx - 1]):
                spans[-1].append(idx - 1)
            else:
                spans.append([])
        last_index = idx
        spans[-1].append(idx)
    return spans


class Entity:

    # Sometimes there are different ways to say the same thing.
    entity_map = {
        'Netherlands': ['Dutch'],
        'Shichuan': ['Sichuan'],
        'France': ['Frence'],
        'al-Qaida': ['Al-Qaida'],
        'Gorazde': ['Gerlaridy'],
        'Sun': ['Solar'],
        'China': ['Sino'],
        'America': ['US', 'U.S.'],
        'U.S.': ['US'],
        'Georgia': ['GA'],
        'Pennsylvania': ['PA', 'PA.'],
        'Missouri': ['MO', 'MO.'],
        'WWII': ['WW2'],
        'WWI': ['WW1'],
        'Iran': ['Ian'],
        'Jew': ['Semitism', 'Semites'],
        'Islam': ['Muslim'],
        'influenza': ['flu'],
    }

    unknown_entity_type = 'ENTITY'

    def __init__(self, span=None, node=None, ner_type=None, amr_type=None, confidence=0, alignment=None):
        self.span = span
        self.node = node
        self.ner_type = ner_type
        self.amr_type = amr_type
        self.confidence = confidence
        self.alignment = alignment
        self.debug = False
        self.is_bn = False
        self.non_en_ops = None

    def __str__(self):
        return 'node:{}\nalignment:{}'.format(str(self.node), self.alignment)

    def get_text_spans(self, amr):
        spans = []
        span = []
        for index in self.span:
            span.append(amr.tokens[index])
        spans.append(' '.join(span))
        span = []
        for op in self.node.ops:
            op = str(op)
            if re.search(r'^".*"$', op):
                op = op[1:-1]
            span.append(op)
        spans.append(' '.join(span))
        span = []
        for op in rephrase_ops(self.node.ops):
            if re.search(r'^".*"$', op):
                op = op[1:-1]
            span.append(op)
        spans.append(' '.join(span))
        return [span for span in spans if len(span) > 0]

    def get_ops(self, en=True):
        ops = []
        if not en:
            return self.non_en_ops
        for op in self.node.ops:
            op = str(op)
            if re.search(r'^".*"$', op):
                op = op[1:-1]
            ops.append(op)
        return ops

    @classmethod
    def get_aligned_entity(cls, node, amr, backup_ner_type, entity_type_lut, babelnet_mapping=None, stemmer=None, stopwords=None, countries=None, amr_type=None):
        if len(node.ops) == 0:
            return None
        alignment = cls.get_alignment_for_ops(rephrase_ops(node.ops), amr, babelnet_mapping, stemmer)
        if len(alignment) == 0:
            alignment = cls.get_alignment_for_ops(node.ops, amr, babelnet_mapping, stemmer)

        entity1 = cls(node=node, alignment=alignment)
        entity1._get_aligned_info(amr, backup_ner_type, entity_type_lut, stopwords=stopwords)

        alignment = cls.get_alignment_for_ops(tokenize_ops(node.ops), amr, babelnet_mapping, stemmer)
        entity2 = cls(node=node, alignment=alignment)
        entity2._get_aligned_info(amr, backup_ner_type, entity_type_lut,  stopwords=stopwords)

        entity_en = entity2 if entity2.confidence > entity1.confidence else entity1
        entity_bn = None
        if babelnet_mapping:
            non_EN_nms=Counter()

            ops_ = " ".join([str(op) for op in node.ops]).replace('"',"").replace("'","")
            if ops_ in babelnet_mapping:
                non_EN_nms.update(babelnet_mapping[ops_])
            elif ops_.replace(" ", "_") in babelnet_mapping:
                non_EN_nms.update(babelnet_mapping[ops_.replace(" ", "_")])
            elif ops_.replace(" ", "-") in babelnet_mapping:
                non_EN_nms.update(babelnet_mapping[ops_.replace(" ", "-")])
            elif ops_.replace(" ", "_").lower() in babelnet_mapping:
                non_EN_nms.update(babelnet_mapping[ops_.replace(" ", "_").lower()])

            if amr_type is not None and countries is not None:
                if amr_type=="country" or backup_ner_type=="LOCATION":
                    if ops_ in countries:
                        nationality =countries[ops_]
                        if nationality in babelnet_mapping:
                            non_EN_nms.update(babelnet_mapping[nationality])
                        elif nationality.lower() in babelnet_mapping:
                            non_EN_nms.update(babelnet_mapping[nationality.lower()])
                        elif nationality.replace(" ","_") in babelnet_mapping:
                            non_EN_nms.update(babelnet_mapping[nationality.replace(" ","_") ])
                        elif nationality.replace(" ","_") in babelnet_mapping:
                            non_EN_nms.update(babelnet_mapping[nationality.replace(" ","_").lower()])

            best_length = 0
            for nm_idx, non_EN_ops in enumerate(non_EN_nms):
                non_en_ops = non_EN_ops.replace("_"," ").replace("-"," ").split()
                alignment = cls.get_alignment_for_ops(non_en_ops, amr, babelnet_mapping, stemmer)
                if len(alignment)==0: continue
                entity = cls(node=node, alignment=alignment)
                entity.non_en_ops=non_en_ops
                entity._get_aligned_info(amr, backup_ner_type, entity_type_lut, stopwords=stopwords)
                if entity_bn is None:
                    entity_bn = entity
                    best_length = len(alignment)
                else:
                    if entity.confidence > entity_bn.confidence:
                        entity_bn=entity
                        best_length = len(alignment)
                    elif entity.confidence==entity_bn.confidence:
                        if len(alignment)<best_length:
                            entity_bn=entity
                            best_length = len(alignment)

                # entity_bn = entity if entity.confidence > entity_bn.confidence else entity_bn

        if entity_bn is not None:
            entity = entity_en if entity_en.confidence>entity_bn.confidence else entity_bn
            entity.is_bn = entity_en.confidence <= entity_bn.confidence
        else:
            entity = entity_en
        return entity


    @staticmethod
    def get_alignment_for_ops(ops, amr, babelnet_mapping=None, stemmer = None):
        alignment = {}

        for i, op in enumerate(ops):
            for j, token in enumerate(amr.tokens):
                confidence = Entity.maybe_align_op_to(op, j, amr,stemmer)
                if confidence > 0:
                    if j not in alignment or (j in alignment and alignment[j][1] < confidence):
                        alignment[j] = (i, confidence)

        return alignment

    @staticmethod
    def maybe_align_op_to(op, index, amr, stemmer=None):
        if stemmer:
            STEMMER = stemmer
        else:
            STEMMER = nltk.stem.SnowballStemmer('english').stem
        if not isinstance(op, str):
            op = str(op)
        if re.search(r'^".*"$', op):
            op = op[1:-1]
        op_lower = op.lower()
        token_lower = amr.tokens[index].lower()
        lemma_lower = amr.lemmas[index].lower()
        stripped_lemma_lower = strip_lemma(lemma_lower)
        # Exact match.
        if amr.tokens[index] == op or amr.lemmas[index] == op:
            return 15
        elif op_lower == token_lower or op_lower == lemma_lower or op_lower == stripped_lemma_lower:
            return 10
        # Stem exact match.
        elif STEMMER(op) == amr.stems[index]:
                return 8
        # Tagged as named entity and match the first 3 chars.
        elif amr.is_named_entity(index) and (
                op_lower[:3] == token_lower[:3] or
                op_lower[:3] == lemma_lower[:3] or
                op_lower[:3] == stripped_lemma_lower[:3]
        ):
            return 5
        # Match the first 3 chars.
        elif (op_lower[:3] == token_lower[:3] or
              op_lower[:3] == lemma_lower[:3] or
              op_lower[:3] == stripped_lemma_lower[:3]
        ):
            if amr.lang!="en":
                return 0
            else:
                return 1
        # Match after mapping.
        elif op in Entity.entity_map:
            return max(Entity.maybe_align_op_to(mapped_op, index, amr, STEMMER) for mapped_op in Entity.entity_map[op])
        else:
            return 0

    @staticmethod
    def collapse_name_nodes(entities, amr, type_counter=None):
        if amr.abstract_map is None:
            amr.abstract_map = {}
        if len(entities) == 0:
            return
        if type_counter is None:
            type_counter = defaultdict(int)
        entities.sort(key=lambda entity: entity.span[-1] if len(entity.span) else float('inf'))
        offset = 0
        for entity in entities:
            if len(entity.span) > 0:
                type_counter[entity.ner_type] += 1
                abstract = '{}_{}'.format(
                    entity.ner_type, type_counter[entity.ner_type])
                span_with_offset = [index - offset for index in entity.span]
                amr.abstract_map[abstract] = Entity.save_collapsed_name_node(
                    entity, span_with_offset, amr)
                amr.replace_span(span_with_offset, [abstract], ['NNP'], [entity.ner_type])
                amr.stems = amr.stems[:span_with_offset[0]] + [abstract] + amr.stems[span_with_offset[-1] + 1:]
                amr.graph.remove_node_ops(entity.node)
                amr.graph.replace_node_attribute(
                    entity.node, 'instance', entity.node.instance, abstract)
                offset += len(entity.span) - 1
            else:
               try:
                    amr.graph.remove_node(entity.node)
               except: continue
        return type_counter

    @staticmethod
    def save_collapsed_name_node(entity, span, amr):
        return dict(
            type='named-entity',
            span=' '.join(map(amr.tokens.__getitem__, span)),
            ops=' '.join(entity.get_ops())
        )

    def _get_aligned_info(self, amr, backup_ner_type, entity_type_lut, stopwords = None):
        spans = group_indexes_to_spans(self.alignment.keys(), amr)
        spans.sort(key=lambda span: len(span), reverse=True)
        spans = self._clean_span(spans, amr)
        amr_type = amr.graph.get_name_node_type(self.node)
        candidate_spans = []
        for span in spans:
            orig_span = span
            if stopwords:
                first = True
                last = True
                while len(span) and first:
                    if amr.tokens[span[0]] in stopwords or amr.tokens[span[0]].lower() in stopwords:
                        if amr.tokens[span[0]].isupper() and span[0]!=0:
                            first=False
                            continue
                        span = span[1:]
                    else:
                        first = False
                while len(span) and last:
                    if amr.tokens[span[-1]] in stopwords or amr.tokens[span[-1]].lower() in stopwords:
                        if amr.tokens[span[-1]].isupper() and span[-1]!= 0:
                            last=False
                            continue
                        span = span[:-1]
                    else:
                        last=False


            if len(span)==0: continue
            if stopwords:
                if amr.tokens[span[0]] in stopwords or amr.tokens[span[-1]] in stopwords:
                    print([amr.tokens[idx] for idx in orig_span])
                    print([amr.tokens[idx] for idx in span])
            # confidence = sum(self.alignment[j][1] if amr.tokens[self.alignment[j][0]] not in stopwords else -1 for j in span if j in self.alignment)
            confidence = sum(self.alignment[j][1] for j in span if j in self.alignment)
            if amr.tokens[span[0]][0].isupper() and span[0]!=0:
                confidence+=5
            # elif amr.tokens[span[0]][0].isupper():
            #     confidence+=2
            candidate_spans.append((span, confidence))
        if len(candidate_spans):
            best_span, confidence = max(candidate_spans, key=lambda x: x[1])
            # Default is backup ner type.
            ner_type = backup_ner_type

            # If all tokens have the same ner type, use it.
            possible_ner_types = set()
            for index in best_span:
                token = amr.tokens[index]
                ner_tag = amr.ner_tags[index]
                if ner_tag not in ('0', 'O') or token[0].isupper():
                    possible_ner_types.add(ner_tag)
            possible_ner_types = list(possible_ner_types)
            if len(possible_ner_types) == 1 and possible_ner_types[0] not in ('O', '0'):
                ner_type = list(possible_ner_types)[0]

            # Get the high-frequency ner type from lut.
            entity_mention = ' '.join([amr.tokens[index] for index in best_span]).lower()
            if entity_mention in entity_type_lut:
                entity_type, freq = max(entity_type_lut[entity_mention].items(), key=lambda x: x[1])
                if freq > 50:
                    ner_type = entity_type

            self.span = best_span
            self.ner_type = ner_type
            self.amr_type = amr_type
            self.confidence = confidence
        else:
            self.span = []
            self.ner_type = backup_ner_type
            self.amr_type = amr_type
            self.confidence = 0

    def _clean_span(self, spans, amr):
        # Make sure each op only appears once in a span.
        clean_spans = []
        for span in spans:
            _align = {}
            trivial_indexes = []
            for index in span:
                if index not in self.alignment:
                    trivial_indexes.append(index)
                    continue
                op, confidence = self.alignment[index]
                if op not in _align or (op in _align and _align[op][1] < confidence):
                    _align[op] = (index, confidence)
            indexes = [i for i, _ in _align.values()] + trivial_indexes
            _spans = group_indexes_to_spans(indexes, amr)
            clean_spans.append(max(_spans, key=lambda s: len(s)))

        # Strip trivial tokens
        ret_spans = []
        for span in clean_spans:
            span = strip_span(span, amr.tokens)
            if len(span) > 0:
                ret_spans.append(span)
        return ret_spans
