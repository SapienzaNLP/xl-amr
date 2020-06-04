import re
import json
from collections import defaultdict, Counter

import penman
import networkx as nx

from xlamr_stog.data.vocabulary import DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN
from xlamr_stog.data.dataset_readers.amr_parsing.graph_repair import GraphRepair
from xlamr_stog.utils.string import find_similar_token, is_abstract_token, is_english_punct
from xlamr_stog.utils.src_tgt_match import find_tgt_token_src
from xlamr_stog.utils import logging


logger = logging.init_logger()


# Disable inverting ':mod' relation.
penman.AMRCodec._inversions.pop('domain')
penman.AMRCodec._deinversions.pop('mod')
from penman import Triple

amr_codec = penman.AMRCodec(indent=6)

WORDSENSE_RE = re.compile(r'-\d\d$')
QUOTED_RE = re.compile(r'^".*"$')
import string

def quantify(x):
    if isinstance(x, int) or isinstance(x, float):
        return True
    else:
        if re.search(r"^[0-9]+/[0-9]+$", x):
            return True
        elif re.search(r"^[0-9]+$", x):
            return True
        else:
            return False
class AMR:

    def __init__(self,
                 id=None,
                 sentence=None,
                 graph=None,
                 tokens=None,
                 lemmas=None,
                 pos_tags=None,
                 ner_tags=None,
                 abstract_map=None,
                 misc=None,
                 graph_pred=None):
        self.id = id
        self.sentence = sentence
        self.graph = graph
        self.tokens = tokens
        self.lemmas = lemmas
        self.pos_tags = pos_tags
        self.ner_tags = ner_tags
        self.abstract_map = abstract_map
        self.misc = misc
        self.graph_pred=graph_pred



    def is_named_entity(self, index):
        return self.ner_tags[index] not in ('0', 'O')

    def get_named_entity_span(self, index):
        if self.ner_tags is None or not self.is_named_entity(index):
            return []
        span = [index]
        tag = self.ner_tags[index]
        prev = index - 1
        while prev > 0 and self.ner_tags[prev] == tag:
            span.append(prev)
            prev -= 1
        next = index + 1
        while next < len(self.ner_tags) and self.ner_tags[next] == tag:
            span.append(next)
            next += 1
        return span

    def find_span_indexes(self, span):
        for i, token in enumerate(self.tokens):
            if token == span[0]:
                _span = self.tokens[i: i + len(span)]
                if len(_span) == len(span) and all(x == y for x, y in zip(span, _span)):
                    return list(range(i, i + len(span)))
        return None

    def replace_span(self, indexes, new, pos=None, ner=None):
        self.tokens = self.tokens[:indexes[0]] + new + self.tokens[indexes[-1] + 1:]
        self.lemmas = self.lemmas[:indexes[0]] + new + self.lemmas[indexes[-1] + 1:]
        if pos is None:
            pos = [self.pos_tags[indexes[0]]]
        self.pos_tags = self.pos_tags[:indexes[0]] + pos + self.pos_tags[indexes[-1] + 1:]
        if ner is None:
            ner = [self.ner_tags[indexes[0]]]
        self.ner_tags = self.ner_tags[:indexes[0]] + ner + self.ner_tags[indexes[-1] + 1:]

    def remove_span(self, indexes):
        self.replace_span(indexes, [], [], [])

    def __repr__(self):
        fields = []
        for k, v in dict(
            id=self.id,
            snt=self.sentence,
            tokens=self.tokens,
            lemmas=self.lemmas,
            pos_tags=self.pos_tags,
            ner_tags=self.ner_tags,
            abstract_map=self.abstract_map,
            misc=self.misc,
            graph=self.graph,
            graph_pred=self.graph_pred
        ).items():
            if v is None:
                continue
            if k == 'misc':
                fields += v
            elif k == 'graph':
                try:
                    fields.append(str(v))
                except:
                    fields.append("BAD_GRAPH")
            elif k == 'graph_pred':
                if v==None: continue
                try:
                    fields.append("\n#****GOLD_GRAPH*******\n")
                    fields.append(str(v))
                except:
                    fields.append("BAD_GRAPH")
            else:
                if not isinstance(v, str):
                    v = json.dumps(v)
                fields.append('# ::{} {}'.format(k, v))
        return '\n'.join(fields)

    def get_src_tokens(self):
        return self.lemmas if self.lemmas else self.sentence.split()


class AMRNode:

    attribute_priority = [
        'instance', 'quant', 'mode', 'value', 'name', 'li', 'mod', 'frequency',
        'month', 'day', 'year', 'time', 'unit', 'decade', 'poss'
    ]

    def __init__(self, identifier, attributes=None, copy_of=None):
        self.identifier = identifier
        if attributes is None:
            self.attributes = []
        else:
            self.attributes = attributes
            # self._sort_attributes()
        self._num_copies = 0
        self.copy_of = copy_of

    def _sort_attributes(self):
        def get_attr_priority(attr):
            if attr in self.attribute_priority:
                return self.attribute_priority.index(attr), attr
            if not re.search(r'^(ARG|op|snt)', attr):
                return len(self.attribute_priority), attr
            else:
                return len(self.attribute_priority) + 1, attr
        self.attributes.sort(key=lambda x: get_attr_priority(x[0]))

    def __hash__(self):
        return hash(self.identifier)

    def __eq__(self, other):
        if not isinstance(other, AMRNode):
            return False
        return self.identifier == other.identifier

    def __repr__(self):
        ret = str(self.identifier)
        for k, v in self.attributes:
            if k == 'instance':
                ret += ' / ' + v
                break
        return ret

    def __str__(self):
        ret = repr(self)
        for key, value in self.attributes:
            if key == 'instance':
                continue
            ret += '\n\t:{} {}'.format(key, value)
        return ret

    @property
    def instance(self):
        for key, value in self.attributes:
            if key == 'instance':
                return value
        else:
            return None

    @property
    def ops(self):
        ops = []
        for key, value in self.attributes:
            if re.search(r'op\d+', key):
                ops.append((int(key[2:]), value))
        if len(ops):
            ops.sort(key=lambda x: x[0])
        return [v for k, v in ops]

    def copy(self):
        attributes = None
        if self.attributes is not None:
            attributes = self.attributes[:]
        self._num_copies += 1
        copy = AMRNode(self.identifier + '_copy_{}'.format(self._num_copies), attributes, self)
        return copy

    def remove_attribute(self, attr, value):
        try:
            self.attributes.remove((attr, value))
        except:
            return

    def add_attribute(self, attr, value):
        self.attributes.append((attr, value))

    def replace_attribute(self, attr, old, new):
        index = self.attributes.index((attr, old))
        self.attributes[index] = (attr, new)

    def get_frame_attributes(self):
        for k, v in self.attributes:
            if isinstance(v, str) and re.search(r'-\d\d$', v):
                yield k, v

    def get_senseless_attributes(self):
        for k, v in self.attributes:
            if isinstance(v, str) and not re.search(r'-\d\d$', v):
                yield k, v


class AMRGraph(penman.Graph):

    edge_label_priority = (
        'mod name time location degree poss domain quant manner unit purpose topic condition part-of compared-to '
        'duration source ord beneficiary concession direction frequency consist-of example medium location-of '
        'manner-of quant-of time-of instrument prep-in destination accompanier prep-with extent instrument-of age '
        'path concession-of subevent-of prep-as prep-to prep-against prep-on prep-for degree-of prep-under part '
        'condition-of prep-without topic-of season duration-of poss-of prep-from prep-at range purpose-of source-of '
        'subevent example-of value path-of scale conj-as-if prep-into prep-by prep-on-behalf-of medium-of prep-among '
        'calendar beneficiary-of prep-along-with extent-of age-of frequency-of dayperiod accompanier-of '
        'destination-of prep-amid prep-toward prep-in-addition-to ord-of name-of weekday direction-of prep-out-of '
        'timezone subset-of'.split())

    def __init__(self, penman_graph):
        super(AMRGraph, self).__init__()
        self._triples = penman_graph._triples
        self._top = penman_graph._top
        self._build_extras()
        self._src_tokens = []
        self.anonymization_list = set([line.rstrip() for line in open("data/misc/NM_anonym.txt").readlines()])

    def __str__(self):
        self._triples = penman.alphanum_order(self._triples)
        return amr_codec.encode(self)




    def _build_extras(self):
        G = nx.DiGraph()

        self.variable_to_node = {}
        for v in self.variables():
            if type(v) is not str:
                continue
            attributes = [(t.relation, t.target) for t in self.attributes(source=v)]
            node = AMRNode(v, attributes)
            G.add_node(node)
            self.variable_to_node[v] = node

        edge_set = set()
        for edge in self.edges():
            if type(edge.source) is not str:
                continue
            try:
                source = self.variable_to_node[edge.source]
                target = self.variable_to_node[edge.target]
                relation = edge.relation
            except:
                continue

            if relation == 'instance':
                continue

            if source == target:
                continue

            if edge.inverted:
                source, target, relation = target, source, amr_codec.invert_relation(edge.relation)

            if (source, target) in edge_set:
                target = target.copy()

            edge_set.add((source, target))
            G.add_edge(source, target, label=relation)

        self._G = G

    def attributes(self, source=None, relation=None, target=None):
        # Refine attributes because there's a bug in penman.attributes()
        # See https://github.com/goodmami/penman/issues/29
        attrmatch = lambda a: (
                (source is None or source == a.source) and
                (relation is None or relation == a.relation) and
                (target is None or target == a.target)
        )
        variables = self.variables()
        attrs = [t for t in self.triples() if t.target not in variables or t.relation == 'instance']
        return list(filter(attrmatch, attrs))

    def _update_penman_graph(self, triples):
        self._triples = triples
        if self._top not in self.variables():
            self._top = None

    def is_name_node(self, node):
        edges = list(self._G.in_edges(node))
        return any(self._G[source][target].get('label', None) == 'name' for source, target in edges)

    def get_name_node_type(self, node):
        edges = list(self._G.in_edges(node))
        for source, target in edges:
            if self._G[source][target].get('label', None) == 'name':
                return source.instance
        raise KeyError

    def get_name_node_wiki(self, node):
        edges = list(self._G.in_edges(node))
        for source, target in edges:
            if self._G[source][target].get('label', None) == 'name':
                for attr, value in source.attributes:
                    if attr == 'wiki':
                        if value != '-':
                            value = value[1:-1]  # remove quotes
                        return value
        return None

    def set_name_node_wiki(self, node, wiki):
        edges = list(self._G.in_edges(node))
        parent = None
        for source, target in edges:
            if self._G[source][target].get('label', None) == 'name':
                parent = source
                break
        if parent:
            if wiki != '-':
                wiki = '"{}"'.format(wiki)
            self.add_node_attribute(parent, 'wiki', wiki)

    def is_date_node(self, node):
        return node.instance == 'date-entity'

    def add_edge(self, source, target, label):
        self._G.add_edge(source, target, label=label)
        t = penman.Triple(source=source.identifier, relation=label, target=target.identifier)
        triples = self._triples + [t]
        triples = penman.alphanum_order(triples)
        self._update_penman_graph(triples)

    def remove_edge(self, x, y):
        if isinstance(x, AMRNode) and isinstance(y, AMRNode):
            self._G.remove_edge(x, y)
        if isinstance(x, AMRNode):
            x = x.identifier
        if isinstance(y, AMRNode):
            y = y.identifier
        triples = [t for t in self._triples if not (t.source == x and t.target == y)]
        self._update_penman_graph(triples)

    def update_edge_label(self, x, y, old, new):
        self._G[x][y]['label'] = new
        triples = []
        for t in self._triples:
            if t.source == x.identifier and t.target == y.identifier and t.relation == old:
                t = Triple(x.identifier, new, y.identifier)
            triples.append(t)
        self._update_penman_graph(triples)

    def add_node(self, instance):
        identifier = instance[0]
        assert identifier.isalpha()
        if identifier in self.variables():
            i = 2
            while identifier + str(i) in self.variables():
                i += 1
            identifier += str(i)
        triples = self._triples + [Triple(identifier, 'instance', instance)]
        self._triples = penman.alphanum_order(triples)

        node = AMRNode(identifier, [('instance', instance)])
        self._G.add_node(node)
        return node

    def remove_node(self, node):
        self._G.remove_node(node)
        triples = [t for t in self._triples if t.source != node.identifier]
        self._update_penman_graph(triples)

    def replace_node_attribute(self, node, attr, old, new):
        node.replace_attribute(attr, old, new)
        triples = []
        found = False
        for t in self._triples:
            if t.source == node.identifier and t.relation == attr and t.target == old:
                found = True
                t = penman.Triple(source=node.identifier, relation=attr, target=new)
            triples.append(t)
        if not found:
            raise KeyError
        self._triples = penman.alphanum_order(triples)

    def remove_node_attribute(self, node, attr, value):
        node.remove_attribute(attr, value)
        triples = [t for t in self._triples if not (t.source == node.identifier and t.relation == attr and t.target == value)]
        self._update_penman_graph(triples)

    def add_node_attribute(self, node, attr, value):
        node.add_attribute(attr, value)
        t = penman.Triple(source=node.identifier, relation=attr, target=value)
        self._triples = penman.alphanum_order(self._triples + [t])

    def remove_node_ops(self, node):
        ops = []
        for attr, value in node.attributes:
            if re.search(r'^op\d+$', attr):
                ops.append((attr, value))
        for attr, value in ops:
            self.remove_node_attribute(node, attr, value)

    def remove_subtree(self, root):
        children = []
        removed_nodes = set()
        for _, child in list(self._G.edges(root)):
            self.remove_edge(root, child)
            children.append(child)
        for child in children:
            if len(list(self._G.in_edges(child))) == 0:
                removed_nodes.update(self.remove_subtree(child))
        if len(list(self._G.in_edges(root))) == 0:
            self.remove_node(root)
            removed_nodes.add(root)
        return removed_nodes

    def get_subtree(self, root, max_depth):
        if max_depth == 0:
            return []
        nodes = [root]
        children = [child for _, child in self._G.edges(root)]
        nodes += children
        for child in children:
            if len(list(self._G.in_edges(child))) == 1:
                nodes = nodes + self.get_subtree(child, max_depth - 1)
        return nodes

    def get_nodes(self):
        return self._G.nodes

    def get_edges(self):
        return self._G.edges

    def set_src_tokens(self, sentence):
        if type(sentence) is not list:
            sentence = sentence.split(" ")
        self._src_tokens = sentence

    def get_src_tokens(self):
        return self._src_tokens

    def get_list_node(self, replace_copy=True):
        visited = defaultdict(int)
        node_list = []

        def dfs(node, relation, parent):

            node_list.append((
                node if node.copy_of is None or not replace_copy else node.copy_of,
                relation,
                parent if parent.copy_of is None or not replace_copy else parent.copy_of))

            if len(self._G[node]) > 0 and visited[node] == 0:
                visited[node] = 1
                for child_node, child_relation in self.sort_edges(self._G[node].items()):
                    dfs(child_node, child_relation["label"], node)

        dfs(
            self.variable_to_node[self._top],
            'root',
            self.variable_to_node[self._top]
        )

        return node_list

    def sort_edges(self, edges):
        return edges

    def get_tgt_tokens(self):
        node_list = self.get_list_node()

        tgt_token = []
        visited = defaultdict(int)

        for node, relation, parent_node in node_list:
            instance = [attr[1] for attr in node.attributes if attr[0] == "instance"]
            try:
                assert len(instance) == 1
            except: #Handling silver-data where the graphs might be not well-formed
                continue
            tgt_token.append(str(instance[0]))

            if len(node.attributes) > 1 and visited[node] == 0:
                for attr in node.attributes:
                    if attr[0] != "instance":
                        tgt_token.append(str(attr[1]))

            visited[node] = 1

        return tgt_token

    def add_language_code(self, token, extra):
        if extra:
            if token in string.punctuation:
                return False
            for nm in self.anonymization_list:
                if nm in token:
                    return False
            if quantify(token):
                return False
        return True

    def get_list_data(self, amr, bos=None, eos=None, bert_tokenizer=None, laser_tokenizer=None, max_tgt_length=None, split="test", language = None, source_copy = True, multilingual=False, extra_check=False, translation_mapping=None, tgt_src_replacements=None):
        node_list = self.get_list_node() #Linearinzing the graph nodes through dfs (node, relation, parent)

        tgt_tokens = []
        head_tags = []
        head_indices = []

        node_to_idx = defaultdict(list)
        visited = defaultdict(int)

        def update_info(node, relation, parent, token):
            head_indices.append(1 + node_to_idx[parent][-1])
            head_tags.append(relation)
            tgt_tokens.append(str(token))

        for node, relation, parent_node in node_list:
            node_to_idx[node].append(len(tgt_tokens))
            instance = [attr[1] for attr in node.attributes if attr[0] == "instance"]
            try:
                assert len(instance) == 1
            except: #Handling silver-data where the graphs might be not well-formed
                continue

            instance = instance[0]

            update_info(node, relation, parent_node, instance)

            if len(node.attributes) > 1 and visited[node] == 0:
                for attr in node.attributes:
                    if attr[0] != "instance":
                        update_info(node, attr[0], node, attr[1])

            visited[node] = 1

        def trim_very_long_tgt_tokens(tgt_tokens, head_tags, head_indices, node_to_idx):
            tgt_tokens = tgt_tokens[:max_tgt_length]
            head_tags = head_tags[:max_tgt_length]
            head_indices = head_indices[:max_tgt_length]
            for node, indices in node_to_idx.items():
                invalid_indices = [index for index in indices if index >= max_tgt_length]
                for index in invalid_indices:
                    indices.remove(index)
            return tgt_tokens, head_tags, head_indices, node_to_idx

        if max_tgt_length is not None:
            tgt_tokens, head_tags, head_indices, node_to_idx = trim_very_long_tgt_tokens(
                tgt_tokens, head_tags, head_indices, node_to_idx)

        copy_offset = 0
        if bos:
            tgt_tokens = [bos] + tgt_tokens
            copy_offset += 1
        if eos:
            tgt_tokens = tgt_tokens + [eos]

        head_indices[node_to_idx[self.variable_to_node[self.top]][0]] = 0

        # Target side Coreference
        tgt_copy_indices = [i for i in range(len(tgt_tokens))]

        for node, indices in node_to_idx.items():
            if len(indices) > 1:
                copy_idx = indices[0] + copy_offset
                for token_idx in indices[1:]:
                    tgt_copy_indices[token_idx + copy_offset] = copy_idx

        tgt_copy_map = [(token_idx, copy_idx) for token_idx, copy_idx in enumerate(tgt_copy_indices)]

        for i, copy_index in enumerate(tgt_copy_indices):
            # Set the coreferred target to 0 if no coref is available.
            if i == copy_index:
                tgt_copy_indices[i] = 0

        tgt_token_counter = Counter(tgt_tokens)
        tgt_copy_mask = [0] * len(tgt_tokens)
        for i, token in enumerate(tgt_tokens):
            if tgt_token_counter[token] > 1:
                tgt_copy_mask[i] = 1

        def add_source_side_tags_to_target_side(_src_tokens, _src_tags, source_copy = True):
            assert len(_src_tags) == len(_src_tokens)

            tag_counter = defaultdict(lambda: defaultdict(int))
            for src_token, src_tag in zip(_src_tokens, _src_tags):
                tag_counter[src_token][src_tag] += 1

            tag_lut = {DEFAULT_OOV_TOKEN: DEFAULT_OOV_TOKEN,
                       DEFAULT_PADDING_TOKEN: DEFAULT_OOV_TOKEN}
            for src_token in set(_src_tokens):
                tag = max(tag_counter[src_token].keys(), key=lambda x: tag_counter[src_token][x])
                tag_lut[src_token] = tag

            tgt_tags = []
            for tgt_token in tgt_tokens:
                if source_copy:
                    sim_token = find_similar_token(tgt_token, _src_tokens)
                    if sim_token is not None:
                        index = _src_tokens.index(sim_token)
                        tag = _src_tags[index]
                    else:
                        tag = DEFAULT_OOV_TOKEN
                else:
                    tag = DEFAULT_OOV_TOKEN
                tgt_tags.append(tag)

            return tgt_tags, tag_lut


        def src_copy_nodes_pos(_src_copy_vocab, _src_tokens, _src_tags, _tgt_tokens):
            assert len(_src_tags) == len(_src_tokens)

            tag_counter = defaultdict(lambda: defaultdict(int))
            for src_token, src_tag in zip(_src_tokens, _src_tags):
                tag_counter[src_token][src_tag] += 1

            tag_lut = {DEFAULT_OOV_TOKEN: DEFAULT_OOV_TOKEN,
                       DEFAULT_PADDING_TOKEN: DEFAULT_OOV_TOKEN}
            for src_token in set(_src_tokens):
                tag = max(tag_counter[src_token].keys(), key=lambda x: tag_counter[src_token][x])
                tag_lut[src_token] = tag

            tgt_tags = []
            _src_copy_indices = []
            # src_copy_indices = [src_copy_vocab.token_to_idx[src_copy_vocab.unk_token] for _ in tgt_tokens]
            for tgt_token in tgt_tokens:
                sim_token = find_tgt_token_src(tgt_token, src_tokens,
                                               src_postags=tag_lut,
                                               language=language,
                                               translation_mappings=translation_mapping,
                                               replacements=tgt_src_replacements[language],
                                               training=split=="train")
                if sim_token is not None:
                    index = _src_tokens.index(sim_token)
                    tag = _src_tags[index]
                    _src_copy_indices.append(_src_copy_vocab.get_token_idx(sim_token))
                else:
                    tag = DEFAULT_OOV_TOKEN
                    _src_copy_indices.append(src_copy_vocab.token_to_idx[src_copy_vocab.unk_token])
                tgt_tags.append(tag)

            return _src_copy_indices, tgt_tags, tag_lut


        # Source Copy
        src_tokens = self.get_src_tokens()
        src_token_ids = None
        src_token_subword_index = None
        bpe_src_token_ids = None
        bpe_src_token_subword_index = None
        src_pos_tags = amr.pos_tags
        src_copy_vocab = SourceCopyVocabulary(src_tokens)
        src_copy_map = src_copy_vocab.get_copy_map(src_tokens)

        if source_copy:
            if language =="en":
                src_copy_indices = src_copy_vocab.index_sequence(tgt_tokens)
                tgt_pos_tags, pos_tag_lut = add_source_side_tags_to_target_side(src_tokens, src_pos_tags)
                src_must_copy_tags = [1 if is_abstract_token(t) else 0 for t in src_tokens]
            else:
                # TODO: this should be modified for other languages to copy based on vector similarity not word form
                # The source vocab is in non-english while the tgt_tokens are always in English
                src_copy_indices, tgt_pos_tags, pos_tag_lut = src_copy_nodes_pos(src_copy_vocab, src_tokens, src_pos_tags, tgt_tokens)
                # src_copy_indices = src_copy_vocab.index_sequence(tgt_tokens)
                # tgt_pos_tags, pos_tag_lut = add_source_side_tags_to_target_side(src_tokens, src_pos_tags)
                src_must_copy_tags = [1 if is_abstract_token(t) else 0 for t in src_tokens]

        else:
            # Disable all source copy info
            src_copy_indices = [src_copy_vocab.token_to_idx[src_copy_vocab.unk_token] for _ in tgt_tokens]
            tgt_pos_tags, pos_tag_lut = add_source_side_tags_to_target_side(src_tokens, src_pos_tags, source_copy)
            src_must_copy_tags = [0 for _ in src_tokens]

        if bert_tokenizer is not None:
            src_token_ids, src_token_subword_index = bert_tokenizer.tokenize(src_tokens, True)

        if laser_tokenizer is not None:
            bpe_src_token_ids, bpe_src_token_subword_index = laser_tokenizer.tokenize(src_tokens, True)

        src_copy_invalid_ids = set(src_copy_vocab.index_sequence([t for t in src_tokens if is_english_punct(t)]))

        return_tgt = tgt_tokens
        return_src=src_tokens
        if multilingual and extra_check:
            return_tgt = ["en_" + t if (t not in [eos, bos] and self.add_language_code(t, extra_check)) else t for t in tgt_tokens]
            return_src = [language + "_" + t if (t not in [eos, bos] and self.add_language_code(t, extra_check)) else t for t in src_tokens]
        elif multilingual:
            return_tgt = ["en_" + t if t not in [eos, bos] else t for t in
                          tgt_tokens]
            return_src = [language + "_" + t if (t not in [eos, bos] and self.add_language_code(t, extra_check)) else t
                          for t in src_tokens]

        return {
            "tgt_tokens": return_tgt,
            "tgt_pos_tags": tgt_pos_tags,
            "tgt_copy_indices": tgt_copy_indices,
            "tgt_copy_map": tgt_copy_map,
            "tgt_copy_mask": tgt_copy_mask,
            "src_tokens": return_src,
            "src_token_ids": src_token_ids,
            "src_token_subword_index": src_token_subword_index,
            "bpe_src_token_ids": bpe_src_token_ids,
            "bpe_src_token_subword_index": bpe_src_token_subword_index,
            "src_must_copy_tags": src_must_copy_tags,
            "src_pos_tags": src_pos_tags,
            "src_copy_vocab": src_copy_vocab,
            "src_copy_indices": src_copy_indices,
            "src_copy_map": src_copy_map,
            "pos_tag_lut": pos_tag_lut,
            "head_tags": head_tags,
            "head_indices": head_indices,
            "src_copy_invalid_ids": src_copy_invalid_ids
        }


    @classmethod
    def decode(cls, raw_graph_string):
        _graph = amr_codec.decode(raw_graph_string)
        return cls(_graph)

    @classmethod
    def from_lists(cls, all_list):
        head_tags = all_list['head_tags']
        head_indices = all_list['head_indices']
        tgt_tokens = all_list['tokens']

        tgt_copy_indices = all_list['coref']
        variables = []
        variables_count = defaultdict(int)
        for i, token in enumerate(tgt_tokens):
            if tgt_copy_indices[i] != i:
                variables.append(variables[tgt_copy_indices[i]])
            else:
                if token[0] in variables_count:
                    variables.append(token[0] + str(variables_count[token[0]]))
                else:
                    variables.append(token[0])

                variables_count[token[0]] += 1

        Triples = []
        for variable, token in zip(variables, tgt_tokens):
            Triples.append(Triple(variable, "instance", token))
            Triples.append(
                Triple(
                    head_indices[variable],
                    head_tags[variable],
                    variable
                )
            )

    @classmethod
    def from_prediction(cls, prediction):

        def is_attribute_value(value):
            return re.search(r'(^".*"$|^[^a-zA-Z]+$)', value) is not None

        def is_attribute_edge(label):
            return label in ('instance', 'mode', 'li', 'value', 'month', 'year', 'day', 'decade', 'ARG6')

        def normalize_number(text):
            if re.search(r'^\d+,\d+$', text):
                text = text.replace(',', '')
            return text

        def normalize_lang(text):
            if text.startswith("en_") or text.startswith("it_"):
                return text[3:]
            return text

        def abstract_node(value):
            return re.search(r'^([A-Z]+|DATE_ATTRS|SCORE_ENTITY|ORDINAL_ENTITY)_\d+$', value)

        def abstract_attribute(value):
            return re.search(r'^_QUANTITY_\d+$', value)

        def correct_multiroot(heads):
            for i in range(1, len(heads)):
                if heads[i] == 0:
                    heads[i] = 1
            return heads

        nodes = [normalize_number(n) for n in prediction['nodes']]
        nodes = [normalize_lang(n) for n in nodes]
        heads = correct_multiroot(prediction['heads'])
        corefs = [int(x) for x in prediction['corefs']]
        head_labels = prediction['head_labels']

        triples = []
        top = None
        # Build the variable map from variable to instance.
        variable_map = {}
        for coref_index in corefs:
            node = nodes[coref_index - 1]
            head_label = head_labels[coref_index - 1]
            if (re.search(r'[/:\\()]', node) or is_attribute_value(node) or
                    is_attribute_edge(head_label) or abstract_attribute(node)):
                continue
            variable_map['vv{}'.format(coref_index)] = node
        for head_index in heads:
            if head_index == 0:
                continue
            node = nodes[head_index - 1]
            coref_index = corefs[head_index - 1]
            variable_map['vv{}'.format(coref_index)] = node
        # Build edge triples and other attribute triples.
        for i, head_index in enumerate(heads):
            if head_index == 0:
                top_variable = 'vv{}'.format(corefs[i])
                if top_variable not in variable_map:
                    variable_map[top_variable] = nodes[i]
                top = top_variable
                continue
            head_variable = 'vv{}'.format(corefs[head_index - 1])
            modifier = nodes[i]
            modifier_variable = 'vv{}'.format(corefs[i])
            label = head_labels[i]
            assert head_variable in variable_map
            if modifier_variable in variable_map:
                triples.append((head_variable, label, modifier_variable))
            else:
                # Add quotes if there's a backslash.
                if re.search(r'[/:\\()]', modifier) and not re.search(r'^".*"$', modifier):
                    modifier = '"{}"'.format(modifier)
                triples.append((head_variable, label, modifier))

        for var, node in variable_map.items():
            if re.search(r'^".*"$', node):
                node = node[1:-1]
            if re.search(r'[/:\\()]', node):
                parts = re.split(r'[/:\\()]', node)
                for part in parts[::-1]:
                    if len(part):
                        node = part
                        break
                else:
                    node = re.sub(r'[/:\\()]', '_', node)
            triples.append((var, 'instance', node))

        if len(triples) == 0:
            triples.append(('vv1', 'instance', 'string-entity'))
            top = 'vv1'
        triples.sort(key=lambda x: int(x[0].replace('vv', '')))
        graph = penman.Graph()
        graph._top = top
        graph._triples = [penman.Triple(*t) for t in triples]
        graph = cls(graph)
        try:
            GraphRepair.do(graph, nodes)
            amr_codec.encode(graph)
        except Exception as e:
            graph._top = top
            graph._triples = [penman.Triple(*t) for t in triples]
            graph = cls(graph)
        return graph


class SourceCopyVocabulary:
    def __init__(self, sentence, pad_token=DEFAULT_PADDING_TOKEN, unk_token=DEFAULT_OOV_TOKEN):
        if type(sentence) is not list:
            sentence = sentence.split(" ")

        self.src_tokens = sentence
        self.pad_token = pad_token
        self.unk_token = unk_token

        self.token_to_idx = {self.pad_token : 0, self.unk_token : 1}
        self.idx_to_token = {0 : self.pad_token, 1 : self.unk_token}

        self.vocab_size = 2

        for token in sentence:
            if token not in self.token_to_idx:
                self.token_to_idx[token] = self.vocab_size
                self.idx_to_token[self.vocab_size] = token
                self.vocab_size += 1

    def get_token_from_idx(self, idx):
        return self.idx_to_token[idx]

    def get_token_idx(self, token):
        return self.token_to_idx.get(token, self.token_to_idx[self.unk_token])

    def index_sequence(self, list_tokens):
        return [self.get_token_idx(token) for token in list_tokens]

    def get_copy_map(self, list_tokens):
        src_indices = [self.get_token_idx(self.unk_token)] + self.index_sequence(list_tokens)
        return [
            (src_idx, src_token_idx) for src_idx, src_token_idx in enumerate(src_indices)
        ]

    def get_special_tok_list(self):
        return [self.pad_token, self.unk_token]

    def __repr__(self):
        return json.dumps(self.idx_to_token)
