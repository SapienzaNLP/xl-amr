import os
import stanza
import numpy as np
from xlamr_stog.data.dataset_readers.amr_parsing.io import AMRIO
from xlamr_stog.utils import logging
import subprocess
import requests
from pycorenlp import StanfordCoreNLP
import json
logger = logging.init_logger()

class FeatureAnnotator(object):

    def __init__(self, lang="it", pipeline=None):
        self.lang = lang
        self.u_pos = dict()
        self.c_ner = {"LOC": "LOCATION", "ORG": "ORGANIZATION", "PER":"PERSON", "MISC":"O", "O":"O"}
        self.convert_postags()

        if self.lang=="it":
            self.server_url = "http://localhost:9200/tint"
            self.nlp_properties = {
                'annotators': "tokenize,ssplit,pos,lemma,ner",
                "tokenize.options": "splitHyphenated=true,normalizeParentheses=false",
                "tokenize.whitespace": False,
                'ssplit.isOneSentence': True,
                'outputFormat': 'json'

            }
        else:
            self.nlp_pipeline = pipeline


    def annotate_file(self, in_path, out):
        with open(out, 'w', encoding='utf-8') as f:
            for i, amr in enumerate(AMRIO.read(os.path.join(in_path))):
                if i % 1000 == 0:
                    logger.info('{} processed.'.format(i))

                sentence = amr.sentence
                if self.lang=="it":
                    annotation = self.tint_annotate(sentence.replace("[ ... ]",""))
                else:
                    annotation = self.stanza_annotate(sentence)
                amr.tokens = annotation['tokens']
                amr.lemmas = annotation['lemmas']
                amr.pos_tags = annotation['pos_tags']
                amr.ner_tags = annotation['ner_tags']
                amr.abstract_map = {}
                AMRIO.dump([amr], f)


    def server_annotate(self, text, properties=None):
        assert isinstance(text, str)
        if properties is None:
            properties = {}
        else:
            assert isinstance(properties, dict)

        # Checks that the Tint server is started.
        try:
            requests.get(self.server_url)
        except requests.exceptions.ConnectionError:
            raise Exception('Check whether you have started the Tint server e.g.\n'
            '$ cd tools/tint/ \n'
            '$ ./tint-server.sh -p 9200')

        r = requests.post(
            self.server_url, params={
                'properties': str(properties)
            }, data={"text":text}, headers={'Connection': 'close'})
        output = r.text
        if ('outputFormat' in properties
             and properties['outputFormat'] == 'json'):
            try:
                output = json.loads(output, encoding='utf-8', strict=True)
            except:
                pass
        return output


    def tint_annotate(self, text):
        tokens = self.server_annotate(text.strip(), self.nlp_properties)
        try:
            tokens = tokens['sentences']#[0]['tokens']
        except:
            # print(text)
            tokens=[]
        output = dict(
            tokens=[], lemmas=[], pos_tags=[], ner_tags=[]
        )
        for sentence in tokens:
            for token in sentence['tokens']:
                output['tokens'].append(token['word'])
                output['lemmas'].append(token['lemma'])
                output['pos_tags'].append(self.u_pos[token['pos']] if token['pos'] in self.u_pos else "X")
                output['ner_tags'].append(self.c_ner[token['ner']] if token["ner"] in self.c_ner else token["ner"])
        return output


    def stanza_annotate(self, text):
        tokens = dict()
        tokens['tokens'] = list()
        tokens['lemmas'] = list()
        tokens['pos_tags'] = list()
        tokens['ner_tags'] = list()
        try:
            doc = self.nlp_pipeline(text)
        except:
            return dict(tokens=[], lemmas=[], pos_tags=[], ner_tags=[])
        entities = doc.entities
        start2idx = dict()
        end2idx = dict()
        i=0
        for s in doc.sentences:
            for t in s.words:

                tokens['tokens'].append(t.text)
                tokens['lemmas'].append(t.lemma)
                tokens['pos_tags'].append(t.upos)
                tokens['ner_tags'].append('O')
                if t.misc is not None:
                    start2idx[int(t.misc.split("|")[0].split("=")[1])] = i
                    end2idx[int(t.misc.split("|")[1].split("=")[1])] = i
                i += 1

        for nm in entities:
            try:
                start_idx = start2idx[nm.start_char]
                end_idx = end2idx[nm.end_char]
                if nm.type in self.c_ner:
                    nm_type = self.c_ner[nm.type]
                else:
                    nm_type = nm.type
                for tok_idx in range(start_idx,end_idx+1):
                    tokens['ner_tags'][tok_idx] = nm_type
            except:
                print(text, nm)

        return tokens


    def convert_postags(self):
        with open("data/misc/pos-conversion-table.{}".format(self.lang),"r") as infile:
            for line in infile:
                fields = line.rstrip().split()
                self.u_pos[fields[0]] = fields[1]



def main(in_path, dataset):
    out_path=in_path+".features"
    dataset.annotate_file(in_path, out_path)

if __name__ =="__main__":

    import argparse

    parser = argparse.ArgumentParser('feature_annotator_multilingual.py')
    parser.add_argument('files', nargs='+', help='files to annotate.')
    parser.add_argument('--lang', default="it", help='Pipeline language.')
    parser.add_argument('--dataset_type', default="silver", help='.')

    args = parser.parse_args()

    print("Annotating {} dataset!".format(args.lang))
    if args.lang=="it":
        annotator = FeatureAnnotator(args.lang)
    else:
        pipeline = stanza.Pipeline(args.lang, processors="tokenize,mwt,pos,lemma,ner", tokenize_no_ssplit=True,
                              use_gpu=True)
        annotator = FeatureAnnotator(args.lang, pipeline=pipeline)

    for file in args.files:
        main(file, annotator)


