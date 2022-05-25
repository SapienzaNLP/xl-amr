import requests
import argparse
import json
from tqdm.auto import tqdm

class Babelnet:
    def __init__(self, key, src_lang, tgt_lang, lemmas=[]):
        self.key = key
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.lemmas = lemmas
        self.senses = dict()

    def get_senses_from_lemmas(self):
        base_url = "https://babelnet.io/v6/getSenses"
        params = {
            "searchLang":self.src_lang,
            "targetLang":self.tgt_lang,
            "key":self.key}
        empty_senses = []
        for lemma in tqdm(self.lemmas):
            try:
                params["lemma"] = lemma
                res = requests.get(base_url, params=params)
                senses = list({i["properties"]["fullLemma"] for i in res.json()})
                if senses:
                    self.senses[lemma] = senses
                else:
                    self.senses[lemma] = lemma
                    empty_senses.append(lemma)
            except Exception as e:
                print(e)
                if not empty_senses:
                    print(f"with empty senses:\n{empty_senses}")
                self.dump_senses_to_file(f"name_{self.src_lang}_{self.tgt_lang}_bn_map_unfinished.json")
                raise Exception("API Limit reached")
        
        if not empty_senses:
            print(f"with empty senses:\n{empty_senses}")
     
    def dump_senses_to_file(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self.senses,f, indent=4)

    #TODO: Handle lemmas and senses change

    @classmethod
    def from_json_file(cls, filepath, key, src_lang, tgt_lang):
        lemmas = []
        with open(filepath, "r") as f:
            lemmas = list(json.load(f).keys())
        if lemmas[0] == '__LICENSE__': lemmas=lemmas[1:]
        return cls(
            key=key, 
            src_lang=src_lang, 
            tgt_lang=tgt_lang, 
            lemmas=lemmas)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('babelnet_sense.py')
    parser.add_argument('--example_mapping')
    parser.add_argument('--target_lang')
    parser.add_argument('--bn_api_key')

    args = parser.parse_args()

    bn = Babelnet.from_json_file(
        args.example_mapping, 
        args.bn_api_key,
        "EN",
        args.target_lang)
    bn.get_senses_from_lemmas()
    bn.dump_senses_to_file(f"name_{bn.src_lang.lower()}_{bn.tgt_lang.lower()}_bn_map.json")