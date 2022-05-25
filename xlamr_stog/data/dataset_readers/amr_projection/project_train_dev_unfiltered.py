import os
from xlamr_stog.data.dataset_readers.amr_parsing.io import AMRIO
from xlamr_stog.utils import logging
logger = logging.init_logger()

class Dataset(object):

    def __init__(self,in_path=None, dump_dir=None, lang="ms", split ="train"):
        self.lang = lang
        self.in_path = in_path
        self.dump_dir = dump_dir
        self.split = split

    def read_file_gold_amr(self, lang_sentences):
        with open(self.dump_dir + '{}_{}.txt'.format(self.split, self.lang), 'w', encoding='utf-8') as f:
            for i, amr in enumerate(AMRIO.read(os.path.join(self.in_path, split+".txt"))):
                if i % 1000 == 0:
                    logger.info('{} processed.'.format(i))

                sentence = amr.sentence
                parallel_sentence = lang_sentences[i]

                amr.sentence = parallel_sentence
                amr.tokens = None
                amr.lemmas = None
                amr.pos_tags = None
                amr.ner_tags = None
                amr.misc = ["# ::tok-{}".format("en") + " " + sentence]
                amr.abstract_map = {}
                AMRIO.dump([amr], f)


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser('project_test.py')
    parser.add_argument('--amr_path', help='directory from which to get AMR graphs.')
    parser.add_argument('--trans_path', help='file from which to get translated sentences.')
    parser.add_argument('--out_path', help='output directory.')

    args = parser.parse_args()
    amr_path = args.amr_path
    translations = args.trans_path
    out_path = args.out_path
    
    for lang in ["ms"]:
        for split in ["train", "dev"]:
            print("Processing {}".format(split))
            lang_sentences=[]
            with open(os.path.join(translations,"{}_{}.txt".format(split,lang)),"r", encoding='utf-8') as infile:
                for line in infile:
                    lang_sentences.append(line.rstrip())

            dataset = Dataset(amr_path, out_path, lang.lower(), split)
            dataset.read_file_gold_amr(lang_sentences)