import os
from xlamr_stog.data.dataset_readers.amr_parsing.io import AMRIO
from xlamr_stog.utils import logging
logger = logging.init_logger()

class Dataset(object):

    def __init__(self,in_path=None, dump_dir=None, lang="it", split ="train"):
        self.lang = lang
        self.in_path = in_path
        self.dump_dir = dump_dir
        self.split = split
        self.translations=dict()

    def read_translations(self, lang_sentences):
        for i, amr in enumerate(AMRIO.read(lang_sentences)):
            if amr.id not in self.translations:
                self.translations[amr.id]=amr


    def read_file_gold_amr(self, lang_sentences):
        self.read_translations(lang_sentences)

        with open(self.dump_dir + '{}_{}.txt'.format(self.split, self.lang), 'w', encoding='utf-8') as f:
            for i, amr in enumerate(AMRIO.read(os.path.join(self.in_path))):
                if i % 1000 == 0:
                    logger.info('{} processed.'.format(i))

                sentence = amr.sentence
                if amr.id not in self.translations: continue
                parallel_sentence_amr = self.translations[amr.id]

                amr.sentence = parallel_sentence_amr.sentence
                amr.tokens = None
                amr.lemmas = None
                amr.pos_tags = None
                amr.ner_tags = None
                amr.misc = ["# ::tok-{}".format("en") + " " + sentence]
                amr.abstract_map = {}
                AMRIO.dump([amr], f)


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser('project_train_dev.py')
    parser.add_argument('--amr_path', help='file dir from which to get AMR graphs.')
    parser.add_argument('--trans_path', help='file dir from which to get translations.')
    parser.add_argument('--out_path', help='dump dir.')


    args = parser.parse_args()


    for split in ["dev","train"]:
        for lang in ["es","de", "it"]:
            amr_file=os.path.join(args.amr_path, "{}.txt".format(split))
            out_path=args.out_path
            dataset = Dataset(amr_file, out_path, lang, split)
            dataset.read_file_gold_amr(os.path.join(args.trans_path, "{}_{}.txt".format(split,lang)))