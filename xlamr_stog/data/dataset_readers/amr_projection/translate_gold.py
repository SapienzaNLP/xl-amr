import torch
from transformers import MarianTokenizer, MarianMTModel
from typing import List

import os

def batches(lines):
    i=0
    b = []
    for line in lines:
        b.append(line)
        i+=1
        if i == 50:
            yield b
            i=0
            b=[]
    if len(b) > 0:
        yield b


if __name__ == '__main__':
    #translator = Translator(service_urls = ['translate.google.com', 'translate.google.it'])
    src = 'en'  # source language
    trg = 'id'  # target language
    #mname = f'Helsinki-NLP/opus-mt-{src}-{trg}'
    mname=f'Helsinki-NLP/opus-mt-ROMANCE-en'
    device = torch.device("cuda")
    model = MarianMTModel.from_pretrained(mname).to(device)
    tok = MarianTokenizer.from_pretrained(mname)

    fs = sorted([d for d in os.listdir('amr_sentences/') if d.startswith('amr')])
    for f in fs:
        print(f)
        T = []
        with open(os.path.join('amr_sentences/', f)) as fr:
            lines = fr.readlines()
        lines = ['>>es<< '+line.replace('\n','') for line in lines]
        i = 0
        b = []
        for line in lines:
            b.append(line)
            i += 1
            if i == 50:

                batch = tok.prepare_translation_batch(src_texts=b)
                translated = model.generate(**batch.to(device))
                T = [tok.decode(t, skip_special_tokens=True) for t in translated]
                with open(os.path.join('amr_sentences/', ''.join(f.split('.')[:-1]) + '_{}.txt'.format(trg)), 'a') as fw:
                    for t in T:
                        fw.write(t+'\n')
                i = 0
                b = []
        batch = tok.prepare_translation_batch(src_texts=b)
        translated = model.generate(**batch.to(device))
        T = [tok.decode(t, skip_special_tokens=True) for t in translated]
        with open(os.path.join('amr_sentences/', ''.join(f.split('.')[:-1]) + '_{}.txt'.format(trg)), 'a') as fw:
            for t in T:
                fw.write(t + '\n')
