# Translation

## Gold (test set)

For the gold data, we're using professional translation to translate each document in test set. We're using the combination of 1 translator - 1 proof reader and have 2 iteration of revision. Afterward we format the data accordingly. With the order of : `file_list = ['bolt', 'consensus', 'dfa', 'proxy', 'xinhua']`. later these files need to be placed in `data/amr_2.0/translation/` with filename of `amr-release-2.0-amrs-test-{doc}.sentences.{lang}.txt`, for example: `amr-release-2.0-amrs-test-bolt.sentences.ms.txt`

## Silver (train-dev set)

For silver data we're using MarianMT en-id model that's hosted by [huggingface](https://huggingface.co/Helsinki-NLP/opus-mt-en-id). we're going to translate it

# Projection

## Gold (test set)

For projection we need to change a few thing in [project_test.py](../xlamr_stog/data/dataset_readers/amr_projection/project_test.py). first we need to change the lang
## Silver (train-dev set)

# Preprocess

Gold vs Silver, for now we need to run it twice with corrseponding argument. The gold should treat the task as normal AMR task, but the silver will treat it as crosslingual AMR. In the sense of the utils needed and the resources.

## Recategorize

Consider adding assertion to check if the following file exists:
- babelnet

also if you're building utils, make sure you pass the arguments `amr_train_file` and 

### `data/cross-lingual-babelnet_mappings/name_en_ms_bn_map.json`
Mapping from NER in `en` to `ms` from babelnet (how to get it?) We could use existing mapping to get the keys. And for lemma that we can't find the sense of we shall not include it in the file. For our case, since babelnet API is restricted to 1000 call/day, make sure to either csk for research account or schedule multiple day for running the script `INSERT SCRIPT NAME`
```JSON
{
  "__LICENSE__": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License",
  "Broadway": [
    "Gateway_Theatre",
    ...
  ],
  "Olympic_Council_of_Asia": [
    "Olympic_Council_of_Asia"
  ],
  "Paper": [
    "Paper",
    ...
  ],
  ...
}
```

also beside this one, there's also `name_span_en_{lang}_map_amr_bn`  that maps a named entity with corrensponding span and its count in train-dev set, which can be generated from `name_en_{lang}_bn_map` for counting language specific span and `name_op_cooccur_counter_en_{lang}` for counting english language span. It'll be done in conjunction with recategorizer when building utils. 
> Make sure to move the result from utils folder to crosslingual folder.

### Stemmer
I also need alternative for snowballstemmer, should I use sastrawi?

### Stopwords
inside `data/cross-lingual-babelnet_mappings/stopwords_ms.txt` we need the format to be newline delimited token:
```
-
aber
alle
allem
allen
aller
alles
...
```
For now we use the list from `Sastrawi` project. consider adding more based on [Rahutomo et. al (2019)](https://jtiik.ub.ac.id/index.php/jtiik/article/view/1226)

## Remove Senses

### `senseless_node_counter`

This file is the same with original utils. So we could just copy-paste this. Similar to theses files:
- frame_lemma_counter
- countries
- lemma_frame_counter
