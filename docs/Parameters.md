# Parameters Meaning

- [Parameters Meaning](#parameters-meaning)
  - [Zero-Shot](#zero-shot)
  - [XL-AMR Parallel](#xl-amr-parallel)
    - [`par_amr` vs `par`](#par_amr-vs-par)
    - [Bilingual](#bilingual)
    - [Multilingual](#multilingual)
    - [Language Specific](#language-specific)
  - [XL-AMR Translation](#xl-amr-translation)
  - [XL-AMR Parallel and Translaction](#xl-amr-parallel-and-translaction)
  - [Comparing with Provided Checkpoint](#comparing-with-provided-checkpoint)
- [How to adapt it to Indonesian Language?](#how-to-adapt-it-to-indonesian-language)
  - [Resources](#resources)

This part will document about the difference and context between current parameter to adapt it into Indonesian Language. To make it clear, all of the data for `mimick_test` is from AMR 2.0 gold-standard translated sentence. Also somehow output prediction will be saved into `test.mult.*` file.

## Zero-Shot

In Cross Lingual AMR, zero shot means that we're training our model using only resources in English, and later will be tested against the target language. Because the params files will only contains configuration for training process, we can see it as training a **normal AMR but using multilingual vocabulary and embedding**. On second glance, it seems to use translated AMR 2.0 test data on its `mimick_test`.

If we're talking about zero-shot learning, the only gold-standard resources of english sentence is coming from AMR 2.0. So the params of [`xl-amr_zeroshot_amr.yaml`](../params/zeroshot/xl-amr_zeroshot_amr.yaml) **only uses AMR 2.0 instances for both train and dev**. As for the other params [`xl-amr_zeroshot_par_amr.yaml`](../params/zeroshot/xl-amr_zeroshot_par_amr.yaml), the only discerning feature is that it uses europarl data in addition to AMR 2.0. This makes sense because assuming that cross lingual AMR won't perform better than standard AMR, by adding gold-standard english sentence with infered graph hopefully will help by the sheer amount of additional data. While targeting the performance of at least standard AMR.

Also another setting that's exclusive for zero-shot configuration is that the `encoder_token_embedding` (numberbatch) is not trainable, and we also have fixed vocabulary.

> How fixed vocab and non-trainable token embedding affect the training process?

## XL-AMR Parallel

XL-AMR Par mainly means that the training and dev set 

### `par_amr` vs `par`

Params inside this category consists of two big categories, `xl-amr_par_amr_*` and `xl-amr_par_*`. Parameter files which has infix of `_amr_` means that it uses AMR 2.0 English dataset for its train and dev set. While those without `_amr_` infix only uses europarl dataset for both train and dev.

### Bilingual

Bilingual means for the train and dev, it uses both english language and a target language. For example `en_de`, `en_es`, etc. While both language will be taken from paralel corpus (in this case `Europarl`), whether we'll also use english data from AMR 2.0 depends on whether the params has `_amr_` infix.

Another notable observation is that somehow for `*_par[_amr]_bilingual_en_es.yaml` an  `*_par[_amr]_bilingual_en_it.yaml`, the test data is different than the rest of parallel configuration. They use `amr_2.0/dev.txt.features.preproc` for the test data while the rest uses `amr_2.0/dev.txt.features.recat` for the test data. And somehow there's no trace of `.preproc` suffix anywhere in preprocessing except inside `stog` codebase.

> Is `.preproc` an artifact of the past that needs to be adapted to `.recat` ? or is there something I'm missing.

### Multilingual

For parallel multilingual, the params will use all of the europarl data for both train and dev set. Whether the test data comes from amr 2.0 or europarl english will depends on whether there's `_amr_` infix or not.

### Language Specific

Language Specific `xl-amr_par_lang_[de|es|it].yaml` only used each own target language for all of the train/dev/test process. Since there are no gold-standard AMR data for each language specific, none of these params will have `_amr_` infix in it.

## XL-AMR Translation

Overall, these parameter is similar to those of [Parallel](#xl-amr-parallel) params. One difference is that there's no `trans_bilingual` bilingual only exists for `trans_amr_bilingual`. Because of the nature of trans silver data where we translate AMR 2.0 to target language to get silver train/dev set, we can't have bilingual configuration without including the Gold AMR and making it `trans_amr_bilingual`.

Another thing to notice is that for trans params, the `vocabulary` for `decoder_token_ids` has `max_vocab_size` of 12200 compared to undefined in either parallel params or zeroshot params.

## XL-AMR Parallel and Translaction

This params is not included in the repository, but looking we need to cross check on how it works inside the paper.
> Actually after cross-checking the paper, it turns out this is not included into the configuration

## Comparing with Provided Checkpoint

Also just in case, the config used for pretrained model is the same as the params given in the params folder for each pretrained configurations.

# How to adapt it to Indonesian Language?

## Resources

The following is several resources of configuration that may need to be changed to get the model working for Indonesian Language because it's needed by the params.
> Outside theses configs, we also need several resources not mentioned here e.g. NER and POS model, Entity Dictionary, etc.

- `amr_data_dir` : `data/AMR` Actually this one doesn't need to change
- `numberbatch` : `data/numberbatch/numberbatch-19.08.en_it_es_de_zh.txt` need to be changed into `data/numberbatch/numberbatch-19.08.en_ms.txt`
- `bert.pretrained_model_dir` : `data/bert-base-multilingual-cased` is sufficient for basic experiment Additionally we'll use 
  - `data/mbart-base-cased`
  - `data/mt5-base-cased`
- `generator.mult_token_mapping` : `data/numberbatch/{}_en_neighbors_model.json` is sufficent, but we need to create one for `ms` language (was done using [this notebook](https://colab.research.google.com/drive/1SgSlDFD1uDFfFcaN08_KutDu4okkH2V0?usp=sharing))
- `mimick_test.data` : `- !!python/list [de, data/AMR/amr_2.0/test_de.txt.features.recat]` need to be changed into
  - `- !!python/list [ms, data/AMR/amr_2.0/test_ms.txt.features.recat]` and we need to create this using translator-annotated test data
- `mimick_test.prediction` : `test.mult.de.pred.txt` need to be `test.mult.ms.pred.txt`
- `mimick_test.word_splitter` : `data/bert-base-multilingual-cased/bert-base-multilingual-cased-vocab.txt` is enough, but need to be adjusted when using either `mbart` or `mt5`
- `data.train_data` and `data.dev_data` need to be adjusted to use `amr_2.0_ms/train_ms.txt.features.recat`
- `environment.serialization_dir` : `models/xl-amr_bilingual_en_de_trans_amr` need to be adjusted for `ms` language code.