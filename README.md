# XL-AMR-ID: Enabling Cross-Lingual AMR Parsing with Transfer Learning Techniques

XL-AMR-ID is a cross-lingual  AMR parser that exploits the existing training data in English to transfer semantic representations across languages. The achieved results shed light on the applicability of AMR  as an interlingua and set the state of the art in Indonesian cross-lingual AMR parsing. Furthermore, a detailed qualitative analysis shows that the proposed parser can overcome common translation divergences among languages.


If you find either our code or our release datasets useful in your work, please cite us with:
```

```

## 1. Install 

Create a conda environment with **Python 3.6** and **PyTorch 1.5.0**, install the dependencies [requirements.txt](requirements.txt) and download the artifacts**.

Via conda:
```bash
conda create -n xlamr python=3.6
source activate xlamr
pip install -r requirements.txt
bash scripts/download_artifacts.sh    
```

> As per 2022-04-25 the newest `overrides` package [breaks current repo](https://github.com/allenai/allennlp/issues/5217), so try to downgrade it to `3.1.0` before proceeding

**Also please unzip all the zipped files you find inside the data folder before continuing with the other steps.

### Preparing Numberbatch

Here we'll use numberbatch 19.08 and filter it for only english and indonesian language. In this case, since both Bahasa Indonesia and Bahasa Malaysia are [denoted with `ms` language keyword](https://github.com/commonsense/conceptnet5/wiki/Languages#identifying-languages), we'll extract it using the following script:
```bash
cd data
wget https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-19.08.txt.gz
# filter the content to only include malaysia and english language code
zgrep -E "/c/(ms|en)/" numberbatch-19.08.txt.gz > numberbatch-19.08.en_ms.txt

# Replace the pattern /c/{lang}/token into {lang}_token inplace
sed -i -E "s/\/c\/([^\/]+)\/([^\\s]+)/\1_\2/g" numberbatch-19.08.en_ms.txt
```

then copy it into `numberbatch/` folder

### Preparing silver data

For the silver data, you can email us for details on translated data, but you can also prepare your own transalted and parallel data.
Translated data can be processed by following this process: 
1. extracting sentences using the script [`scripts/extract_sentences.sh'](scripts/extract_sentences.sh). 
2. Translate the sentences using MTL model (marianMT for example)
3. Project the sentences to their AMR graph using [`scripts/project_train_dev_unfiltered.sh`](scripts/project_train_dev_unfiltered.sh). Also adjust the variable to reflect your translated data information. 

For the process to replicate the original thesis, we also need to filter sentences with low similarity  score.

# TODO: Create script to filter the resulting silver data with low similarity score (how low?)

### If you need our silver data based on AMR 2.0 translations please contact us by email! 

Download XL-AMR best checkpoints* per language here: [CHECKPOINTS](https://drive.google.com/drive/folders/1_tu6EJET20pi5IG3T-807hDpBjtkPs7W?usp=sharing)

<sub>*Please take care of the paths in the config file according to your file structure in case of exceptions, before sending an email or opening an issue.


## 2. Gold Dataset
1 - Download AMR 2.0 ([LDC2017T10](https://catalog.ldc.upenn.edu/LDC2017T10)) and AMR 2.0 - Four Translations ([LDC2020T07](https://catalog.ldc.upenn.edu/LDC2020T07)).

2 - Unzip the AMR 2.0 corpus to `data/AMR/LDC2017T10`. It should look like this:

```bash
(xlamr)$ tree data/AMR/LDC2017T10 -L 2 data/AMR/LDC2017T10
├── data
│   ├── alignments
│   ├── amrs
│   └── frames
├── docs
│   ├── AMR-alignment-format.txt
│   ├── amr-guidelines-v1.2.pdf
│   ├── file.tbl
│   ├── frameset.dtd
│   ├── PropBank-unification-notes.txt
│   └── README.txt
└── index.html
``` 
Prepare training/dev/test datas plit:
```bash
./scripts/prepare_data.sh -v 2 -p data/AMR/LDC2017T10
``` 
3 - Unzip the Translations corpus to `data/AMR/LDC2020T07` and copy ```*.txt``` files into ```data/amr_2.0/translations/```  .

### Expected directory tree
```bash
$ tree data/AMR -L 3
data/AMR
├── amr_2.0
│   ├── (train|dev|test)[.snt].txt
│   └── translations
│       └── amr-release-2.0-amrs-test-(bolt|consensus|dfa|proxy|xinhua).sentences.ms.txt
├── amr_2.0_ms
│   └── translations
│       └── (dev|train)_(ms).txt
├── en_es_it_de_zh_utils
│   ├── countries.json
│   ├── entity_type_cooccur_counter_en_zh.json
│   ├── frame_lemma_counter[.json]
│   ├── lemma_frame_counter[.json]
│   ├── name_op_cooccur_counter[_en_(de|es|it|zsh)].json
│   ├── name_type_cooccur_counter_en_zh.json
│   ├── senseless_node_counter[.json]
│   ├── text_anonymization_en-(de|es|it|zsh).json
│   ├── wiki_span_cooccur_counter[_en_(de|es|it|zsh)].json
│   └── wiki_span_cooccur_counter.json
├── europarl_en_de_es_it
│   └── (dev|train)_(de|en|it|es).txt[.features.recat]
└── panl_en_ms
    ├── (dev|train)_(en|id)[.pred].txt
    └── (dev|train)_(en|id).txt.(features[.recat]|preproc)
```

Project English test AMR graphs across languages:
```
./scripts/project_test.sh 
```
    
## 3. Silver dataset for Indonesian

#### 3.1 Annotation Projection through parallel sentences
For Indonesian language, we'll make use of [PANL-BPPT corpus](https://github.com/prasastoadi/parallel-corpora-en-id/) that's parsed using English AMR parser by Zhang et al. 2019 [2].

Data used for training and development are found in the following folder: 
```bash
    cd xl-amr/data/AMR/panl_en_ms/
```

#### 3.2 Annotation Projection through automatic translations
We machine translated sentences of AMR 2.0 using [OPUS-MT](https://huggingface.co/transformers/model_doc/marian.html) pretrained models and filtered less accurate translations. The translated sentences are found in the following folder: 
```
    cd data/AMR/amr_2.0_id/translations/
```
To respect the LDC agreement for AMR 2.0, we release the translations without the gold graphs. Therefore to project the AMR graphs from AMR 2.0 run:
```
    ./scripts/project_train_dev.sh
```  

#### ****AFTER THIS STEP, DATA COLLECTION IS COMPLETE AND WE CAN CONTINUE WITH THE PARSING PROCEDURES****


## 4 Preprocessing

#### 4.1 Lemmatization, PoS-tagging, NER:

- For English we use [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/index.html) (version **3.9.2**). Before running the script, start a CoreNLP server following the [API documentation](https://stanfordnlp.github.io/CoreNLP/corenlp-server.html#api-documentation).
```bash
        cd stanford-corenlp-full-2018-10-05
        java -mx4g -cp "*" edu.stanford.nlp.pipeline StanfordCoreNLPServer -port 9000 -timeout 15000
```
- For Italian we use [Tint](http://tint.fbk.eu/) (package **tint-runner-0.2-bin.tar.gz**). Before running the script, start a Tint server 
```bash
       ./tools/tint/tint-server.sh -p 9200
```    
- For German and Spanish we use [Stanza](https://stanfordnlp.github.io/stanza/). Before running the script, dowload the models.
```python    
        import stanza
        stanza.download("zh")
        stanza.download("de")
        stanza.download("es")
```        

Run the script ```lang -> {en, zh, de, es, it}``` and ```dataset_type -> {silver, gold}```:
```
    ./scripts/annotate_features_multilingual.sh {data_dir} {lang} {dataset_type}
```
        
 
#### 4.2 Re-categorization and Anonymization

> TODO: Create expected dictionary tree

To preprocess the training and dev data run: 

    ./scripts/preprocess_multlingual.sh {data_dir} {dataset_type}

```dataset_type -> {silver, gold}```

To preprocess the test data run: 

    ./scripts/preprocess_test_multilingual.sh
    

## 5. Training
XL-AMR models are trained in one GeForce GTX TITAN X GPU.
We provide the params ```.yaml``` files for all the models in the paper. 


    python -u -m xlamr_stog.commands.train params/{method}/{xl-amr_configuration}.yaml

```method -> {zeroshot, xl-amr_par, xl-amr_trns}```
## 6. Prediction

To evaluate the XL-AMR models run: 

    ./scripts/predict_multilingual.sh {lang} {model_dir}
    
## 7. Postprocessing

For postprocessing two steps are needed: 

1 - Run wikification using **DBPedia Spotlight API** on the test sentences for each language. Before running the script, start a docker server (following the instructions here [spotlight-docker](https://github.com/dbpedia-spotlight/spotlight-docker)) for the the specific language model at ```{port}``` for example (the italian case):

    docker run -itd --restart unless-stopped -p 2230:80 dbpedia/spotlight-italian spotlight.sh
   
 
 then run:
    
    ./scripts/spotlight_dump.sh {lang} {port}
    
 ```lang -> {en, de, es, it}```
 
 <sub>*For Chinese we use [Babelfy](http://babelfy.org/) as shown in [Babelfy HTTP API example](http://babelfy.org/guide).</sub>
 
       
2 - Run postprocessing script:

    ./postprocess_multilingual.sh {lang} {model_dir}
    
```lang -> {en, zh, de, es, it}```

## 8. Evaluation using [Smatch](https://github.com/snowblink14/smatch) and [Fine-Grained](https://github.com/mdtux89/amr-evaluation) metrics

    ./compute_smatch.sh {lang} {model_dir}


## References
[1] Philipp Koehn. 2005. [Europarl: A Parallel Corpus for Statistical Machine  Translation](http://homepages.inf.ed.ac.uk/pkoehn/publications/europarl-mtsummit05.pdf). In Conference Proceedings: the tenth Machine Translation Summit, pages 79–86, Phuket, Thailand. AAMT, AAMT. 

[2] Sheng Zhang, Xutai Ma, Kevin Duh, and Benjamin Van Durme. 2019. [AMR Parsing as Sequence-to-Graph Transduction](https://www.aclweb.org/anthology/P19-1009/). In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 80–94, Florence, Italy. Association for Computational Linguistics.

## License
This project is released under the CC-BY-NC-SA 4.0 license (see `LICENSE`). If you use XL-AMR, please put a link to this repo.


## Acknowledgements

The authors gratefully acknowledgethe support of the <a href="http://mousse-project.org">ERC ConsolidatorGrant MOUSSE</a> No. 726487 and the <a href="https://elex.is/">ELEXIS</a> project No. 731015 under the European Union’s Horizon 2020 research and innovation programme. 

This work was supported in part by the MIUR under the grant "Dipartimenti di eccellenza 2018-2022" of the Department of Computer Science of the Sapienza University of Rome.


We adopted modules or code snippets from the open-source projects:
- [stog](https://github.com/sheng-z/stog)
- [AllenNLP](https://github.com/allenai/allennlp)
- [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
- [NeuroNLP2](https://github.com/XuezheMax/NeuroNLP2)
- [huggingface](https://huggingface.co/transformers/)

Thank you for making research easier!

