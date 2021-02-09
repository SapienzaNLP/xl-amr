# XL-AMR: Enabling Cross-Lingual AMR Parsing with Transfer Learning Techniques

XL-AMR ([Camera ready print](https://www.aclweb.org/anthology/2020.emnlp-main.195.pdf)) is a cross-lingual  AMR parser that exploits the existing training data in English to transfer semantic representations across languages. The achieved results shed light on the applicability of AMR  as an interlingua and set the state of the art in Chinese, German, Italian and Spanish cross-lingual AMR parsing. Furthermore, a detailed qualitative analysis shows that the proposed parser can overcome common translation divergences among languages.


If you find either our code or our release datasets useful in your work, please cite us with:
```
@inproceedings{blloshmi-etal-2020-enabling,
    title = "{XL-AMR}: {E}nabling Cross-Lingual {AMR} Parsing with Transfer Learning Techniques",
    author = "Blloshmi, Rexhina  and
      Tripodi, Rocco  and
      Navigli, Roberto",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.195",
    pages = "2487--2500",
}
```

#### If you need our silver data and/or XL-AMR checkpoints, please contact us by email! 

## 1. Install 

Create a conda environment with **Python 3.6** and **PyTorch 1.5.0** and install the dependencies [requirements.txt](requirements.txt).

Via conda:

    conda create -n xlamr python=3.6
    source activate xlamr
    pip install -r requirements.txt
    

## 2. Gold Dataset
1 - Download AMR 2.0 ([LDC2017T10](https://catalog.ldc.upenn.edu/LDC2017T10)) and AMR 2.0 - Four Translations ([LDC2020T07](https://catalog.ldc.upenn.edu/LDC2020T07)).

2 - Unzip the AMR 2.0 corpus to `data/AMR/LDC2017T10`. It should look like this:

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
    
Prepare training/dev/test data:

    ./scripts/prepare_data.sh -v 2 -p data/AMR/LDC2017T10
    
3 - Unzip the Translations corpus to `data/AMR/LDC2020T07` and copy ```*.txt``` files into ```data/amr_2.0/translations/```  .

Project English test AMR graphs across languages:

    ./scripts/project_test.sh 
   
    
## 3. Silver dataset for {Chinese, German, Italian, Spanish}
NOTE: The data and related artifacts can be downloaded from [to be uploaded soon](#).
#### 3.1 Annotation Projection through parallel sentences
These data make use of Europarl corpus [1]  and the English AMR parser by Zhang et al. 2019 [2] to parse the English side of the parallel corpus.

Data used for training and development are found in the following folder: 
 
    cd xl-amr/data/AMR/europarl_en_de_es_it/
    
<sub>*We do not produce silver AMR graphs for Chinese in this approach since Europarl does not cover Chinese language.</sub>

#### 3.2 Annotation Projection through automatic translations
We machine translated sentences of AMR 2.0 using [OPUS-MT](https://huggingface.co/transformers/model_doc/marian.html) pretrained models and filtered less accurate translations. The translated sentences are found in the following folder: 

    cd data/AMR/amr_2.0_zh_de_es_it/translations/

To respect the LDC agreement for AMR 2.0, we release the translations without the gold graphs. Therefore to project the AMR graphs from AMR 2.0 run:

    ./scripts/project_train_dev.sh
  

#### ****AFTER THIS STEP, DATA COLLECTION IS COMPLETE AND WE CAN CONTINUE WITH THE PARSING PROCEDURES****


## 4 Preprocessing

#### 4.1 Lemmatization, PoS-tagging, NER:

- For English we use [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/index.html) (version **3.9.2**). Before running the script, start a CoreNLP server following the [API documentation](https://stanfordnlp.github.io/CoreNLP/corenlp-server.html#api-documentation).

        cd stanford-corenlp-full-2018-10-05
        java -mx4g -cp "*" edu.stanford.nlp.pipeline StanfordCoreNLPServer -port 9000 -timeout 15000

- For Italian we use [Tint](http://tint.fbk.eu/) (package **tint-runner-0.2-bin.tar.gz**). Before running the script, start a Tint server 

       ./tools/tint/tint-server.sh -p 9200
    
- For German and Spanish we use [Stanza](https://stanfordnlp.github.io/stanza/). Before running the script, dowload the models.
    
        import stanza
        stanza.download("zh")
        stanza.download("de")
        stanza.download("es")
        

Run the script ```lang -> {en, zh, de, es, it}``` and ```dataset_type -> {silver, gold}```:

    ./scripts/annotate_features_multilingual.sh {data_dir} {lang} {dataset_type}

        
 
#### 4.2 Re-categorization and Anonymization

To preprocess the training and dev data run: 

    ./scripts/preprocess_multlingual.sh {data_dir} {dataset_type}

```dataset_type -> {silver, gold}```

To preprocess the test data run: 

    ./scripts/preprocess_test_multilingual.sh
    

## 5. Training
XL-AMR models are trained in one GeForce GTX TITAN X GPU.
We provide the params ```.yaml``` files for all the models in the paper. 


    python -u -m stog.commands.train params/{method}/{xl-amr_configuration}.yaml

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
This project is released under the CC-BY-NC 4.0 license (see `LICENSE`). If you use XL-AMR, please put a link to this repo.


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

