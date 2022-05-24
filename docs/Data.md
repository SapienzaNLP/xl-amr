# Data Meaning and Format

Here we'll explore and try to find out the format of the data involved and possible usage for each so that it could be adapted for another language. Also create a sample data for rapid iteration, especially in feature annotation, preprocessing and postprocessing phases.

## AMR

### AMR 2.0

AMR 2.0 or LDC2019T10 is English AMR dataset that contains 16k Data. Originially it's split into several textfile based on it's source. But in our case we'll just split it into train/dev/test.

#### Raw

The raw form of AMR dataset will be like the following:
```
# ::id bolt12_64556_5627.1 ::date 2012-12-04T17:55:20 ::annotator SDL-AMR-09 ::preferred
# ::snt Resolutely support the thread starter! I compose a poem in reply:
# ::save-date Sun Dec 8, 2013 ::file bolt12_64556_5627_1.txt
(m / multi-sentence
      :snt1 (s / support-01 :mode imperative
            :ARG0 (y / you)
            :ARG1 (p / person
                  :ARG0-of (s2 / start-01
                        :ARG1 (t / thread)))
            :manner (r / resolute))
      :snt2 (r2 / reply-01
            :ARG0 (i / i)
            :ARG2 (c / compose-02
                  :ARG0 i
                  :ARG1 (p2 / poem))))
```
which contains, an `id`, `date` annotated, the `annotator`, the `sentence itself`, `save-date` and its `file` name, and lastly the AMR graf itself. But for the purpose of AMR research, we only use its `id`, `sentence` and AMR the graph.

### AMR 2.0 Translation

AMR 2.0 translation into multiple language or LDC2020T07 contains multiple `.txt` files of test split from AMR 2.0. This dataset will be put into `data/amr_2.0/translations/` and will be projected to original AMR 2.0 data with `scripts/project_test.sh`

#### Original Contents

Looking at `project_test.sh` we could see that the name of file is in the format of:`amr-release-2.0-amrs-test-{file}.sentences.{lang}.txt` where `{file}` is one of `file_list = ['bolt', 'consensus', 'dfa', 'proxy', 'xinhua']` and `{lang}` is one of `["ES", "DE", "IT", "ZH"]`. And somehow, it is important that the order of `file_list` must be retained.

> It turns out that the `file_list` order is important because translation data doesn't contains any ID, so to make sure the projection between AMR 2.0 test data and the translated data is correct, we should make sure the order of `file_list` to follow the one used int test set of AMR 2.0.

Also the original content of the translated dataset only contains sentences without ID. More or less the content will be like this:

```
...
Paham ini menjauhkan pemerintah dari masyarakat dengan meninggalkan tanggungjawabnya sebagai pelayan dan pengatur urusan publik. 
Kemudian mengalihkan peran pemerintah kepada para kapitalis baik investor asing maupun investor lokal. 
Syariat menggariskan pemerintah memiliki peranan kuat dalam perekonomian sehingga tidak boleh berlepastangan terhadap hak-hak rakyatnya. 
...
```

#### Projected File

After running the `project_test.sh` which takes in AMR 2.0 test path (`amr_test`), the translated AMR directory (`trans_path`) and output directory for the result (`out_path`) we will get 4 files with the name of `test_{lang}.txt` where `{lang}` is lower cased language code `["es" ,"de", "it", "zh"]`. The output content will be like this:

```
# ::id 3
# ::snt Syariat menggariskan pemerintah memiliki peranan kuat dalam perekonomian sehingga tidak boleh berlepastangan terhadap hak-hak rakyatnya.
# ::tok-en Syariah dictates government has strong role in economy so may not release grip on public rights.
(vv1 / dictate
      :ARG0 (vv2 / person
            :ARG0-of (vv3 / have-org-role
                  :ARG2 (vv4 / authority)))
      :ARG1 (vv5 / have
            :ARG0 (vv6 / government-organization
                  :ARG0-of (vv7 / govern))
            :ARG1 (vv8 / role
                  :ARG1-of (vv9 / strong-02)
                  :poss (vv10 / economy))
            :purpose (vv11 / possible
                  :ARG1 (vv12 / release-01
                        :ARG0 vv6
                        :ARG1 (vv14 / grip
                              :ARG1 (vv15 / right-05
                                    :ARG1 (vv16 / public)))))))
```
It'll then be pased into feature annotation script.

#### Features 

#### Recat

#### Nosense
