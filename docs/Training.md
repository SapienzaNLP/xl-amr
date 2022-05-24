# Training

## Embedding Data

There are two embedding model that's needed for CL-AMR, the first one is the multilingual embedding (numberbatch in our case), another is crosslingual neighbor model (mapping from target language to english token with high similarity)

## Vocabulary

`data/vocab/non_padded_namespaces.txt` is missing, it is used when fixed_vocab is True. And only contains 2 entry, so just create this file yourself.:
```
must_copy_tags
coref_tags

```
But, for other vocab files such ase encoder ids, need to be fixed to make sure the resulting parser can handle mix of source and target language.