# partyembed

Scripts to fit and explore word embedding models augmented with political metadata.

## Description

This is a release of materials for the study cited below.  We will post developments on this page as time permits.

To download the pre-trained models, use this [link](https://drive.google.com/open?id=17DWO_2UyjEZ9HnH3AACSdH3JZLpvb4bP).  Additional information can be found on the [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/K0OYQF).

Make sure to extract the model files in the 'models' directory, i.e.:
```
tar xvzf partyembed_models.tar.gz -C partyembed/partyembed/models
```

The scripts are organized as a Python module, and functionalities will be added in this version.  Consult the file examples.ipynb for a tutorial.

The src/ directory contains example scripts to process the raw corpora and fit augmented embedding models on political texts.  The three scripts in that directory illustrate how to replicate the embeddings model for the US House.

## Citation

Please cite the following paper if using these materials:  

Ludovic Rheault and Christopher Cochrane.  2020.  [Word Embeddings for the Analysis of Ideological Placement in Parliamentary Corpora.](https://ludovicrheault.weebly.com/uploads/3/9/4/0/39408253/rheaultcochrane2019_pa.pdf)  Political Analysis 28(1): 112-133.

```
@article{RHE20,
  author    = {Ludovic Rheault and Christopher Cochrane},
  title     = {Word Embeddings for the Analysis of Ideological Placement in Parliamentary Corpora},
  journal   = {Political Analysis},
  year      = {2020},
  volume    = {28},
  number    = {1},
  pages     = {112-133}
}
```
