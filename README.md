# partyembed

Scripts to fit and explore word embedding models augmented with political metadata.

## Description

This is a pre-release of the replication scripts for the study cited below.

To download the pre-trained models, use the following link: 
https://drive.google.com/open?id=17DWO_2UyjEZ9HnH3AACSdH3JZLpvb4bP

Make sure to extract the model files in the 'models' directory, i.e.:
```
tar xvzf partyembed_models.tar.gz -C partyembed/partyembed/models
```

The scripts are organized as a Python module, and functionalities will be added in this version.  Consult the file examples.ipynb for a tutorial.

The src/ directory contains example scripts to process the raw corpora and fit augmented embedding models on political texts.

## Citation

An earlier version of our manuscript was presented at the 2018 Conference of the Society for Political Methodology. The reference will be updated in the future. 

Ludovic Rheault and Christopher Cochrane.  2018.  Word Embeddings for the Estimation of Ideological Placement in Parliamentary Corpora. Presented at the 35th annual meeting of the Society for Political Methodology, Provo, Utah, July 18-21.

```
@inproceedings{RHE18,
  author    = {Ludovic Rheault and Christopher Cochrane},
  title     = {Word Embeddings for the Estimation of Ideological Placement in Parliamentary Corpora},
  booktitle = {PolMeth 2018},
  year      = {2018},
  address   = {Provo, UT},
  publisher = {Society for Political Methodology},
}
```
