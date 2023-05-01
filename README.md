# VariabilityEarlyVisualCortex

This repository contains all source code necessary to replicate our recent work entitled "Variability of visual field maps in human early extrastriate cortex challenges the canonical model of organization of V2 and V3" available on [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.10.16.511648v2.abstract). 

## Table of Contents
* [Installation and requirements](#installation-and-requirements)
* [Reproducibility](#reproducibility)
* [Other resources](#other-resources)
* [Citation](#citation)
* [Contact](#contact)


## Installation and requirements 

- Create a conda environment;
```bash
    conda create -n VariabilityEarlyVisualCortex python=3.8
    conda activate VariabilityEarlyVisualCortex
```
- Install the required packages that are available at requirements.txt: 

```bash
    pip install -r requirements.txt
```
- Clone this repository (or alternatevely you can clone a forked version of this repository):

```bash
    git clone https://github.com/felenitaribeiro/VariabilityEarlyVisualCortex.git
```   

## Reproducibility

Given that the updated version of nilearn (in which I modified a few plotting functionalities) requires an updated version of scikit-learn, and unfortunately, the random seeds differ, if you want to reproduce the same clusterings we report in our manuscript, you will have to create a different conda environment to run the clustering analyses.

 Create a conda environment;
```bash
    conda create -n VariabilityEarlyVisualCortex_clustering python=3.6
    conda activate VariabilityEarlyVisualCortex_clustering
```
- Install the required packages that are available at requirements_clustering.txt: 

```bash
    pip install -r requirements_clustering.txt
```
- Then, you may run: 

```bash
    cd ./main/
    python clustering_JaccardSimilarity_PAmaps.py
```

## Other resources

#TODO

## Citation

Please cite our manuscript if it was somewhat helpful for you :wink:

    @article {Ribeiro2022,
        author = {Fernanda L. Ribeiro and Ashley York and Elizabeth Zavitz and Steffen Bollmann and Marcello G. P. Rosa and Alexander M. Puckett},
        title = {Variability of visual field maps in human early extrastriate cortex challenges the canonical model of organization of V2 and V3},
        elocation-id = {2022.10.16.511648},
        year = {2023},
        doi = {10.1101/2022.10.16.511648},
        publisher = {Cold Spring Harbor Laboratory},
        URL = {https://www.biorxiv.org/content/early/2023/01/25/2022.10.16.511648},
        eprint = {https://www.biorxiv.org/content/early/2023/01/25/2022.10.16.511648.full.pdf},
        journal = {bioRxiv}
    }


## Contact
Fernanda Ribeiro <[fernanda.ribeiro@uq.edu.au](fernanda.ribeiro@uq.edu.au)>