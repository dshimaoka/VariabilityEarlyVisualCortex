# VariabilityEarlyVisualCortex

This repository contains all source code necessary to replicate our recent work entitled "Variability of visual field maps in human early extrastriate cortex challenges the canonical model of organization of V2 and V3" available on [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.10.16.511648v2.abstract). 

## Table of Contents
* [Visualization and Reproducibility](#Visualization-and-Reproducibility)
* [Reproducibility of clusters](#Reproducibility-of-clusters)
* [Other resources](#other-resources)
* [Citation](#citation)
* [Contact](#contact)


## Visualization and Reproducibility
For visualization and further analysis, we recommend using NeuroDesk[]:

1. Clone this repository (or a forked version of this repository):

```bash
    git clone https://github.com/felenitaribeiro/VariabilityEarlyVisualCortex.git
```   

2. Download the available data at [Balsa](https://balsa.wustl.edu/study/9Zkk) following these steps: https://www.neurodesk.org/tutorials/functional_imaging/connectomeworkbench/ and extract to ./matlab

Note that you only need the "Retinotopy_HCP_7T_181_Fit1.scene", "Retinotopy_HCP_7T_181_Fit2.scene", "Retinotopy_HCP_7T_181_Fit3.scene", and "Retinotopy_HCP_7T_181_fsaverage.scene" files.

3. Set up your Matlab license following these steps: https://www.neurodesk.org/tutorials/programming/matlab/  

This step is required to convert the "dscalar.nii" files to ".mat"

4. Run our envPrep.sh script:
```bash
    bash ./envPrep.sh
```   
This bash script includes data preparation and conda environment creation with required packages.

5. Finally, you can launch our Jupyter notebook by double-clicking on maps-visualization.ipynb on Jupyter lab:


## Reproducibility of clusters

Given that the updated version of nilearn (in which we modified a few plotting functionalities) requires an updated version of scikit-learn, and unfortunately, the random seeds differ, to reproduce the same clusters we report in our manuscript, you will have to create a different conda environment to run the clustering analyses.

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