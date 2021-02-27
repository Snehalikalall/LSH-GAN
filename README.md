# LSH-GAN

Run the **LSH-GAN_parabolasimulation.py** from folder Parabola_simulation_plots_codes.  It will generate the parabola figures for different iteration.


Download the Raw demo dataset (yan.rds). Put the dataset and codes in same folder.

Run the **DataProcessing.R**  file to preprocess the raw datasets.  There are three user parameters: min_Reads, min_Cell, min_Gene. 

It will generate preprocessdata.csv. (The preprocessed Data)

Run the file **LSH-GAN.py**. It will generate the samples with different size (#0.25p, 0.5p, 0.75p, 1p, 1.25p, 1.5p) , p is the feature size. 


The user input for number of **iter**  in LSH-GAN for LSH step is given **1** as default for given dataset (yan.rds) . It can be changed by users depending upon sample size of datasets,  **iter** is **2** for klein dataset. 


The another user input is number of **epoch** for training the LSH-GAN. Default is **10000**.  The optimal sample size for datasets used in our work is given in main paper.


## LSH-GAN Validation

The Wasserstein metric computation code is given in **Validation_Wasserstein.py** file.


The Feature Selection (FS) and Adjusted Rand Index (ARI) computation code is given in **Validation_FS_ARI.R** file.

## Pre-requisites
> R version  4.0.2


> R packages: SingleCellExperiment, scDataset.


> Python 3.7


> Python packages: sklearn-0.19.2, multiprocessing, tensorflow



