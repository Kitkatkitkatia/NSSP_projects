# NSSP - Mini Project 1 (Autumn 2026)

We investigate task-related brain activity in one subject from the Human Connectome Project (HCP), using a language task adapted from a paradigm designed to map ATL language areas

## Folder structure
NSSP_projects\
|_data (not on repo) <= original data\
|_derivatives (not on repo) <= produced data file
  |_anat\ <= anatomical preprocessing
  |_func\ <= funcitonal preprocessing
  |_glm\ <= regressors
|_figures <= produced figures

## Notebook structure

### Part 1
We apply structural and functional preprocessing and map task-related activity using a general linear model (GLM)

Preprocessing in part1_Preprocessing.ipynb\
GLM in part1_GLM.ipynb\
helpers.py

### Part 2
We identify dominant patterns of fMRI variability using principal component analysis (PCA) and relate their temporal scores to the GLM regressors.

part2_PCA.ipynb\
helpers_2.py

## Authors 
Coraline Bès, Emma Cerruti-Bozzola, Maguette Diouf, Katia Häfliger, Luca Segovia 
