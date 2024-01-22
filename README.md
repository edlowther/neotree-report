### Neotree report

#### 1. Getting started
To reproduce the analysis in this report, you can rebuild the `conda` environment using `conda env create --file environment.yml`. To use this environment with Jupyter, run:
```
conda activate nt
python -m ipykernel install --user --name nt
```
It has been tested on Linux. 

#### 2. Data source
Key assumptions about the data and full details on, for example, how we construct the composite outcome variable are codified in `./tests/test_datamanager.py`. If you have access to the same dataset used for this report, you should be able to update the filepath on line 6 of this file then run `python -m pytest` from inside the conda environment with no errors. If you are using an updated dataset, it would be worthwhile to check first whether these assumptions are still valid. 


