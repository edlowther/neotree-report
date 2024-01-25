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

#### 3. Replicating data summary details
For all datapoints appearing in the text of the report, please see the `data-exploration.ipynb` notebook. This also calls the code to generate the data summary table, which is stored separately in `./src/tablebuilder.py`.

#### 4. Main analysis
The models themselves are trained and validated using the notebook at `analysis.ipynb`, which depends on the helper files `./src/datamanager.py` to handle feature engineering and `./src/studymanager.py` to configure the hyperparameter optimisation process. 
