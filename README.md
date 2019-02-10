https://github.gatech.edu/ytalmor3/7641.A1
==========================================

# OMSCS 7641 - Machine Learning

This repository is for work submitted in Spring 2019 for the
Machine Learning (7641) class offered by GATech via Udacity.

## Getting Started

This project was built with `Python 3.7.0`.
The environment may be installed as such:
```
virtualenv .venv --python ~/local/path/to/3.7.0/bin/python
echo  'export PYTHONPATH="../:."' >> ./.venv/bin/activate
source ./.venv/bin/activate
python -m pip install --requirement requirements.txt
```

If running on a Mac, also run:
```
sed -i -e 's/: macosx/: TkAgg/g' ./.venv/lib/python3.7/site-packages/matplotlib/mpl-data/matplotlibrc
```

### Prerequisites

All pip prerequisites are listed in **requirements.txt**.
They may be installed with **pip** using:
```
pip install -r requirements.txt
```

### Environment
The environment may be activated with:
```
source ./.venv/bin/activate
```
It may be deactivated with: `deactivate`


## Abridged directory structure
```
├── src
│   ├── model_train.py
│   ├── model_evaluation.py
│   ├── visualize_data.py
│   └── clf_*.py
├── data
│   ├── banana.csv
│   └── contraceptive.csv
├── plots
│   └── *
│       └── *.png
├── README.txt
└── requirements.txt
```

## Instructions
Run `visualize_data.py` to visualize all datasets.
Run `model_train.py` followed by `model_evaluation.py` to run and visualize
all models using all datasets.
All generated visualizations are saved under the `plots` directory.


## Project 1
A survey project exploring a number of machine learning algorithms on a couple datasets.


## Authors
* **Yaniv Talmor**
