import os
import pandas as pd
from helpers import create_path_if_not_exists

RAW_DATA_DIR = "./raw_data/"
DATA_DIR = "./data/"

create_path_if_not_exists(RAW_DATA_DIR)
create_path_if_not_exists(DATA_DIR)


def process_banknotes():
    """
    ## Banknote Authentication ##

    Name		                 Data Type
    ----		                 -----------
    Variance of Wavelet image    (continuous)
    Skewness of Wavelet image    (continuous)
    Curtosis of Wavelet image    (continuous)
    Entropy of image             (continuous)
    Class                        (integer)
    """

    labels = ['varience', 'skewness', 'curtosis', 'entropy', 'class']

    df = pd.read_csv(
        os.path.join(RAW_DATA_DIR, 'data_banknote_authentication.txt'),
        header=None,
        names=labels)

    df.to_csv(path_or_buf=os.path.join(DATA_DIR, "banknotes.csv"), index=False)


def process_contraceptives():
    """
    ## Contraceptives ##

    Name		                  Data Type	    	 Description
    ----		                  -----------		 -----------
    Wife's age                    (numerical)
    Wife's education              (categorical)      1=low, 2, 3, 4=high
    Husband's education           (categorical)      1=low, 2, 3, 4=high
    Number of children ever born  (numerical)
    Wife's religion               (binary)           0=Non-Islam, 1=Islam
    Wife's now working?           (binary)           0=Yes, 1=No
    Husband's occupation          (categorical)      1, 2, 3, 4
    Standard-of-living index      (categorical)      1=low, 2, 3, 4=high
    Media exposure                (binary)           0=Good, 1=Not good
    Class (Contraceptive method)  (class attribute)  0=No-use
                                                     1=Long-term
                                                     2=Short-term
    """
    labels = ['w_age', 'w_education', 'h_education', 'children', 'w_religion',
              'w_employed', 'h_occupation', 'living', 'media', 'class']

    df = pd.read_csv(
        os.path.join(RAW_DATA_DIR, 'cmc.data'),
        header=None,
        names=labels
    )
    df['class'] -= 1

    df.to_csv(path_or_buf=os.path.join(DATA_DIR, "contraceptive.csv"), index=False)


def process_abalone():
    """
    ## Abalone Rings ##

    Name		    Data Type	    Meas.	Description
    ----		    -----------	    -----	-----------
    Sex		        (nominal)	  	        M, F, and I (infant)
    Length		    (continuous)  	mm	    Longest shell measurement
    Diameter	    (continuous)  	mm	    perpendicular to length
    Height		    (continuous)  	mm	    with meat in shell
    Whole weight	(continuous)  	grams	whole abalone
    Shucked weight	(continuous)  	grams	weight of meat
    Viscera weight	(continuous)  	grams	gut weight (after bleeding)
    Shell weight	(continuous)  	grams	after being dried
    Rings		    (integer)	  		    +1.5 gives the age in years
    """

    labels = ['sex', 'length', 'diameter', 'height', 'whole', 'shucked', 'viscera', 'shell', 'class']

    df = pd.read_csv(
        os.DATA_DIR.join(RAW_DATA_DIR, 'abalone.data'),
        header=None,
        names=labels)
    df = pd.get_dummies(df, drop_first=True)
    cols = list(df.columns.values)
    cols.pop(cols.index('class'))
    cols += ['class']
    df = df[cols]

    df.to_csv(path_or_buf=os.DATA_DIR.join(DATA_DIR, "abalone.csv"), index=False)


def process_diabetes():
    """
    ## Pima Indians Diabetes Database ##

    Name
    ----
    1. Number of times pregnant
    2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
    3. Diastolic blood pressure (mm Hg)
    4. Triceps skin fold thickness (mm)
    5. 2-Hour serum insulin (mu U/ml)
    6. Body mass index (weight in kg/(height in m)^2)
    7. Diabetes pedigree function
    8. Age (years)
    9. Class variable (0 or 1)
    """
    labels = ['num_pregnant', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi', 'pedigree', 'age', 'class']
    df = pd.read_csv(
        os.path.join(RAW_DATA_DIR, 'pima-indians-diabetes.data'),
        header=None,
        names=labels)

    df.to_csv(path_or_buf=os.path.join(DATA_DIR, "diabetes.csv"), index=False)


if __name__ == '__main__':
    """
    Run all the processing scripts
    """
    process_abalone()
    process_banknotes()
    process_contraceptives()
    process_diabetes()
