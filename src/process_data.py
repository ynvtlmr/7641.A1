import os
import pandas as pd

RAW_DATA_DIR = "./raw_data/"
DATA_DIR = "./data/"

## Banknote Authentication ##

"""
1. variance of Wavelet Transformed image (continuous) 
2. skewness of Wavelet Transformed image (continuous) 
3. curtosis of Wavelet Transformed image (continuous) 
4. entropy of image (continuous) 
5. class (integer) 
"""

labels = ['varience', 'skewness', 'curtosis', 'entropy', 'class']

df = pd.read_csv(
    os.path.join(RAW_DATA_DIR, 'data_banknote_authentication.txt'),
    header=None,
    names=labels)

df.to_csv(path_or_buf=os.path.join(DATA_DIR, "banknotes.csv"), index=False)

## Contraceptives ##

"""
   1. Wife's age                     (numerical)
   2. Wife's education               (categorical)      1=low, 2, 3, 4=high
   3. Husband's education            (categorical)      1=low, 2, 3, 4=high
   4. Number of children ever born   (numerical)
   5. Wife's religion                (binary)           0=Non-Islam, 1=Islam
   6. Wife's now working?            (binary)           0=Yes, 1=No
   7. Husband's occupation           (categorical)      1, 2, 3, 4
   8. Standard-of-living index       (categorical)      1=low, 2, 3, 4=high
   9. Media exposure                 (binary)           0=Good, 1=Not good
   10. Contraceptive method used     (class attribute)  1=No-use 
                                                        2=Long-term
                                                        3=Short-term
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

# ## Abalone Rings ##
# """
#     Name		    Data Type	Meas.	Description
#     ----		    ---------	-----	-----------
#     Sex		        nominal		        M, F, and I (infant)
#     Length		    continuous	mm	    Longest shell measurement
#     Diameter	    continuous	mm	    perpendicular to length
#     Height		    continuous	mm	    with meat in shell
#     Whole weight	continuous	grams	whole abalone
#     Shucked weight	continuous	grams	weight of meat
#     Viscera weight	continuous	grams	gut weight (after bleeding)
#     Shell weight	continuous	grams	after being dried
#     Rings		    integer			    +1.5 gives the age in years
# """
#
# labels = ['sex', 'length', 'diameter', 'height', 'whole', 'shucked', 'viscera', 'shell', 'class']
#
# df = pd.read_csv(
#     os.DATA_DIR.join(RAW_DATA_DIR, 'abalone.data'),
#     header=None,
#     names=labels)
# df = pd.get_dummies(df, drop_first=True)
# cols = list(df.columns.values)
# cols.pop(cols.index('class'))
# cols += ['class']
# df = df[cols]
#
# df.to_csv(path_or_buf=os.DATA_DIR.join(DATA_DIR, "abalone.csv"), index=False)

## Pima Indians Diabetes Database ##
"""
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
