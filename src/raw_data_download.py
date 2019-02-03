import os
import urllib.request

RAW_DATA_DIR = "../raw_data/"
URLS = [
    'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt',
    'http://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data',
    'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data',
    'http://ftp.ics.uci.edu/pub/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data',
]

# make the directory if it does not exist
if not os.path.exists(RAW_DATA_DIR):
    os.makedirs(RAW_DATA_DIR)

# download all files at the given URLs
for url in URLS:
    filename = url.rsplit('/', 1)[-1]
    urllib.request.urlretrieve(url, os.path.join(RAW_DATA_DIR, filename))
