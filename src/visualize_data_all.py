import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')

DATA_DIR = '../data/'
DATA_PLOTS_DIR = '../plots/data/'

# for each file in data data_plots_dir
for filename in os.listdir(DATA_DIR):

    file_path = os.path.join(DATA_DIR, filename)  # base input file path
    df = pd.read_csv(file_path)  # data-frame from CSV
    cols = df.columns  # column names

    # create a subdirectory for each dataset (if one does not exist)
    data_plots_dir = os.path.join(DATA_PLOTS_DIR, filename.split('.')[0])
    if not os.path.exists(data_plots_dir):
        os.makedirs(data_plots_dir)

    # get number of unique classes in class
    count_unique_classes = df['class'].nunique()

    # does the same as df.groupby('class').hist() but saves all resulting images.
    for i in range(count_unique_classes):
        df.loc[df['class'] == i].groupby('class').hist()
        plt.savefig(os.path.join(data_plots_dir, 'histogram_' + str(i)))
        plt.close()
