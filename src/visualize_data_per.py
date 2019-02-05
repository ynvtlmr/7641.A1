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
    count_unique_classes = df['class'].nunique()

    # create a subdirectory for each dataset (if one does not exist)
    data_plots_dir = os.path.join(DATA_PLOTS_DIR, filename.split('.')[0])
    if not os.path.exists(data_plots_dir):
        os.makedirs(data_plots_dir)

    # create a visualization for each attribute in dataset
    for c in cols:

        # plot title
        title = ' - '.join([filename.split('.')[0].capitalize(), c.capitalize()])
        plt.title(title)
        plt.legend(range(count_unique_classes))

        if c == 'class':
            df.groupby('class')[c].hist(bins=2)
        else:
            df.groupby('class')[c].plot(kind='kde')
            # df.groupby('class')[c].plot(kind='hist')
            # df.groupby('class')[c].plot(kind='hist', alpha=0.5)

        # save the plot
        plt.savefig(os.path.join(data_plots_dir, c))
        plt.close()
