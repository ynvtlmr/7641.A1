import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')
DATA_DIR = '../data/'
DATA_PLOTS_DIR = '../plots/data/'


def visualize_all(filename):
    file_path = os.path.join(DATA_DIR, filename)  # base input file path
    df = pd.read_csv(file_path)  # data-frame from CSV

    # create a subdirectory for each dataset (if one does not exist)
    data_plots_dir = os.path.join(DATA_PLOTS_DIR, filename.split('.')[0])
    if not os.path.exists(data_plots_dir):
        os.makedirs(data_plots_dir)

    # get number of unique classes in class
    count_unique_classes = df['class'].nunique()

    # does the same as df.groupby('class').hist() but saves all resulting images.
    for i in range(count_unique_classes):
        df.loc[df['class'] == i].groupby('class').hist()
        plt.savefig(os.path.join(data_plots_dir, 'HIST_' + str(i)))
        plt.close()


def visualize_pairs(filename):
    file_path = os.path.join(DATA_DIR, filename)  # base input file path
    df = pd.read_csv(file_path)  # data-frame from CSV
    cols = df.columns  # column names
    count_unique_classes = df['class'].nunique()

    # create a subdirectory for each dataset (if one does not exist)
    data_plots_dir = os.path.join(DATA_PLOTS_DIR, filename.split('.')[0])
    if not os.path.exists(data_plots_dir):
        os.makedirs(data_plots_dir)

    # create a visualization for each attribute in dataset
    for i in range(len(cols[:-2])):
        for j in range(len(cols[i + 1:-1])):
            # plot title
            title = ' - '.join([filename.split('.')[0].capitalize(),
                                cols[i].capitalize(),
                                cols[i + j + 1].capitalize()])
            plt.title(title)
            plt.legend(range(count_unique_classes))

            fig, ax = plt.subplots()

            colors = {0: 'C0', 1: 'C1', 2: 'C2', 3: 'C3'}

            grouped = df.groupby('class')
            for key, group in grouped:
                group.plot(ax=ax, kind='scatter',
                           x=cols[i], y=cols[i + j + 1],
                           alpha=0.5, label=key, color=colors[key])

            # save the plot
            plt.savefig(os.path.join(data_plots_dir, "TWO_{}-{}".format(cols[i], cols[i + j + 1])))
            plt.close()


def visualize_each(filename):
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

        # save the plot
        plt.savefig(os.path.join(data_plots_dir, "ONE_{}".format(c)))
        plt.close()


def visualize_bars(filename):
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

        df.groupby([c, 'class'])[c].size().unstack().plot(kind='bar', stacked=False)

        # save the plot
        plt.savefig(os.path.join(data_plots_dir, "BAR2_{}".format(c)))
        plt.close()




# for each file in data data_plots_dir
for filename in os.listdir(DATA_DIR):
    # visualize_all(filename)
    # visualize_pairs(filename)
    # visualize_each(filename)
    if 'contraceptive' in filename:
        visualize_bars(filename)
