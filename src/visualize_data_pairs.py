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
    for i in range(len(cols[:-2])):
        for j in range(len(cols[i + 1:-1])):
            # plot title
            title = ' - '.join([filename.split('.')[0].capitalize(),
                                cols[i].capitalize(),
                                cols[i + j + 1].capitalize()])
            plt.title(title)
            plt.legend(range(count_unique_classes))

            # df.groupby('class')[cols[i]].plot(kind='kde')
            # plt.scatter(df[cols[i]], df[cols[i+j]], label='class')
            # df.groupby('class')[cols[i], cols[i+j]].plot(kind='scatter')
            # df.groupby('class').plot.scatter(x=cols[i], y=cols[i + j + 1])

            fig, ax = plt.subplots()

            colors = {0: 'C0', 1: 'C1', 2: 'C2', 3: 'C3'}

            grouped = df.groupby('class')
            for key, group in grouped:
                group.plot(ax=ax, kind='scatter',
                           x=cols[i], y=cols[i + j + 1],
                           alpha=0.5, label=key, color=colors[key])

            # plt.show()

            # save the plot
            plt.savefig(os.path.join(data_plots_dir, cols[i] + "-" + cols[i + j + 1]))
            # plt.show()
            plt.close()

            #
            # # Create data
            # N = 60
            # g1 = (0.6 + 0.6 * np.random.rand(N), np.random.rand(N))
            # g2 = (0.4 + 0.3 * np.random.rand(N), 0.5 * np.random.rand(N))
            # g3 = (0.3 * np.random.rand(N), 0.3 * np.random.rand(N))
            #
            # data = (g1, g2, g3)
            # colors = ("red", "green", "blue")
            # groups = ("coffee", "tea", "water")
            #
            # # Create plot
            # fig = plt.figure()
            # ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
            #
            # for data, color, group in zip(data, colors, groups):
            #     x, y = data
            #     ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
            #
            # plt.title('Matplot scatter plot')
            # plt.legend(loc=2)
            # plt.show()
