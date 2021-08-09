import seaborn as sns
from matplotlib import pyplot as plt

from config import DIRECTORY_FOR_LOG

def plot_and_save(filename,plot_title,xlabel,ylabel,data):
    sns.relplot(x=xlabel, y=ylabel, data=data).fig.suptitle(plot_title)
    plt.savefig(DIRECTORY_FOR_LOG + filename + ".png")
    plt.show()


def plot_dist_one_figure(test_data, neural_network_results):
    # Plot on one figure
    f, axes = plt.subplots(1, 2)
    sns.lineplot(x="błąd", y="% błędnych próbek", data=test_data, ax=axes[1], label='Excel')
    sns.lineplot(x="błąd", y="% błędnych próbek", data=neural_network_results, ax=axes[0], color='orange',
                 label='Sieć neuronowa')
    plt.savefig(DIRECTORY_FOR_LOG + "dystrybuanta_sep.png")
    plt.show()


def plot_dist_one_axis(test_data,neural_network_results):
    # Plot both lines on one figure
    ax = sns.lineplot(x="błąd", y="% błędnych próbek", data=neural_network_results, color='orange')
    sns.lineplot(x="błąd", y="% błędnych próbek", data=test_data, ax=ax)
    plt.legend(title='Dystrybuanta', loc='lower right', labels=['Sieć neuronowa', 'Excel'])
    plt.savefig(DIRECTORY_FOR_LOG + "dystrybuanta.png")
    plt.show()