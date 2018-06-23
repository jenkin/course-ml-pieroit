import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

def main():
    """
    Esercizio 2
    """
    # carica dataset dataset
    dataset = load_breast_cancer()

    # converti a dataframe pandas e assegna nomi alle colonne
    dataset_df          = pd.DataFrame(dataset.data)
    dataset_df.columns  = dataset.feature_names

    # stampa informazioni
    print(dataset_df.head())
    print(dataset_df.describe())

    # grafici esplorativi
    plt.rcParams['axes.labelsize'] = 4
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["xtick.major.size"] = 0
    plt.rcParams["ytick.major.size"] = 0
    plt.rcParams["xtick.minor.size"] = 0
    plt.rcParams["ytick.minor.size"] = 0
    scatter_matrix(dataset_df[ dataset_df.columns ], figsize=(5, 5), s = 1)
    plt.savefig('breast_cancer_dataset.png', dpi = 1200)
    #plt.show()

main()
