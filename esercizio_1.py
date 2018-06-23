# importiamo le funzionalita'
import matplotlib.pyplot as plt
#import numpy as np
#from scikitplot.estimators import plot_learning_curve
from scikitplot.metrics import plot_confusion_matrix
from scikitplot.estimators import plot_learning_curve
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def main():
    """
    Esercizio 1
    """
    # carica dataset
    dataset = load_iris()
    x_data = dataset.data
    y_target = dataset.target

    # Esercizio
    # 1 - separa i dati in train e test
    # 2 - addestra il modello
    # 3 - raccogli le predizioni sul test set
    # 4 - misura accuratezza

    ## 1 - dividiamo i dati in training e test set
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_target)

    ## 2 - addestra il modello
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)

    ## 2b - learning curve
    plot_learning_curve(model, x_data, y_target)

    ## 3 - raccoglie le predizioni sul test set
    y_pred = model.predict(x_test)

    ## 4 - misura accuratezza
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy', accuracy)

    ## 4b - facciamo un grafico della matrice di confusione
    plot_confusion_matrix(y_test, y_pred)
    plt.show()

main()
