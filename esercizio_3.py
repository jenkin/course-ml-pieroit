"""
Esercizio 3
"""

import matplotlib.pyplot as plt
from scikitplot.estimators import plot_learning_curve
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler


def main():
    """
    Esercizio 3
    """

    dataset = load_wine()
    #print(dataset['DESCR'])

    scaler = RobustScaler()
    x_data = scaler.fit_transform(dataset.data)
    y_target = dataset.target

    print(x_data)
    print(y_target)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_target, train_size = 0.33)

    print(x_train)
    print(x_test)
    print(y_train)
    print(y_test)

    model = KNeighborsClassifier(n_neighbors = 3)
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    acc_train = accuracy_score(y_train, pred_train)
    print("Train accuracy", acc_train)
    acc_test = accuracy_score(y_test, pred_test)
    print("Test accuracy", acc_test)

    plot_learning_curve(model, x_data, y_target)
    plt.show()

main()
