import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


def plotar_matriz_confusao(
    y_real: pd.Series,
    y_pred: pd.Series,
):
    """
    Gera a matriz de confusão.
    """

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ConfusionMatrixDisplay.from_predictions(
        y_real,
        y_pred,
        cmap="Blues",
        ax=ax,
    )

    ax.set_title(
        "Matriz de Confusão"
    )

    plt.tight_layout()

    return fig


def plotar_curva_roc(
    y_real: pd.Series,
    prob_alfabetizado: pd.Series,
):
    """
    Gera a curva ROC utilizando probabilidades.
    """

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    RocCurveDisplay.from_predictions(
        y_real,
        prob_alfabetizado,
        ax=ax,
    )

    ax.set_title(
        "Curva ROC"
    )

    ax.grid(
        alpha=0.2
    )

    plt.tight_layout()

    return fig