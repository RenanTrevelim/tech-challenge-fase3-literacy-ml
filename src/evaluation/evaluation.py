import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


THRESHOLD = 0.52


def avaliar_modelo(
    y_real: pd.Series,
    prob_alfabetizado: pd.Series,
) -> dict:
    """
    Avalia as previsões utilizando o threshold
    operacional definido no projeto.
    """

    y_pred = (
        prob_alfabetizado
        >= THRESHOLD
    ).astype(int)

    metricas = {
        "accuracy":
            accuracy_score(
                y_real,
                y_pred,
            ),

        "precision_0":
            precision_score(
                y_real,
                y_pred,
                pos_label=0,
                zero_division=0,
            ),

        "recall_0":
            recall_score(
                y_real,
                y_pred,
                pos_label=0,
                zero_division=0,
            ),

        "f1_0":
            f1_score(
                y_real,
                y_pred,
                pos_label=0,
                zero_division=0,
            ),

        "f1_macro":
            f1_score(
                y_real,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "roc_auc":
            roc_auc_score(
                y_real,
                prob_alfabetizado,
            ),
    }

    return metricas