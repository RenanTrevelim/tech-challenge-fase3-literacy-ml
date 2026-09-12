from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

CAMINHO_MODELO = (
    ROOT
    / "models"
    / "modelo_final_random_forest.pkl"
)


def carregar_modelo_final():
    """
    Carrega o pipeline final serializado com Joblib.
    """

    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {CAMINHO_MODELO}"
        )

    return joblib.load(
        CAMINHO_MODELO
    )


def gerar_probabilidades(
    modelo,
    dados: pd.DataFrame,
) -> pd.DataFrame:
    """
    Gera as probabilidades das duas classes
    utilizando o pipeline final.
    """

    probabilidades = (
        modelo.predict_proba(dados)
    )

    resultado = pd.DataFrame(
        {
            "prob_nao_alfabetizado":
                probabilidades[:, 0],

            "prob_alfabetizado":
                probabilidades[:, 1],
        },
        index=dados.index,
    )

    return resultado