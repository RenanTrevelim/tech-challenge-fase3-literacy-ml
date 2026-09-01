from pathlib import Path

import joblib
import pandas as pd


# ==================================================
# CAMINHOS
# ==================================================
ROOT = Path(__file__).resolve().parents[1]

CAMINHO_MODELO = (
    ROOT
    / "models"
    / "modelo_final_random_forest.pkl"
)


# ==================================================
# CONFIGURAÇÕES DO MODELO
# ==================================================
THRESHOLD = 0.52

FEATURES_MODELO = [
    "ano",
    "rede",
    "sigla_uf",
    "regiao",
    "pib_per_capita",
    "populacao_2022",
    "area_km2",
    "densidade_demografica",
    "qtd_escolas_censo",
    "prop_escolas_rurais",
    "prop_escolas_agua_potavel",
    "prop_escolas_esgoto_rede",
    "prop_escolas_biblioteca_leitura",
    "prop_escolas_lab_informatica",
    "prop_escolas_internet",
    "prop_escolas_internet_aprendizagem",
    "prop_escolas_banda_larga",
    "prop_escolas_rampas_acessibilidade",
    "media_alunos_turma_fund_ai",
    "media_alunos_docente_fund_ai",
    "media_prop_salas_climatizadas",
    "renda_domiciliar_per_capita_mediana",
]


# ==================================================
# CARREGAMENTO DO MODELO
# ==================================================
def carregar_modelo():

    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {CAMINHO_MODELO}"
        )

    return joblib.load(
        CAMINHO_MODELO
    )


# ==================================================
# VALIDAÇÃO DAS FEATURES
# ==================================================
def validar_features(
    dados: pd.DataFrame,
) -> None:

    colunas_faltantes = [
        coluna
        for coluna in FEATURES_MODELO
        if coluna not in dados.columns
    ]

    if colunas_faltantes:
        raise ValueError(
            "Colunas necessárias não encontradas: "
            f"{colunas_faltantes}"
        )


# ==================================================
# GERAÇÃO DAS PREDIÇÕES
# ==================================================
def gerar_predicoes(
    modelo,
    dados: pd.DataFrame,
) -> pd.DataFrame:

    validar_features(
        dados
    )

    resultado = dados.copy()

    X = resultado[
        FEATURES_MODELO
    ].copy()

    probabilidades = (
        modelo.predict_proba(X)
    )

    resultado[
        "prob_nao_alfabetizado"
    ] = probabilidades[:, 0]

    resultado[
        "prob_alfabetizado"
    ] = probabilidades[:, 1]

    resultado[
        "score_risco"
    ] = (
        resultado[
            "prob_nao_alfabetizado"
        ]
        * 100
    ).round(2)

    resultado[
        "classe_prevista"
    ] = (
        resultado[
            "prob_alfabetizado"
        ]
        >= THRESHOLD
    ).astype(int)

    resultado[
        "situacao_prevista"
    ] = (
        resultado[
            "classe_prevista"
        ]
        .map(
            {
                0: "Não alfabetizado",
                1: "Alfabetizado",
            }
        )
    )

    resultado[
        "prioridade"
    ] = (
        resultado[
            "classe_prevista"
        ]
        .map(
            {
                0: "Prioritário",
                1: "Monitoramento",
            }
        )
    )

    return resultado