import pandas as pd


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


def validar_colunas(
    dados: pd.DataFrame,
) -> None:
    """
    Valida se todas as features esperadas pelo
    pipeline final estão presentes no DataFrame.
    """

    colunas_faltantes = [
        coluna
        for coluna in FEATURES_MODELO
        if coluna not in dados.columns
    ]

    if colunas_faltantes:
        raise ValueError(
            "Colunas obrigatórias ausentes: "
            f"{colunas_faltantes}"
        )


def selecionar_features(
    dados: pd.DataFrame,
) -> pd.DataFrame:
    """
    Retorna somente as features utilizadas pelo
    modelo final.
    """

    validar_colunas(dados)

    return dados[
        FEATURES_MODELO
    ].copy()