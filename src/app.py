from textwrap import dedent

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)

from predict import (
    FEATURES_MODELO,
    THRESHOLD,
    carregar_modelo,
    gerar_predicoes,
)


# ==================================================
# CONFIGURAÇÃO DA PÁGINA
# ==================================================
st.set_page_config(
    page_title="Literacy Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# CONFIGURAÇÕES
# ==================================================
LIMIAR_RISCO_PRIORITARIO = (
    1 - THRESHOLD
) * 100


# ==================================================
# HTML
# ==================================================
def renderizar_html(
    conteudo: str,
) -> None:

    st.html(
        dedent(conteudo).strip()
    )


# ==================================================
# ESTILO VISUAL
# ==================================================
renderizar_html(
    """
    <style>

        .stApp,
        [data-testid="stAppViewContainer"] {
            background-color: #F5F7FB;
            color: #0F172A;
        }

        .block-container {
            max-width: 1400px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #172554 0%,
                #1E3A8A 100%
            );
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label {
            color: #FFFFFF !important;
        }

        /* Hero */
        .hero {
            padding: 2.3rem 2.5rem;
            border-radius: 24px;

            background: linear-gradient(
                135deg,
                #1E3A8A 0%,
                #2563EB 55%,
                #06B6D4 100%
            );

            margin-bottom: 1.8rem;

            box-shadow:
                0 18px 45px
                rgba(37, 99, 235, 0.20);
        }

        .hero h1 {
            margin: 0;

            color: #FFFFFF !important;

            font-size: 2.65rem;
            line-height: 1.15;
        }

        .hero p {
            max-width: 950px;

            margin-top: 1rem;
            margin-bottom: 0;

            color: #DBEAFE !important;

            font-size: 1.05rem;
            line-height: 1.7;
        }

        .badge {
            display: inline-block;

            padding: 0.38rem 0.85rem;
            margin-bottom: 1rem;

            border-radius: 999px;

            background-color:
                rgba(255, 255, 255, 0.18);

            color: #FFFFFF !important;

            font-size: 0.82rem;
            font-weight: 700;
        }

        /* Títulos */
        .section-title {
            margin-top: 1.8rem;
            margin-bottom: 1rem;

            color: #0F172A !important;

            font-size: 1.45rem;
            font-weight: 800;
        }

        /* Cards */
        .info-card {
            min-height: 180px;

            padding: 1.5rem;

            border: 1px solid #DCE4F0;
            border-radius: 18px;

            background-color: #FFFFFF;

            box-shadow:
                0 8px 25px
                rgba(15, 23, 42, 0.06);
        }

        .info-card h3 {
            margin-top: 0;
            margin-bottom: 0.8rem;

            color: #1D4ED8 !important;

            font-size: 1.3rem;
        }

        .info-card p {
            margin-bottom: 0;

            color: #475569 !important;

            line-height: 1.65;
        }

        /* Cards de análise */
        .analysis-card {
            min-height: 210px;

            padding: 1.5rem;

            border: 1px solid #DCE4F0;
            border-radius: 18px;

            background-color: #FFFFFF;

            box-shadow:
                0 8px 24px
                rgba(15, 23, 42, 0.06);
        }

        .analysis-card h3 {
            margin-top: 0;

            color: #0F172A !important;
        }

        .analysis-card p,
        .analysis-card li {
            color: #475569 !important;

            line-height: 1.65;
        }

        .card-blue {
            border-top: 6px solid #2563EB;
        }

        .card-green {
            border-top: 6px solid #059669;
        }

        .card-orange {
            border-top: 6px solid #F97316;
        }

        /* Notas */
        .model-note {
            padding: 1.1rem 1.3rem;

            border-left: 5px solid #2563EB;
            border-radius: 14px;

            background-color: #EFF6FF;

            color: #1E3A8A !important;

            line-height: 1.65;
        }

        .warning-note {
            padding: 1.1rem 1.3rem;

            border-left: 5px solid #F97316;
            border-radius: 14px;

            background-color: #FFF7ED;

            color: #9A3412 !important;

            line-height: 1.65;
        }

        /* Métricas */
        div[data-testid="stMetric"] {
            min-height: 112px;

            padding: 1rem 1.1rem;

            border: 1px solid #DCE4F0;
            border-radius: 16px;

            background-color: #FFFFFF !important;

            box-shadow:
                0 6px 18px
                rgba(15, 23, 42, 0.05);
        }

        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] * {
            color: #475569 !important;

            opacity: 1 !important;

            font-weight: 600 !important;
        }

        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] * {
            color: #0F172A !important;

            opacity: 1 !important;
        }

        /* Upload */
        [data-testid="stFileUploader"] {
            padding: 1rem;

            border: 1px solid #DCE4F0;
            border-radius: 18px;

            background-color: #FFFFFF;
        }

        /* Tabelas */
        [data-testid="stDataFrame"] {
            overflow: hidden;

            border: 1px solid #DCE4F0;
            border-radius: 14px;
        }

        /* Botão download */
        .stDownloadButton > button {
            border: none;
            border-radius: 10px;

            background: linear-gradient(
                90deg,
                #2563EB,
                #06B6D4
            );

            color: #FFFFFF;

            font-weight: 700;
        }

        /* Rodapé */
        .footer {
            margin-top: 3rem;
            padding-top: 1.5rem;

            border-top: 1px solid #DCE4F0;

            color: #64748B !important;

            font-size: 0.9rem;
            text-align: center;
        }

    </style>
    """
)


# ==================================================
# CACHE DO MODELO
# ==================================================
@st.cache_resource
def carregar_modelo_cache():

    return carregar_modelo()


# ==================================================
# LEITURA DO ARQUIVO
# ==================================================
def carregar_arquivo(
    arquivo,
) -> pd.DataFrame:

    nome = arquivo.name.lower()

    if nome.endswith(".csv"):

        return pd.read_csv(
            arquivo,
            sep=None,
            engine="python",
        )

    if nome.endswith(".parquet"):

        return pd.read_parquet(
            arquivo
        )

    raise ValueError(
        "Formato de arquivo não suportado."
    )


# ==================================================
# IDENTIFICAÇÃO DA COLUNA DE MUNICÍPIO
# ==================================================
def identificar_coluna_municipio(
    dados: pd.DataFrame,
):

    candidatos = [
        "id_municipio_nome",
        "municipio",
        "nome_municipio",
        "id_municipio",
    ]

    for coluna in candidatos:

        if coluna in dados.columns:

            return coluna

    return None


# ==================================================
# NORMALIZAÇÃO DO TARGET
# ==================================================
def normalizar_target(
    serie: pd.Series,
) -> pd.Series:

    if pd.api.types.is_numeric_dtype(
        serie
    ):

        return pd.to_numeric(
            serie,
            errors="coerce",
        )

    valores = (
        serie
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapa = {
        "0": 0,
        "1": 1,
        "não": 0,
        "nao": 0,
        "sim": 1,
        "não alfabetizado": 0,
        "nao alfabetizado": 0,
        "alfabetizado": 1,
    }

    return valores.map(
        mapa
    )


# ==================================================
# GRÁFICO DE PRIORIDADE
# ==================================================
def criar_grafico_prioridade(
    dados: pd.DataFrame,
):

    resumo = (
        dados["prioridade"]
        .value_counts()
        .rename_axis("prioridade")
        .reset_index(name="alunos")
    )

    fig, ax = plt.subplots(
        figsize=(9, 4.5)
    )

    sns.barplot(
        data=resumo,
        x="prioridade",
        y="alunos",
        hue="prioridade",
        legend=False,
        palette={
            "Prioritário": "#DC2626",
            "Monitoramento": "#2563EB",
        },
        ax=ax,
    )

    ax.set_title(
        "Distribuição das classificações",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Alunos")

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    ax.spines[
        ["top", "right"]
    ].set_visible(False)

    plt.tight_layout()

    return fig


# ==================================================
# GRÁFICO DISTRIBUIÇÃO DO SCORE
# ==================================================
def criar_grafico_score(
    dados: pd.DataFrame,
):

    fig, ax = plt.subplots(
        figsize=(9, 4.5)
    )

    sns.histplot(
        data=dados,
        x="score_risco",
        bins=25,
        color="#2563EB",
        edgecolor="white",
        ax=ax,
    )

    ax.axvline(
        LIMIAR_RISCO_PRIORITARIO,
        color="#DC2626",
        linestyle="--",
        linewidth=2,
        label=(
            "Limite prioritário "
            f"({LIMIAR_RISCO_PRIORITARIO:.0f}%)"
        ),
    )

    ax.set_title(
        "Distribuição do score de risco",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel(
        "Score de risco (%)"
    )

    ax.set_ylabel(
        "Alunos"
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    ax.spines[
        ["top", "right"]
    ].set_visible(False)

    plt.tight_layout()

    return fig


# ==================================================
# GRÁFICO REGIONAL
# ==================================================
def criar_grafico_regional(
    dados_regiao: pd.DataFrame,
):

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.barplot(
        data=dados_regiao,
        x="score_medio",
        y="regiao",
        color="#2563EB",
        ax=ax,
    )

    ax.set_title(
        "Score médio de risco por região",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel(
        "Score médio (%)"
    )

    ax.set_ylabel("")

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    ax.spines[
        ["top", "right"]
    ].set_visible(False)

    plt.tight_layout()

    return fig


# ==================================================
# GRÁFICO MUNICIPAL
# ==================================================
def criar_grafico_municipal(
    dados_municipio: pd.DataFrame,
):

    top_municipios = (
        dados_municipio
        .head(15)
        .copy()
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.barplot(
        data=top_municipios,
        x="score_medio",
        y="municipio_exibicao",
        color="#2563EB",
        ax=ax,
    )

    ax.set_title(
        "Municípios com maior score médio de risco",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel(
        "Score médio (%)"
    )

    ax.set_ylabel("")

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    ax.spines[
        ["top", "right"]
    ].set_visible(False)

    plt.tight_layout()

    return fig


# ==================================================
# CARREGAMENTO DO MODELO
# ==================================================
try:

    modelo = carregar_modelo_cache()

except Exception as erro:

    st.error(
        f"Erro ao carregar o modelo: {erro}"
    )

    st.stop()


# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:

    st.markdown(
        "## 📚 Literacy Intelligence"
    )

    st.caption(
        "Predição e Inteligência Analítica "
        "para Alfabetização no Brasil"
    )

    st.divider()

    pagina = st.radio(
        "Navegação",
        [
            "Analisar dados",
            "Sobre o projeto",
        ],
    )

    st.divider()

    st.markdown(
        """
### Modelo final

**Random Forest**

**Classe prioritária**

`0 — Não alfabetizado`

**Threshold operacional**

`0.52`

**Recall classe 0**

`≈ 78%`
        """
    )

    st.divider()

    st.caption(
        "Projeto desenvolvido por "
        "Renan Assis Trevelim."
    )


# ==================================================
# ANALISAR DADOS
# ==================================================
if pagina == "Analisar dados":

    renderizar_html(
        """
        <div class="hero">

            <span class="badge">
                Machine Learning • Educação
            </span>

            <h1>
                Análise de risco de alfabetização
            </h1>

            <p>
                Envie uma base de dados para executar
                automaticamente o pipeline de Machine
                Learning, gerar probabilidades,
                classificar os registros e explorar
                os resultados por região e município.
            </p>

        </div>
        """
    )

    # ==============================================
    # UPLOAD
    # ==============================================
    renderizar_html(
        """
        <div class="section-title">
            Carregar base para análise
        </div>
        """
    )

    arquivo = st.file_uploader(
        "Selecione um arquivo",
        type=[
            "csv",
            "parquet",
        ],
        help=(
            "O arquivo deve conter as variáveis "
            "utilizadas durante o treinamento."
        ),
    )

    with st.expander(
        "Ver variáveis obrigatórias do modelo"
    ):

        st.code(
            "\n".join(
                FEATURES_MODELO
            ),
            language="text",
        )

    # ==============================================
    # SEM ARQUIVO
    # ==============================================
    if arquivo is None:

        renderizar_html(
            """
            <div class="model-note">

                Envie um arquivo <strong>CSV</strong>
                ou <strong>Parquet</strong> para iniciar
                a análise.

                O pipeline executará automaticamente
                o pré-processamento, a seleção de
                variáveis e a inferência com o
                Random Forest.

            </div>
            """
        )

    # ==============================================
    # COM ARQUIVO
    # ==============================================
    else:

        # ------------------------------------------
        # LEITURA
        # ------------------------------------------
        try:

            dados_entrada = (
                carregar_arquivo(
                    arquivo
                )
            )

        except Exception as erro:

            st.error(
                "Não foi possível ler o arquivo."
            )

            st.exception(
                erro
            )

            st.stop()

        # ------------------------------------------
        # PREDIÇÕES
        # ------------------------------------------
        try:

            dados = gerar_predicoes(
                modelo=modelo,
                dados=dados_entrada,
            )

        except Exception as erro:

            st.error(
                "Não foi possível executar "
                "o pipeline de predição."
            )

            st.exception(
                erro
            )

            st.stop()

        # ==========================================
        # IDENTIFICAÇÃO DE ALTO RISCO
        # ==========================================
        scores_prioritarios = (
            dados.loc[
                dados[
                    "prioridade"
                ].eq(
                    "Prioritário"
                ),
                "score_risco",
            ]
        )

        if not scores_prioritarios.empty:

            limite_alto_risco = (
                scores_prioritarios
                .quantile(0.75)
            )

            dados[
                "alto_risco"
            ] = (
                dados[
                    "score_risco"
                ]
                >= limite_alto_risco
            )

        else:

            limite_alto_risco = None

            dados[
                "alto_risco"
            ] = False

        # ==========================================
        # IDENTIFICA MUNICÍPIO
        # ==========================================
        coluna_municipio = (
            identificar_coluna_municipio(
                dados
            )
        )

        # ==========================================
        # FILTROS
        # ==========================================
        renderizar_html(
            """
            <div class="section-title">
                Filtros da análise
            </div>
            """
        )

        dados_filtrados = (
            dados.copy()
        )

        regiao_selecionada = "Todas"
        municipio_selecionado = "Todos"

        col1, col2 = (
            st.columns(2)
        )

        # ------------------------------------------
        # REGIÃO
        # ------------------------------------------
        if "regiao" in dados.columns:

            regioes = sorted(
                dados[
                    "regiao"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            regiao_selecionada = (
                col1.selectbox(
                    "Região",
                    ["Todas"] + regioes,
                )
            )

            if (
                regiao_selecionada
                != "Todas"
            ):

                dados_filtrados = (
                    dados_filtrados[
                        dados_filtrados[
                            "regiao"
                        ].astype(str)
                        == regiao_selecionada
                    ]
                    .copy()
                )

        else:

            col1.info(
                "Região não disponível."
            )

        # ------------------------------------------
        # MUNICÍPIO
        # ------------------------------------------
        if coluna_municipio is not None:

            dados_filtrados = (
                dados_filtrados.copy()
            )

            if (
                "sigla_uf"
                in dados_filtrados.columns
            ):

                dados_filtrados[
                    "municipio_filtro"
                ] = (
                    dados_filtrados[
                        coluna_municipio
                    ]
                    .astype(str)
                    .str.strip()
                    + " - "
                    + dados_filtrados[
                        "sigla_uf"
                    ]
                    .astype(str)
                    .str.strip()
                )

            else:

                dados_filtrados[
                    "municipio_filtro"
                ] = (
                    dados_filtrados[
                        coluna_municipio
                    ]
                    .astype(str)
                    .str.strip()
                )

            municipios = sorted(
                dados_filtrados[
                    "municipio_filtro"
                ]
                .dropna()
                .unique()
                .tolist()
            )

            municipio_selecionado = (
                col2.selectbox(
                    "Município",
                    ["Todos"] + municipios,
                )
            )

            if (
                municipio_selecionado
                != "Todos"
            ):

                dados_filtrados = (
                    dados_filtrados[
                        dados_filtrados[
                            "municipio_filtro"
                        ]
                        == municipio_selecionado
                    ]
                    .copy()
                )

        else:

            col2.info(
                "Município não disponível."
            )

        # ==========================================
        # RESUMO DO FILTRO
        # ==========================================
        renderizar_html(
            """
            <div class="section-title">
                Resumo da análise
            </div>
            """
        )

        total = len(
            dados_filtrados
        )

        prioritarios = (
            dados_filtrados[
                "prioridade"
            ]
            .eq(
                "Prioritário"
            )
            .sum()
        )

        score_medio = (
            dados_filtrados[
                "score_risco"
            ]
            .mean()
        )

        pct_prioritarios = (
            prioritarios
            / total
            * 100
            if total > 0
            else 0
        )

        alto_risco_total = (
            dados_filtrados[
                "alto_risco"
            ]
            .sum()
        )

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        col1.metric(
            "Alunos analisados",
            f"{total:,}"
            .replace(",", "."),
        )

        col2.metric(
            "Score médio de risco",
            f"{score_medio:.2f}%"
            if total > 0
            else "N/A",
        )

        col3.metric(
            "Prioritários",
            f"{prioritarios:,}"
            .replace(",", "."),
        )

        col4.metric(
            "Alto risco",
            f"{alto_risco_total:,}"
            .replace(",", "."),
        )

        if total > 0:

            st.caption(
                f"{pct_prioritarios:.2f}% "
                "dos registros filtrados "
                "foram classificados como prioritários."
            )

        # ==========================================
        # GRÁFICOS GERAIS
        # ==========================================
        if total > 0:

            renderizar_html(
                """
                <div class="section-title">
                    Distribuição das previsões
                </div>
                """
            )

            col1, col2 = (
                st.columns(2)
            )

            with col1:

                st.pyplot(
                    criar_grafico_prioridade(
                        dados_filtrados
                    ),
                    use_container_width=True,
                )

            with col2:

                st.pyplot(
                    criar_grafico_score(
                        dados_filtrados
                    ),
                    use_container_width=True,
                )

        # ==========================================
        # CRITÉRIO DE ALTO RISCO
        # ==========================================
        if limite_alto_risco is not None:

            renderizar_html(
                f"""
                <div class="model-note">

                    <strong>Critério de alto risco:</strong>

                    o percentil 75 dos scores dos
                    registros prioritários da base
                    enviada corresponde a
                    <strong>{limite_alto_risco:.2f}%</strong>.

                    Esse limite é utilizado como uma
                    camada adicional de priorização e
                    não altera o threshold de
                    classificação do modelo.

                </div>
                """
            )

        # ==========================================
        # COMPARAÇÃO COM TARGET REAL
        # ==========================================
        if (
            "alfabetizado"
            in dados_filtrados.columns
            and total > 0
        ):

            y_real = (
                normalizar_target(
                    dados_filtrados[
                        "alfabetizado"
                    ]
                )
            )

            mascara_valida = (
                y_real.notna()
            )

            if mascara_valida.any():

                y_real_valido = (
                    y_real[
                        mascara_valida
                    ]
                    .astype(int)
                )

                y_pred = (
                    dados_filtrados.loc[
                        mascara_valida,
                        "classe_prevista",
                    ]
                )

                y_proba = (
                    dados_filtrados.loc[
                        mascara_valida,
                        "prob_alfabetizado",
                    ]
                )

                accuracy = (
                    accuracy_score(
                        y_real_valido,
                        y_pred,
                    )
                )

                recall_0 = (
                    recall_score(
                        y_real_valido,
                        y_pred,
                        pos_label=0,
                        zero_division=0,
                    )
                )

                f1_macro = (
                    f1_score(
                        y_real_valido,
                        y_pred,
                        average="macro",
                        zero_division=0,
                    )
                )

                try:

                    auc = (
                        roc_auc_score(
                            y_real_valido,
                            y_proba,
                        )
                    )

                except ValueError:

                    auc = None

                renderizar_html(
                    """
                    <div class="section-title">
                        Desempenho nos registros filtrados
                    </div>
                    """
                )

                col1, col2, col3, col4 = (
                    st.columns(4)
                )

                col1.metric(
                    "Accuracy",
                    f"{accuracy:.2%}",
                )

                col2.metric(
                    "Recall classe 0",
                    f"{recall_0:.2%}",
                )

                col3.metric(
                    "F1 Macro",
                    f"{f1_macro:.3f}",
                )

                col4.metric(
                    "ROC-AUC",
                    (
                        f"{auc:.3f}"
                        if auc is not None
                        else "N/A"
                    ),
                )

        # ==========================================
        # ANÁLISE REGIONAL
        # ==========================================
        if (
            "regiao"
            in dados_filtrados.columns
            and total > 0
        ):

            renderizar_html(
                """
                <div class="section-title">
                    Análise por região
                </div>
                """
            )

            risco_regiao = (
                dados_filtrados
                .groupby(
                    "regiao",
                    as_index=False,
                )
                .agg(
                    alunos=(
                        "score_risco",
                        "size",
                    ),
                    score_medio=(
                        "score_risco",
                        "mean",
                    ),
                    prioritarios=(
                        "prioridade",
                        lambda x: (
                            x
                            == "Prioritário"
                        ).sum(),
                    ),
                    alto_risco=(
                        "alto_risco",
                        "sum",
                    ),
                )
            )

            risco_regiao[
                "pct_prioritarios"
            ] = (
                risco_regiao[
                    "prioritarios"
                ]
                / risco_regiao[
                    "alunos"
                ]
                * 100
            )

            risco_regiao[
                "pct_alto_risco"
            ] = (
                risco_regiao[
                    "alto_risco"
                ]
                / risco_regiao[
                    "alunos"
                ]
                * 100
            )

            risco_regiao = (
                risco_regiao
                .sort_values(
                    "score_medio",
                    ascending=False,
                )
            )

            col1, col2 = (
                st.columns(
                    [1.5, 1]
                )
            )

            with col1:

                st.pyplot(
                    criar_grafico_regional(
                        risco_regiao
                    ),
                    use_container_width=True,
                )

            with col2:

                st.dataframe(
                    risco_regiao,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "regiao":
                            "Região",

                        "alunos":
                            "Alunos",

                        "score_medio":
                            st.column_config.NumberColumn(
                                "Score médio",
                                format="%.2f%%",
                            ),

                        "prioritarios":
                            "Prioritários",

                        "pct_prioritarios":
                            st.column_config.NumberColumn(
                                "% prioritários",
                                format="%.2f%%",
                            ),

                        "alto_risco":
                            "Alto risco",

                        "pct_alto_risco":
                            st.column_config.NumberColumn(
                                "% alto risco",
                                format="%.2f%%",
                            ),
                    },
                )

        # ==========================================
        # ANÁLISE MUNICIPAL
        # ==========================================
        if (
            coluna_municipio is not None
            and total > 0
        ):

            renderizar_html(
                """
                <div class="section-title">
                    Análise por município
                </div>
                """
            )

            colunas_grupo = [
                coluna_municipio
            ]

            if (
                "sigla_uf"
                in dados_filtrados.columns
            ):

                colunas_grupo.append(
                    "sigla_uf"
                )

            risco_municipio = (
                dados_filtrados
                .groupby(
                    colunas_grupo,
                    as_index=False,
                )
                .agg(
                    alunos=(
                        "score_risco",
                        "size",
                    ),
                    score_medio=(
                        "score_risco",
                        "mean",
                    ),
                    prioritarios=(
                        "prioridade",
                        lambda x: (
                            x
                            == "Prioritário"
                        ).sum(),
                    ),
                    alto_risco=(
                        "alto_risco",
                        "sum",
                    ),
                )
            )

            risco_municipio[
                "pct_prioritarios"
            ] = (
                risco_municipio[
                    "prioritarios"
                ]
                / risco_municipio[
                    "alunos"
                ]
                * 100
            )

            risco_municipio[
                "pct_alto_risco"
            ] = (
                risco_municipio[
                    "alto_risco"
                ]
                / risco_municipio[
                    "alunos"
                ]
                * 100
            )

            if (
                "sigla_uf"
                in risco_municipio.columns
            ):

                risco_municipio[
                    "municipio_exibicao"
                ] = (
                    risco_municipio[
                        coluna_municipio
                    ]
                    .astype(str)
                    + " - "
                    + risco_municipio[
                        "sigla_uf"
                    ]
                    .astype(str)
                )

            else:

                risco_municipio[
                    "municipio_exibicao"
                ] = (
                    risco_municipio[
                        coluna_municipio
                    ]
                    .astype(str)
                )

            risco_municipio = (
                risco_municipio
                .sort_values(
                    "score_medio",
                    ascending=False,
                )
            )

            # --------------------------------------
            # MÍNIMO DE 30 PARA RANKING GERAL
            # --------------------------------------
            if (
                municipio_selecionado
                == "Todos"
            ):

                risco_municipio_exibicao = (
                    risco_municipio[
                        risco_municipio[
                            "alunos"
                        ] >= 30
                    ]
                    .copy()
                )

            else:

                risco_municipio_exibicao = (
                    risco_municipio.copy()
                )

            if (
                not risco_municipio_exibicao.empty
            ):

                if (
                    len(
                        risco_municipio_exibicao
                    ) > 1
                ):

                    st.pyplot(
                        criar_grafico_municipal(
                            risco_municipio_exibicao
                        ),
                        use_container_width=True,
                    )

                st.dataframe(
                    risco_municipio_exibicao[
                        [
                            "municipio_exibicao",
                            "alunos",
                            "score_medio",
                            "prioritarios",
                            "pct_prioritarios",
                            "alto_risco",
                            "pct_alto_risco",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "municipio_exibicao":
                            "Município",

                        "alunos":
                            "Alunos",

                        "score_medio":
                            st.column_config.NumberColumn(
                                "Score médio",
                                format="%.2f%%",
                            ),

                        "prioritarios":
                            "Prioritários",

                        "pct_prioritarios":
                            st.column_config.NumberColumn(
                                "% prioritários",
                                format="%.2f%%",
                            ),

                        "alto_risco":
                            "Alto risco",

                        "pct_alto_risco":
                            st.column_config.NumberColumn(
                                "% alto risco",
                                format="%.2f%%",
                            ),
                    },
                )

            else:

                st.info(
                    "Nenhum município possui pelo "
                    "menos 30 registros dentro do "
                    "filtro selecionado."
                )

        # ==========================================
        # RESULTADOS INDIVIDUAIS
        # ==========================================
        if total > 0:

            renderizar_html(
                """
                <div class="section-title">
                    Resultados individuais
                </div>
                """
            )

            colunas_resultado = []

            for coluna in [
                "id_aluno",
                coluna_municipio,
                "sigla_uf",
                "regiao",
                "rede",
                "score_risco",
                "prob_nao_alfabetizado",
                "prob_alfabetizado",
                "situacao_prevista",
                "prioridade",
                "alto_risco",
            ]:

                if (
                    coluna is not None
                    and coluna
                    in dados_filtrados.columns
                    and coluna
                    not in colunas_resultado
                ):

                    colunas_resultado.append(
                        coluna
                    )

            tabela_resultado = (
                dados_filtrados[
                    colunas_resultado
                ]
                .sort_values(
                    "score_risco",
                    ascending=False,
                )
            )

            configuracao_colunas = {
                "id_aluno":
                    "Aluno",

                "sigla_uf":
                    "UF",

                "regiao":
                    "Região",

                "rede":
                    "Rede",

                "score_risco":
                    st.column_config.ProgressColumn(
                        "Score de risco",
                        min_value=0,
                        max_value=100,
                        format="%.2f%%",
                    ),

                "prob_nao_alfabetizado":
                    st.column_config.NumberColumn(
                        "P(Não alfabetizado)",
                        format="%.4f",
                    ),

                "prob_alfabetizado":
                    st.column_config.NumberColumn(
                        "P(Alfabetizado)",
                        format="%.4f",
                    ),

                "situacao_prevista":
                    "Situação prevista",

                "prioridade":
                    "Prioridade",

                "alto_risco":
                    "Alto risco",
            }

            if coluna_municipio is not None:

                configuracao_colunas[
                    coluna_municipio
                ] = "Município"

            st.dataframe(
                tabela_resultado,
                use_container_width=True,
                hide_index=True,
                column_config=(
                    configuracao_colunas
                ),
            )

        # ==========================================
        # DOWNLOAD
        # ==========================================
        if total > 0:

            renderizar_html(
                """
                <div class="section-title">
                    Exportar resultados
                </div>
                """
            )

            dados_download = (
                dados_filtrados
                .drop(
                    columns=[
                        "municipio_filtro"
                    ],
                    errors="ignore",
                )
            )

            csv = (
                dados_download
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8-sig"
                )
            )

            st.download_button(
                "⬇️ Baixar resultados filtrados",
                data=csv,
                file_name=(
                    "resultado_alfabetizacao.csv"
                ),
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )

        renderizar_html(
            """
            <div class="warning-note">

                Os resultados representam estimativas
                produzidas pelo modelo sobre os registros
                enviados.

                A classificação deve ser utilizada como
                ferramenta de <strong>triagem e apoio à
                decisão</strong>, e não como diagnóstico
                pedagógico definitivo.

            </div>
            """
        )


# ==================================================
# SOBRE O PROJETO
# ==================================================
else:

    renderizar_html(
        """
        <div class="hero">

            <span class="badge">
                Tech Challenge — Fase 3
            </span>

            <h1>
                Sobre o projeto
            </h1>

            <p>
                Projeto de Ciência de Dados aplicado
                à alfabetização no Brasil, envolvendo
                enriquecimento de dados, análise
                exploratória, Machine Learning,
                interpretabilidade e aplicação
                prática das previsões.
            </p>

        </div>
        """
    )

    # ==============================================
    # OBJETIVO
    # ==============================================
    renderizar_html(
        """
        <div class="section-title">
            Objetivo
        </div>
        """
    )

    renderizar_html(
        """
        <div class="model-note">

            O objetivo principal é identificar alunos
            potencialmente <strong>não alfabetizados</strong>
            e transformar as probabilidades do modelo
            em informações que possam apoiar ações de
            acompanhamento e priorização educacional.

        </div>
        """
    )

    # ==============================================
    # METODOLOGIA
    # ==============================================
    renderizar_html(
        """
        <div class="section-title">
            Metodologia
        </div>
        """
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:

        renderizar_html(
            """
            <div class="info-card">

                <h3>
                    🗂️ Dados
                </h3>

                <p>
                    Integração da camada Gold com
                    informações territoriais,
                    populacionais, socioeconômicas
                    e dados do Censo Escolar.
                </p>

            </div>
            """
        )

    with col2:

        renderizar_html(
            """
            <div class="info-card">

                <h3>
                    🤖 Machine Learning
                </h3>

                <p>
                    Comparação de modelos, tuning,
                    ensembles, validação cruzada e
                    seleção do Random Forest como
                    solução final.
                </p>

            </div>
            """
        )

    with col3:

        renderizar_html(
            """
            <div class="info-card">

                <h3>
                    🔎 Interpretabilidade
                </h3>

                <p>
                    Feature Importance, SHAP e
                    Learning Curve foram utilizados
                    para compreender o comportamento
                    e a estabilidade do modelo.
                </p>

            </div>
            """
        )

    # ==============================================
    # RESULTADOS
    # ==============================================
    renderizar_html(
        """
        <div class="section-title">
            Modelo final
        </div>
        """
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Modelo",
        "Random Forest",
    )

    col2.metric(
        "Recall classe 0",
        "≈ 78%",
    )

    col3.metric(
        "ROC-AUC",
        "≈ 0.62",
    )

    col4.metric(
        "Threshold",
        "0.52",
    )

    # ==============================================
    # PIPELINE
    # ==============================================
    renderizar_html(
        """
        <div class="section-title">
            Pipeline de inferência
        </div>
        """
    )

    st.code(
        """Dados de entrada
        ↓
Pré-processamento
        ↓
SimpleImputer
        ↓
StandardScaler / OneHotEncoder
        ↓
ColumnTransformer
        ↓
VarianceThreshold
        ↓
Random Forest
        ↓
predict_proba()
        ↓
Threshold operacional
        ↓
Score de risco
        ↓
Priorização
        ↓
Análise regional e municipal""",
        language="text",
    )

    # ==============================================
    # TECNOLOGIAS
    # ==============================================
    renderizar_html(
        """
        <div class="section-title">
            Tecnologias utilizadas
        </div>
        """
    )

    st.code(
        """Python
Pandas
NumPy
Scikit-learn
XGBoost
SHAP
DuckDB
Matplotlib
Seaborn
Streamlit
Joblib
Parquet
SQL""",
        language="text",
    )

    # ==============================================
    # LIMITAÇÕES
    # ==============================================
    renderizar_html(
        """
        <div class="section-title">
            Limitações
        </div>
        """
    )

    renderizar_html(
        """
        <div class="analysis-card card-orange">

            <ul>

                <li>
                    O desempenho discriminatório
                    do modelo é moderado.
                </li>

                <li>
                    Muitas features representam
                    características contextuais
                    e territoriais.
                </li>

                <li>
                    O threshold prioriza a
                    identificação da classe
                    não alfabetizada.
                </li>

                <li>
                    Feature Importance e SHAP
                    não estabelecem causalidade.
                </li>

                <li>
                    Os resultados territoriais
                    não representam taxas oficiais
                    de alfabetização.
                </li>

                <li>
                    A solução deve ser utilizada
                    como ferramenta de triagem e
                    apoio à decisão.
                </li>

            </ul>

        </div>
        """
    )

    # ==============================================
    # AUTOR
    # ==============================================
    renderizar_html(
        """
        <div class="section-title">
            Autor
        </div>
        """
    )

    renderizar_html(
        """
        <div class="analysis-card card-blue">

            <h3>
                Renan Assis Trevelim
            </h3>

            <p>
                Projeto desenvolvido para o
                Tech Challenge — Fase 3,
                aplicando Ciência de Dados,
                Machine Learning e Inteligência
                Analítica ao contexto da
                alfabetização no Brasil.
            </p>

        </div>
        """
    )


# ==================================================
# RODAPÉ
# ==================================================
renderizar_html(
    """
    <div class="footer">

        Literacy Intelligence
        • Machine Learning
        • Educação
        • Renan Assis Trevelim

    </div>
    """
)