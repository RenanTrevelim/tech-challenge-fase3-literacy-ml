# Dados do projeto

Esta pasta concentra os conjuntos de dados utilizados ao longo do projeto, organizados por finalidade e estágio de processamento.

A estrutura segue uma separação entre **dados intermediários, dados consolidados, fontes externas e dados utilizados pela aplicação**, facilitando a rastreabilidade e a reprodução do fluxo de preparação dos dados.

---

## Estrutura

```text
data/
│
├── app/
│   └── dados_teste.parquet
│
├── external/
│   ├── microdados_censo_escolar_2023.zip
│   ├── microdados_censo_escolar_2024.zip
│   └── PIB dos Municípios - base de dados 2010-2023.xlsx
│
├── gold/
│   ├── dataset_eda.parquet
│   ├── dataset_modelagem.parquet
│   ├── gold_alunos.parquet
│   ├── gold_dataset_analitico.parquet
│   ├── gold_dataset_enriquecido.parquet
│   └── gold_municipal.parquet
│
├── silver/
│   ├── alunos.parquet
│   └── municipio.parquet
│
└── README.md
```

---

## `app/`

Contém os dados utilizados na demonstração da aplicação Streamlit.

### `dados_teste.parquet`

Subconjunto de teste utilizado para validar o funcionamento do pipeline exportado e alimentar a aplicação interativa.

Os registros são enviados ao pipeline completo, que executa automaticamente:

```text
Dados de entrada
      ↓
Pré-processamento
      ↓
VarianceThreshold
      ↓
Random Forest
      ↓
Probabilidades
      ↓
Score de risco
      ↓
Priorização
```

Esse arquivo também permite demonstrar os filtros e análises territoriais da aplicação.

---

## `external/`

Armazena as fontes externas utilizadas no enriquecimento da base analítica.

Entre os principais conjuntos utilizados estão:

- microdados do **Censo Escolar 2023**;
- microdados do **Censo Escolar 2024**;
- dados de **PIB dos municípios**;
- indicadores socioeconômicos;
- informações provenientes de fontes oficiais como **IBGE, SIDRA e INEP**.

Esses dados foram utilizados para adicionar informações territoriais, econômicas, populacionais e educacionais ao dataset original.

---

## `silver/`

Contém as bases intermediárias provenientes das etapas anteriores de tratamento dos dados.

### `alunos.parquet`

Base tratada com informações em nível de aluno.

### `municipio.parquet`

Base tratada contendo informações consolidadas em nível municipal.

A camada Silver funciona como uma etapa intermediária antes da construção das bases analíticas consolidadas.

---

## `gold/`

Contém os datasets consolidados utilizados nas etapas de análise exploratória, enriquecimento e Machine Learning.

### `gold_alunos.parquet`

Base consolidada contendo informações relacionadas aos alunos.

### `gold_municipal.parquet`

Base consolidada com informações em nível municipal.

### `gold_dataset_analitico.parquet`

Dataset construído a partir da integração das principais informações necessárias para a Fase 3.

### `gold_dataset_enriquecido.parquet`

Versão do dataset analítico após a inclusão das fontes externas utilizadas no enriquecimento.

### `dataset_eda.parquet`

Base preparada para a etapa de **Análise Exploratória de Dados**, mantendo informações necessárias para análises estatísticas, territoriais e de negócio.

### `dataset_modelagem.parquet`

Base final preparada para o treinamento dos modelos de Machine Learning, contendo somente as variáveis selecionadas para a etapa de modelagem.

---

## Fluxo dos dados

O fluxo principal utilizado no projeto pode ser resumido como:

```text
Silver
  +
Dados externos
      ↓
Preparação e integração
      ↓
Dataset analítico
      ↓
Enriquecimento
      ↓
Gold
      ↓
Análise Exploratória
      ↓
Dataset de modelagem
      ↓
Machine Learning
      ↓
Conjunto de teste
      ↓
Aplicação Streamlit
```

---

## Organização das camadas

A separação dos dados por finalidade permite manter uma estrutura mais clara durante o desenvolvimento:

| Camada | Finalidade |
|---|---|
| `external` | Fontes externas utilizadas no enriquecimento |
| `silver` | Dados intermediários e tratados |
| `gold` | Bases consolidadas para análise e modelagem |
| `app` | Dados utilizados na demonstração da aplicação |

Essa organização ajuda a manter **rastreabilidade, reprodutibilidade e separação entre as diferentes etapas do pipeline de dados**.