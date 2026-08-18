# Data

Esta pasta concentra os datasets utilizados no desenvolvimento do **Tech Challenge — Fase 3**, organizados em três camadas:

```text
data/
├── bronze/
├── silver/
└── gold/
```

A separação entre as camadas facilita a organização dos dados e deixa claro quais arquivos representam a entrada, o tratamento intermediário e as bases finais utilizadas nas análises.

---

## Bronze

A camada Bronze contém os dados em formato Parquet, preservando a estrutura original utilizada no projeto.

Principais conjuntos de dados:

* dados de alunos;
* indicadores de alfabetização por município;
* indicadores de alfabetização por UF;
* metas municipais de alfabetização;
* metas estaduais de alfabetização;
* metas nacionais de alfabetização.

Essa camada funciona como ponto de partida para as transformações realizadas nas etapas seguintes.

---

## Silver

A camada Silver contém os dados tratados, padronizados e organizados para consumo analítico.

Arquivos principais:

```text
alunos.parquet
municipio.parquet
```

### `alunos.parquet`

Mantém os dados em nível individual, com informações relacionadas a aluno, escola, município, rede de ensino, presença, alfabetização e proficiência.

Granularidade:

```text
1 linha = 1 aluno por ano
```

### `municipio.parquet`

Contém os indicadores educacionais consolidados em nível municipal, incluindo informações de alfabetização e metas.

Essa base é utilizada como apoio para análises territoriais e para enriquecimento das bases finais.

---

## Gold

A camada Gold contém os datasets preparados para as etapas de **Análise Exploratória de Dados e Machine Learning**.

Arquivos principais:

```text
gold_alunos.parquet
gold_municipal.parquet
```

### `gold_alunos.parquet`

Base individual utilizada para:

* análise do comportamento dos alunos;
* investigação da variável de alfabetização;
* preparação dos modelos supervisionados;
* avaliação de possíveis variáveis explicativas.

A base mantém variáveis relevantes para análise, enquanto decisões como seleção de features, tratamento de valores faltantes e prevenção de data leakage são realizadas posteriormente durante a etapa de modelagem.

### `gold_municipal.parquet`

Base analítica voltada para:

* análise territorial;
* comparação entre municípios;
* acompanhamento de indicadores;
* análise de metas educacionais;
* identificação de regiões com maior vulnerabilidade;
* geração de insights para apoio à tomada de decisão.

---

## Estrutura Atual

As principais bases finais possuem os seguintes volumes:

```text
Silver Alunos
57.782 registros
13 colunas

Silver Municipal
23.995 registros
47 colunas

Gold Alunos
57.782 registros
16 colunas

Gold Municipal
23.995 registros
51 colunas
```

---

## Fluxo dos Dados

```text
Bronze
   ↓
Silver
   ↓
Gold
   ↓
EDA
   ↓
Machine Learning
   ↓
Avaliação e Interpretabilidade
```

A camada Gold é utilizada como principal fonte de dados para as análises e modelos desenvolvidos na Fase 3.

---

## Observação

Os dados originais utilizados para geração da camada Bronze não são mantidos neste diretório.

A pasta `data/` contém apenas os datasets necessários para dar continuidade às etapas analíticas do projeto.
