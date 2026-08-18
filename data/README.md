# Data — Arquitetura Medalhão

Esta pasta contém os dados processados do projeto **Tech Challenge — Fase 3**, organizados segundo a arquitetura Medalhão:

```text
data/
├── bronze/
├── silver/
└── gold/
```

Os arquivos originais utilizados como fonte não são mantidos nesta pasta. Eles são utilizados apenas no processo de ingestão para geração da camada Bronze.

---

## Arquitetura dos Dados

O fluxo adotado no projeto é:

```text
Dados de origem
      ↓
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
Interpretabilidade e Insights
```

Cada camada possui uma responsabilidade específica dentro do pipeline.

---

## `bronze/`

A camada Bronze representa a primeira etapa de ingestão dos dados.

Os arquivos provenientes das fontes originais são convertidos para o formato Parquet, preservando o conteúdo o mais próximo possível da origem.

Principais características:

* preservação da granularidade original;
* conversão dos arquivos para Parquet;
* ausência de regras de negócio;
* ausência de imputação de valores faltantes;
* ausência de transformações analíticas.

Arquivos utilizados:

```text
Alunos.parquet

br_inep_avaliacao_alfabetizacao_municipio.parquet
br_inep_avaliacao_alfabetizacao_uf.parquet

br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_municipio.parquet
br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_uf.parquet
br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_brasil.parquet
```

A camada Bronze funciona como ponto de entrada reproduzível do pipeline.

---

## `silver/`

A camada Silver contém os dados tratados, padronizados e integrados.

Nesta etapa são aplicadas transformações de qualidade e preparação necessárias para consumo analítico.

Principais transformações:

* padronização dos nomes das colunas;
* padronização de identificadores;
* conversão de tipos;
* limpeza de campos textuais;
* remoção de duplicatas exatas;
* padronização da rede de ensino;
* tratamento de códigos municipais;
* integração entre indicadores e metas.

### `alunos.parquet`

Dataset em nível individual.

Granularidade:

```text
1 linha = 1 aluno por ano
```

Contém informações como:

* ano;
* município;
* escola;
* aluno;
* caderno;
* série;
* rede de ensino;
* presença;
* preenchimento do caderno;
* alfabetização;
* proficiência;
* peso do aluno.

A granularidade individual é preservada para permitir análises e modelagem supervisionada nas etapas seguintes.

### `municipio.parquet`

Dataset analítico em nível municipal.

Granularidade principal:

```text
município × ano × rede
```

Contém a integração entre:

* indicador de alfabetização;
* métricas educacionais;
* metas municipais;
* metas estaduais;
* metas nacionais.

Os joins foram validados para evitar multiplicação indevida de registros.

---

## `gold/`

A camada Gold contém os datasets preparados para consumo analítico.

Nesta etapa, os dados da Silver são enriquecidos e organizados de acordo com os objetivos da Fase 3.

### `gold_alunos.parquet`

Dataset individual utilizado como principal base para:

* análise exploratória;
* análise de alfabetização;
* análise de presença;
* preparação da modelagem supervisionada.

Granularidade:

```text
1 linha = 1 aluno por ano
```

A Gold preserva variáveis como:

* alfabetização;
* presença;
* proficiência;
* contexto escolar;
* município;
* rede de ensino;
* informações contextuais adicionadas a partir da camada municipal.

Nesta etapa, variáveis potencialmente associadas a `data leakage` ainda são mantidas para análise.

A decisão final sobre filtros e seleção de features será realizada durante a EDA e a modelagem.

### `gold_municipal.parquet`

Dataset voltado para análises agregadas e inteligência analítica.

Granularidade:

```text
município × ano × rede
```

Pode ser utilizado para:

* análise territorial;
* comparação de indicadores;
* acompanhamento de metas;
* identificação de municípios vulneráveis;
* rankings;
* mapas;
* geração de insights para políticas públicas.

Também contém variáveis analíticas derivadas relacionadas ao atingimento das metas.

---

## Volume Atual das Camadas

Após o processamento:

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

As tabelas finais não apresentaram duplicatas exatas após o processamento.

---

## Decisões de Modelagem

Algumas decisões são propositalmente deixadas para as etapas de EDA e Machine Learning.

Entre elas:

* utilização ou não de alunos ausentes;
* uso da variável `presenca`;
* utilização ou exclusão da variável `proficiencia`;
* tratamento de valores faltantes;
* seleção das features;
* encoding de variáveis categóricas;
* normalização;
* definição de treino, validação e teste.

Essa abordagem evita antecipar decisões de modelagem durante a engenharia de dados.

---

## Data Leakage

O projeto considera explicitamente o risco de `data leakage`.

Variáveis que possuem relação direta com a variável alvo não devem ser automaticamente utilizadas como features.

Um exemplo relevante é:

```text
proficiencia
```

Como a classificação de alfabetização pode estar diretamente relacionada à proficiência, essa variável será avaliada durante a EDA antes da modelagem.

A Gold preserva essa informação para permitir análise e auditoria.

---

## Reprodutibilidade

A arquitetura permite reconstruir os dados seguindo o fluxo:

```text
Dados de origem
      ↓
Bronze
      ↓
Silver
      ↓
Gold
```

Essa separação facilita:

* rastreabilidade;
* manutenção;
* reprocessamento;
* validação das transformações;
* reprodutibilidade analítica.

---

## Versionamento dos Dados

Os arquivos originais não são mantidos no repositório.

Dependendo do tamanho dos datasets, também pode ser interessante excluir algumas camadas do Git utilizando `.gitignore`.

Exemplo:

```gitignore
data/bronze/
data/silver/
data/gold/
```

Caso seja necessário disponibilizar exemplos no repositório, pode ser criada uma pasta:

```text
data/sample/
```

contendo apenas pequenas amostras dos datasets.

---

## Próximas Etapas

Com a camada Gold finalizada, o projeto segue para:

```text
Gold
 ↓
Análise Exploratória de Dados
 ↓
Definição de hipóteses
 ↓
Seleção de features
 ↓
Pipeline de Machine Learning
 ↓
Treinamento e validação
 ↓
Avaliação dos modelos
 ↓
Interpretabilidade
 ↓
Insights para tomada de decisão
```

A camada Gold será utilizada como principal fonte de dados para as etapas analíticas e preditivas da Fase 3.
