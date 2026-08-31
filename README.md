# Tech Challenge Fase 3 — Predição e Inteligência Analítica para Alfabetização no Brasil

Projeto de Ciência de Dados desenvolvido para o **Tech Challenge — Fase 3**, com foco na construção de uma solução de Machine Learning capaz de apoiar a identificação de alunos potencialmente **não alfabetizados**.

A solução utiliza dados provenientes da **camada Gold construída na Fase 2**, enriquecidos com informações territoriais, populacionais, socioeconômicas e educacionais.

O projeto contempla desde a preparação e análise dos dados até a construção, validação e interpretação do modelo, finalizando com uma aplicação prática das probabilidades para **priorização de alunos, municípios e regiões**.

---

## Visão geral

O projeto busca responder principalmente à seguinte questão:

> **Quais alunos apresentam maior risco estimado de não alfabetização e como essas informações podem apoiar ações educacionais?**

Para isso, foi desenvolvido um problema de classificação supervisionada:

```text
0 → Não alfabetizado
1 → Alfabetizado
```

Como o principal objetivo é apoiar a identificação de alunos potencialmente não alfabetizados, a **classe 0** foi definida como prioritária durante a avaliação dos modelos.

A solução foi estruturada para gerar informações em três níveis:

```text
Aluno
  ↓
Município
  ↓
Região
```

O modelo deve ser interpretado como uma ferramenta de **triagem e apoio à decisão**, e não como diagnóstico pedagógico definitivo.

---

## Objetivos

Os principais objetivos do projeto são:

- utilizar a camada Gold construída na Fase 2;
- integrar e enriquecer dados educacionais com informações externas;
- analisar fatores associados à alfabetização;
- investigar padrões territoriais e socioeconômicos;
- identificar possíveis fontes de data leakage;
- construir um pipeline completo de Machine Learning;
- comparar diferentes algoritmos de classificação;
- otimizar os modelos mais promissores;
- avaliar estratégias de ensemble;
- validar estabilidade e generalização;
- interpretar as previsões com Feature Importance e SHAP;
- priorizar a identificação de alunos não alfabetizados;
- transformar as probabilidades em informações úteis para apoio à decisão.

---

# Dados utilizados

A base analítica foi construída a partir da integração dos dados da camada **Gold** da Fase 2.

Posteriormente, o dataset foi enriquecido com informações externas oficiais para ampliar o contexto disponível durante a análise e a modelagem.

Foram utilizadas informações relacionadas a:

- Indicador Criança Alfabetizada;
- informações municipais;
- Unidade Federativa e região;
- população;
- área territorial;
- densidade demográfica;
- PIB per capita;
- renda domiciliar per capita mediana;
- rede de ensino;
- infraestrutura escolar;
- acesso à internet;
- biblioteca e sala de leitura;
- água potável e esgotamento sanitário;
- acessibilidade;
- média de alunos por turma;
- média de alunos por docente;
- indicadores temporais.

### Principais fontes externas

- **IBGE**
- **SIDRA**
- **INEP**
- **Censo Escolar**

Os dados do Censo Escolar foram agregados por contexto de:

```text
município + ano + rede
```

permitindo incorporar informações educacionais complementares sem perda dos registros individuais dos alunos.

> Os indicadores agregados representam o contexto educacional do município, ano e rede, não necessariamente a escola específica frequentada pelo aluno.

---

# Arquitetura da solução

```text
Camada Gold — Fase 2
        ↓
Preparação do dataset analítico
        ↓
Enriquecimento com dados externos
        ↓
Análise Exploratória
        ↓
Seleção das variáveis
        ↓
Treino / Validação / Teste
        ↓
Pré-processamento
        │
        ├── SimpleImputer
        ├── StandardScaler
        ├── OneHotEncoder
        └── ColumnTransformer
        ↓
VarianceThreshold
        ↓
Modelos de Machine Learning
        ↓
Otimização de hiperparâmetros
        ↓
Voting / Stacking
        ↓
Seleção do Random Forest
        ↓
Ajuste de threshold
        ↓
Validação cruzada
        ↓
Avaliação final
        ↓
Feature Importance + SHAP
        ↓
Learning Curve
        ↓
Score de risco
        ↓
Priorização de alunos e territórios
        ↓
Pipeline final
```

---

# Estrutura do projeto

```text
tech-challenge-fase3-literacy-ml/
│
├── data/
│   ├── external/
│   ├── gold/
│   └── silver/
│
├── models/
│   └── modelo_final_random_forest.pkl
│
├── notebooks/
│   ├── 00_preparacao_dataset_analitico.ipynb
│   ├── 01_enriquecimento_dados_externos.ipynb
│   ├── 02_analise_exploratoria.ipynb
│   └── 03_modelos_machine_learning.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# Notebooks

## 00 — Preparação do dataset analítico

Responsável pela construção da base principal utilizada na Fase 3.

Principais etapas:

- leitura das bases Gold;
- integração dos dados de alunos e municípios;
- validação das chaves;
- análise de duplicidades;
- validação da variável alvo;
- geração do dataset analítico.

---

## 01 — Enriquecimento com dados externos

Responsável pela inclusão de novas dimensões ao dataset.

Foram adicionadas informações relacionadas a:

- região;
- PIB per capita;
- população;
- área territorial;
- densidade demográfica;
- renda domiciliar per capita;
- infraestrutura escolar;
- conectividade;
- acessibilidade;
- organização das turmas.

O objetivo foi ampliar a representação do contexto territorial, socioeconômico e educacional dos alunos.

---

## 02 — Análise Exploratória de Dados

Responsável pela investigação dos padrões presentes na base antes da modelagem.

Foram analisados:

- qualidade dos dados;
- valores ausentes;
- distribuição da variável alvo;
- variáveis numéricas;
- variáveis categóricas;
- correlação de Spearman;
- Cramér's V;
- diferenças entre regiões e UFs;
- padrões municipais;
- comportamento do caderno 12;
- hipóteses;
- possíveis fontes de data leakage.

### Data leakage

Durante o EDA foram identificadas variáveis com forte relação com o próprio processo de avaliação.

Entre elas:

```text
presenca
preenchimento_caderno
proficiencia
```

Essas variáveis foram removidas dos preditores para evitar que o modelo utilizasse informações que não seriam adequadas em um cenário real de previsão.

A variável `caderno` também foi retirada após investigação de um comportamento operacional atípico, principalmente associado ao caderno 12.

As variáveis de metas municipais e estaduais também foram removidas da modelagem devido à elevada proporção de valores ausentes.

---

## 03 — Modelagem de Machine Learning

Responsável pela construção e avaliação dos modelos supervisionados.

O fluxo inclui:

- definição do target;
- separação em treino, validação e teste;
- pré-processamento;
- seleção por variância;
- treinamento de modelos baseline;
- otimização de hiperparâmetros;
- ensembles;
- comparação consolidada;
- análise de threshold;
- validação cruzada;
- avaliação no conjunto de teste;
- Feature Importance;
- SHAP;
- Learning Curve;
- aplicação prática das probabilidades;
- salvamento do pipeline.

---

# Pré-processamento

As variáveis numéricas e categóricas receberam tratamentos diferentes.

### Variáveis numéricas

- `SimpleImputer(strategy="median")`
- `StandardScaler`

### Variáveis categóricas

- `SimpleImputer(strategy="most_frequent")`
- `OneHotEncoder`

As transformações foram integradas através do `ColumnTransformer`.

O preprocessador foi ajustado exclusivamente sobre os dados de treino para evitar vazamento de informação.

Após o pré-processamento foi aplicado:

```text
VarianceThreshold(threshold=0.01)
```

reduzindo o conjunto de:

```text
49 → 46 features
```

---

# Divisão dos dados

A base foi dividida em:

```text
64% → Treino
16% → Validação
20% → Teste
```

A separação foi realizada de forma estratificada, preservando a distribuição da variável alvo.

O conjunto de teste permaneceu isolado durante as etapas de treinamento, tuning e definição do threshold.

---

# Modelos avaliados

Foram avaliados inicialmente:

- Regressão Logística;
- Árvore de Decisão;
- Random Forest;
- KNN;
- XGBoost.

Os modelos mais promissores avançaram para a otimização de hiperparâmetros utilizando:

```text
RandomizedSearchCV
```

Também foram avaliadas estratégias de ensemble:

- Voting Classifier;
- Stacking Classifier.

---

# Comparação dos modelos

Após tuning e ensembles, os principais resultados no conjunto de validação foram:

| Modelo | Accuracy | Recall 0 | Recall 1 | F1 Macro | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0,5922 | 0,6394 | 0,5428 | 0,5906 | 0,6337 |
| Stacking | 0,5902 | 0,6199 | 0,5592 | 0,5895 | 0,6323 |
| Voting | 0,5873 | 0,6831 | 0,4873 | 0,5822 | 0,6300 |
| XGBoost | 0,5792 | 0,7264 | 0,4255 | 0,5677 | 0,6199 |
| Regressão Logística | 0,5221 | 0,8723 | 0,1567 | 0,4469 | 0,5562 |

A Regressão Logística apresentou o maior Recall da classe 0, porém com forte perda de desempenho na classe alfabetizada.

XGBoost e Voting também aumentaram a sensibilidade para a classe prioritária, mas apresentaram menor equilíbrio entre as classes.

O **Random Forest otimizado** apresentou o melhor compromisso entre:

- Recall da classe prioritária;
- equilíbrio entre as classes;
- F1 Macro;
- ROC-AUC;
- simplicidade operacional.

Por esse motivo, foi selecionado como modelo final.

---

# Ajuste do threshold

O threshold padrão de classificação é:

```text
0.50
```

Como o objetivo principal é aumentar a identificação dos alunos não alfabetizados, diferentes valores foram analisados no conjunto de validação.

Foi selecionado:

```text
Threshold operacional = 0.52
```

Esse threshold é aplicado sobre:

```text
P(Alfabetizado)
```

Portanto:

```text
P(Alfabetizado) >= 0.52
→ Alfabetizado

P(Alfabetizado) < 0.52
→ Não alfabetizado
```

A escolha elevou o Recall da classe 0 de aproximadamente:

```text
63,9% → 78,3%
```

na validação, assumindo como trade-off uma redução no Recall da classe alfabetizada.

A decisão foi tomada considerando o modelo como ferramenta de **triagem**, priorizando a identificação de alunos potencialmente em risco.

---

# Validação cruzada

O Random Forest foi avaliado utilizando validação cruzada estratificada com 5 folds.

| Métrica | Média | Desvio padrão |
|---|---:|---:|
| Accuracy | 0,5901 | 0,0022 |
| Recall 0 | 0,6514 | 0,0127 |
| Recall 1 | 0,5261 | 0,0147 |
| F1 0 | 0,6187 | 0,0046 |
| F1 1 | 0,5567 | 0,0075 |
| F1 Macro | 0,5877 | 0,0026 |
| ROC-AUC | 0,6290 | 0,0043 |

Os baixos desvios entre os folds indicaram comportamento consistente em diferentes divisões dos dados.

---

# Resultado final

O conjunto de teste foi utilizado após a conclusão das etapas de seleção do modelo e definição do threshold.

Com o Random Forest e threshold `0.52`, foram obtidos:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0,58 |
| Recall — Não alfabetizado | **0,78** |
| Recall — Alfabetizado | 0,37 |
| F1 — Não alfabetizado | 0,66 |
| F1 Macro | 0,56 |
| ROC-AUC | **0,62** |

O modelo identificou corretamente:

```text
4.616 de 5.894 alunos não alfabetizados
```

correspondendo a aproximadamente **78% de Recall da classe prioritária**.

O resultado está alinhado ao objetivo do projeto de aumentar a sensibilidade para alunos potencialmente não alfabetizados.

---

# Interpretabilidade

A interpretação do modelo foi realizada utilizando duas abordagens.

## Feature Importance

A importância interna do Random Forest destacou principalmente variáveis relacionadas a:

- Unidade Federativa;
- região;
- quantidade de escolas;
- internet para aprendizagem;
- acesso à internet;
- densidade demográfica;
- biblioteca;
- água potável;
- população.

## SHAP

Para aprofundar a análise foi utilizado o **SHAP — SHapley Additive exPlanations**.

Foram construídos:

- Summary Plot em barras;
- Summary Plot;
- Dependence Plot;
- Waterfall Plot.

As análises foram direcionadas principalmente para:

```text
Classe 0 → Não alfabetizado
```

O SHAP permitiu analisar:

- quais features possuem maior impacto;
- direção das contribuições;
- comportamento de variáveis específicas;
- explicação individual das previsões.

> Feature Importance e SHAP representam relações preditivas aprendidas pelo modelo e não devem ser interpretadas como causalidade.

---

# Diagnóstico de overfitting

Foi construída uma **Learning Curve** utilizando o Recall da classe 0.

Resultados finais:

| Indicador | Resultado |
|---|---:|
| Recall treino | 0,6647 |
| Recall validação | 0,6556 |
| Gap | 0,0091 |

O pequeno gap entre treino e validação indica **baixo indício de overfitting**.

As curvas também apresentaram estabilização conforme o volume de dados aumentou, reforçando a consistência do modelo.

---

# Aplicação prática do modelo

Além da classificação, as probabilidades produzidas pelo Random Forest foram utilizadas para construir uma camada de inteligência analítica.

Foi utilizado **DuckDB** para realizar consultas SQL diretamente sobre os resultados das previsões.

## Score de risco

A probabilidade da classe 0 foi transformada em um score:

```text
P(Não alfabetizado)
        ↓
Score de risco (%)
```

Esse score permite ordenar os alunos de acordo com o risco estimado.

---

## Priorização dos alunos

Como o threshold operacional de `0.52` é aplicado à classe alfabetizada, o limite equivalente para a classe de risco corresponde aproximadamente a:

```text
48%
```

Foi utilizada a seguinte regra:

```text
Score > 48%
→ Prioritário

Score <= 48%
→ Monitoramento
```

Essa classificação cria uma primeira fila de priorização para acompanhamento.

Como uma parcela elevada dos alunos foi classificada como prioritária, foi criado um segundo nível utilizando o **percentil 75 dos scores prioritários**.

O limite encontrado foi aproximadamente:

```text
56,86%
```

permitindo destacar os casos considerados de **alto risco**.

---

## Análise municipal

Os resultados foram agregados por município para identificar localidades com maior concentração de alunos classificados como alto risco.

Para reduzir distorções causadas por poucos registros, foram considerados apenas municípios com pelo menos:

```text
30 alunos
```

A análise municipal permite comparar:

- total de alunos;
- score médio;
- alunos de alto risco;
- percentual de alto risco.

> Os percentuais representam os registros presentes na amostra analisada e não devem ser interpretados como taxas oficiais de alfabetização municipal.

---

## Análise regional

Os resultados também foram agregados pelas regiões brasileiras.

Foram analisados:

- total de alunos;
- score médio;
- percentual de prioritários;
- quantidade de alunos de alto risco;
- percentual de alto risco.

Essa visão complementa a análise municipal, permitindo identificar padrões territoriais mais amplos e apoiar decisões de planejamento.

---

# Geração de valor

A solução transforma a saída do Machine Learning em diferentes níveis de informação:

```text
Modelo
   ↓
Probabilidade
   ↓
Score de risco
   ↓
Priorização de alunos
   ↓
Casos de alto risco
   ↓
Municípios prioritários
   ↓
Análise regional
   ↓
Apoio à decisão
```

Entre as possíveis aplicações estão:

- priorização de alunos para avaliação pedagógica;
- direcionamento de acompanhamento;
- identificação de territórios que demandam maior atenção;
- apoio à alocação de recursos;
- planejamento de ações educacionais;
- suporte à tomada de decisão baseada em dados.

---

# Pipeline final

As principais etapas utilizadas durante a modelagem foram agrupadas em um único pipeline:

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
```

O pipeline foi exportado utilizando `Joblib`:

```text
models/modelo_final_random_forest.pkl
```

Isso permite reutilizar as mesmas transformações realizadas no treinamento durante novas inferências.

---

# Tecnologias

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- DuckDB
- SQL
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook
- Parquet
- Git
- GitHub

---

# Instalação

Clone o repositório:

```bash
git clone <URL_DO_REPOSITORIO>
cd tech-challenge-fase3-literacy-ml
```

Crie o ambiente virtual:

```bash
python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# Execução

Execute os notebooks na seguinte ordem:

```text
00 → Preparação do dataset analítico
01 → Enriquecimento com dados externos
02 → Análise Exploratória
03 → Modelagem de Machine Learning
```

---

# Limitações

Algumas limitações devem ser consideradas:

- o ROC-AUC indica capacidade discriminatória moderada;
- muitas features representam características contextuais e territoriais;
- o modelo possui poucas informações individuais diretamente relacionadas ao processo de aprendizagem;
- o threshold de `0.52` aumenta o Recall da classe não alfabetizada, mas reduz o desempenho da classe alfabetizada;
- municípios e regiões são analisados somente com base nos registros disponíveis na amostra;
- alunos inseridos em contextos semelhantes podem receber scores próximos;
- Feature Importance e SHAP não representam causalidade;
- a solução deve ser utilizada como ferramenta de triagem e não como diagnóstico definitivo.

---

# Próximos passos

Possíveis evoluções do projeto incluem:

- inclusão de novas variáveis socioeconômicas e educacionais;
- incorporação de informações mais individualizadas dos alunos;
- monitoramento de drift;
- rastreamento de experimentos com MLflow;
- criação de API com FastAPI;
- desenvolvimento de dashboard para gestores;
- integração com banco de dados;
- deploy em cloud;
- pipeline CI/CD;
- monitoramento contínuo do modelo.

---

# Conclusão

O projeto implementa um fluxo completo de Ciência de Dados aplicado ao contexto da alfabetização, desde a preparação e enriquecimento dos dados até a construção e interpretação do modelo final.

O **Random Forest otimizado** foi selecionado por apresentar o melhor equilíbrio geral entre os modelos avaliados.

Com o threshold operacional de **0,52**, o modelo alcançou aproximadamente **78% de Recall para alunos não alfabetizados no conjunto de teste**, atendendo ao objetivo de priorizar a identificação da classe de maior interesse.

As análises de **Feature Importance, SHAP e Learning Curve** contribuíram para avaliar transparência, comportamento e capacidade de generalização do modelo.

Além da previsão, as probabilidades foram transformadas em um **score de risco**, permitindo gerar análises em nível de:

```text
Aluno
  ↓
Município
  ↓
Região
```

Dessa forma, o projeto demonstra como Machine Learning pode ser utilizado não apenas para gerar previsões, mas também para apoiar **triagem, priorização e planejamento de ações educacionais orientadas por dados**.

---

## Autor

Desenvolvido por **Renan Trevelim**.

Projeto desenvolvido para o **Tech Challenge — Fase 3**, com foco em Ciência de Dados, Machine Learning, interpretabilidade e inteligência analítica aplicada à alfabetização no Brasil.