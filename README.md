# Tech Challenge Fase 3 — Predição e Inteligência Analítica para Alfabetização no Brasil

Projeto de Ciência de Dados desenvolvido para o **Tech Challenge — Fase 3**, com foco na construção de uma solução de Machine Learning capaz de apoiar a identificação de alunos potencialmente **não alfabetizados**.

A solução utiliza dados provenientes da **camada Gold construída na Fase 2**, enriquecidos com informações territoriais, populacionais, socioeconômicas e educacionais.

O projeto contempla **preparação e enriquecimento dos dados, análise exploratória, modelagem supervisionada, otimização de hiperparâmetros, ensembles, validação, interpretabilidade, geração de scores de risco e uma aplicação interativa em Streamlit** para consumo das previsões.

---

## Visão geral

O projeto busca responder principalmente à seguinte questão:

> **Quais alunos apresentam maior risco estimado de não alfabetização e como essas informações podem apoiar ações educacionais?**

O problema foi estruturado como uma classificação supervisionada:

```text
0 → Não alfabetizado
1 → Alfabetizado
```

Como o principal objetivo é apoiar a identificação de alunos potencialmente não alfabetizados, a **classe 0** foi definida como prioritária durante a avaliação dos modelos.

A solução permite analisar os resultados em diferentes níveis:

```text
Aluno
  ↓
Município
  ↓
Região
```

O modelo deve ser utilizado como uma ferramenta de **triagem e apoio à decisão**, e não como diagnóstico pedagógico definitivo.

---

## Objetivos

Os principais objetivos do projeto são:

- utilizar a camada Gold construída na Fase 2;
- enriquecer os dados com informações externas oficiais;
- investigar fatores associados à alfabetização;
- analisar padrões territoriais e socioeconômicos;
- identificar e remover possíveis fontes de data leakage;
- construir um pipeline completo e reprodutível de Machine Learning;
- comparar diferentes algoritmos de classificação;
- otimizar os modelos mais promissores;
- avaliar estratégias de ensemble;
- validar estabilidade e generalização;
- interpretar as previsões com Feature Importance e SHAP;
- priorizar a identificação de alunos não alfabetizados;
- transformar probabilidades em scores de risco;
- disponibilizar os resultados em uma aplicação Streamlit interativa.

---

# Dados utilizados

A base analítica foi construída a partir da integração dos dados da camada **Gold** produzida na Fase 2.

Posteriormente, o dataset foi enriquecido com informações externas para ampliar a representação do contexto territorial, econômico e educacional dos alunos.

Foram incorporadas informações relacionadas a:

- Unidade Federativa e região;
- população;
- área territorial;
- densidade demográfica;
- PIB per capita;
- renda domiciliar per capita mediana;
- rede de ensino;
- quantidade de escolas;
- infraestrutura escolar;
- acesso à internet;
- internet voltada à aprendizagem;
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

Os dados do Censo Escolar foram agregados no contexto:

```text
município + ano + rede
```

permitindo incorporar informações educacionais complementares à base individual.

> Os indicadores agregados representam o contexto educacional do município, ano e rede e não necessariamente a escola específica frequentada pelo aluno.

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
Random Forest final
        ↓
Ajuste de threshold
        ↓
Validação cruzada
        ↓
Avaliação no conjunto de teste
        ↓
Feature Importance + SHAP
        ↓
Learning Curve
        ↓
Pipeline serializado
        ↓
Aplicação Streamlit
        │
        ├── Upload CSV / Parquet
        ├── Inferência
        ├── Score de risco
        ├── Priorização
        ├── Filtros territoriais
        ├── Análise municipal
        ├── Análise regional
        └── Exportação dos resultados
```

---

# Estrutura do projeto

```text
tech-challenge-fase3-literacy-ml/
│
├── data/
│   ├── external/
│   ├── silver/
│   ├── gold/
│   └── app/
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
├── src/
│   ├── __init__.py
│   ├── app.py
│   └── predict.py
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

O enriquecimento amplia a representação do contexto territorial, socioeconômico e educacional utilizado pelo modelo.

---

## 02 — Análise Exploratória de Dados

Responsável pela investigação dos padrões presentes na base antes da modelagem.

Foram analisados:

- qualidade dos dados;
- valores ausentes;
- distribuição da variável alvo;
- variáveis numéricas e categóricas;
- correlação de Spearman;
- Cramér's V;
- diferenças entre regiões e UFs;
- padrões municipais;
- comportamento do caderno 12;
- hipóteses;
- possíveis fontes de data leakage.

### Data leakage

Durante o EDA foram identificadas variáveis fortemente relacionadas ao próprio processo de avaliação:

```text
presenca
preenchimento_caderno
proficiencia
```

Essas variáveis foram removidas dos preditores para evitar que o modelo utilizasse informações inadequadas em um cenário real de inferência.

A variável `caderno` também foi retirada após a investigação de um comportamento operacional atípico, principalmente associado ao caderno 12.

As variáveis de metas municipais e estaduais foram removidas devido à elevada proporção de valores ausentes.

---

## 03 — Modelagem de Machine Learning

Responsável pela construção, validação e aplicação dos modelos supervisionados.

O fluxo inclui:

- definição do target;
- divisão entre treino, validação e teste;
- pré-processamento;
- seleção por variância;
- treinamento de modelos baseline;
- otimização de hiperparâmetros;
- Voting e Stacking;
- comparação dos modelos;
- ajuste de threshold;
- validação cruzada;
- avaliação final;
- Feature Importance;
- SHAP;
- Learning Curve;
- aplicação prática das probabilidades;
- serialização do pipeline.

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

O conjunto de teste permaneceu isolado durante treinamento, tuning, seleção do modelo e definição do threshold.

---

# Modelos avaliados

Foram avaliados inicialmente:

- Regressão Logística;
- Árvore de Decisão;
- Random Forest;
- KNN;
- XGBoost.

Os modelos mais promissores avançaram para otimização utilizando:

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

A Regressão Logística apresentou alto Recall para a classe 0, porém com forte perda de desempenho na classe alfabetizada.

XGBoost e Voting também aumentaram a sensibilidade para a classe prioritária, mas apresentaram menor equilíbrio entre as classes.

O **Random Forest otimizado** apresentou o melhor compromisso geral entre:

- Recall da classe prioritária;
- equilíbrio entre as classes;
- F1 Macro;
- ROC-AUC;
- estabilidade;
- simplicidade operacional.

Por esse motivo, foi selecionado como modelo final.

---

# Ajuste do threshold

O threshold padrão de classificação é:

```text
0.50
```

Como o objetivo principal é aumentar a identificação de alunos não alfabetizados, diferentes valores foram avaliados no conjunto de validação.

Foi selecionado:

```text
Threshold operacional = 0.52
```

O threshold é aplicado sobre:

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

Na validação, o Recall da classe 0 aumentou aproximadamente de:

```text
63,9% → 78,3%
```

assumindo como trade-off uma redução na identificação da classe alfabetizada.

A decisão foi tomada considerando o modelo como ferramenta de **triagem**, onde identificar alunos potencialmente em risco possui maior prioridade.

---

# Validação cruzada

O Random Forest final foi avaliado utilizando validação cruzada estratificada com 5 folds.

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

O conjunto de teste foi utilizado somente após a conclusão das decisões de modelagem.

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

O resultado está alinhado ao objetivo de aumentar a sensibilidade para alunos potencialmente não alfabetizados.

---

# Interpretabilidade

A interpretação do modelo foi realizada utilizando **Feature Importance** e **SHAP**.

## Feature Importance

Entre as principais variáveis destacadas pelo Random Forest aparecem características relacionadas a:

- Unidade Federativa;
- região;
- quantidade de escolas;
- internet para aprendizagem;
- acesso à internet;
- densidade demográfica;
- biblioteca e sala de leitura;
- água potável;
- população.

## SHAP

Para aprofundar a interpretação foi utilizado **SHAP — SHapley Additive exPlanations**.

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

- importância global das features;
- magnitude das contribuições;
- direção dos impactos;
- comportamento de variáveis específicas;
- explicações individuais das previsões.

> Feature Importance e SHAP explicam o comportamento preditivo do modelo, mas não estabelecem relações de causalidade.

---

# Diagnóstico de overfitting

Foi construída uma **Learning Curve** utilizando Recall da classe 0.

| Indicador | Resultado |
|---|---:|
| Recall treino | 0,6647 |
| Recall validação | 0,6556 |
| Gap | 0,0091 |

O pequeno gap entre treino e validação indica **baixo indício de overfitting**.

As curvas também apresentaram estabilização conforme o volume de dados aumentou, indicando comportamento consistente do modelo.

---

# Aplicação prática das previsões

As probabilidades produzidas pelo Random Forest foram transformadas em uma camada de inteligência analítica.

## Score de risco

A probabilidade da classe 0 foi transformada em um score percentual:

```text
P(Não alfabetizado)
        ↓
Score de risco (%)
```

Quanto maior o score, maior o risco estimado pelo modelo.

---

## Priorização dos alunos

Como o threshold `0.52` é aplicado sobre a probabilidade da classe alfabetizada, o limite equivalente para o risco de não alfabetização é:

```text
48%
```

A regra operacional utilizada é:

```text
Score > 48%
→ Prioritário

Score <= 48%
→ Monitoramento
```

Essa classificação cria uma primeira camada de priorização.

Como uma parcela elevada dos registros foi classificada como prioritária, foi criado um segundo nível utilizando o **percentil 75 dos scores prioritários**.

No conjunto analisado, o limite encontrado foi aproximadamente:

```text
56,86%
```

permitindo destacar os casos considerados de **alto risco**.

---

## Análise municipal

Os resultados foram agregados por município para identificar localidades com maior concentração de risco.

Foram analisados:

- quantidade de alunos;
- score médio;
- alunos prioritários;
- alunos de alto risco;
- percentual de alto risco.

Na análise do notebook, foi utilizado um mínimo de 30 registros por município para reduzir distorções causadas por amostras muito pequenas.

> Os resultados representam os registros presentes na amostra analisada e não devem ser interpretados como taxas oficiais de alfabetização municipal.

---

## Análise regional

Os resultados também foram agregados pelas regiões brasileiras.

Foram analisados:

- total de alunos;
- score médio;
- percentual de prioritários;
- quantidade de alunos de alto risco;
- percentual de alto risco.

Essa visão complementa a análise municipal e permite identificar padrões territoriais mais amplos.

---

# Aplicação Streamlit

Para disponibilizar a solução de forma interativa foi desenvolvida uma aplicação utilizando **Streamlit**.

A aplicação consome diretamente o pipeline serializado e permite executar novas inferências sobre arquivos enviados pelo usuário.

O fluxo da aplicação é:

```text
Upload CSV / Parquet
        ↓
Validação das features
        ↓
Pipeline de Machine Learning
        ↓
Pré-processamento
        ↓
VarianceThreshold
        ↓
Random Forest
        ↓
predict_proba()
        ↓
Threshold 0.52
        ↓
Score de risco
        ↓
Priorização
        ↓
Análises territoriais
```

### Funcionalidades

A aplicação permite:

- upload de arquivos em CSV ou Parquet;
- execução automática do pipeline salvo;
- cálculo da probabilidade de alfabetização;
- cálculo da probabilidade de não alfabetização;
- geração do score de risco;
- classificação em Prioritário ou Monitoramento;
- identificação de registros de alto risco;
- visualização de indicadores gerais;
- análise da distribuição dos scores;
- filtragem por região;
- filtragem dinâmica por município;
- análise agregada por região;
- análise agregada por município;
- visualização dos resultados individuais;
- comparação com a situação real quando o target está disponível;
- download dos resultados processados.

A filtragem funciona de forma hierárquica:

```text
Todos os registros
       ↓
Região
       ↓
Município
       ↓
Alunos
```

Ao selecionar uma região ou município, os indicadores, gráficos e tabelas são recalculados automaticamente para o subconjunto escolhido.

> A aplicação utiliza o modelo como ferramenta de triagem e inteligência analítica. As classificações não representam diagnósticos pedagógicos individuais.

---

# Pipeline final

As etapas utilizadas para inferência foram agrupadas em um único `Pipeline`:

```text
Dados originais
      ↓
Pré-processamento
      ↓
ColumnTransformer
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

Dessa forma, novas bases podem ser enviadas diretamente ao pipeline sem necessidade de executar manualmente as etapas de preparação utilizadas durante o treinamento.

A lógica de inferência utilizada pela aplicação está centralizada em:

```text
src/predict.py
```

enquanto a interface e as visualizações estão implementadas em:

```text
src/app.py
```

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
- Streamlit
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

Atualize o `pip` e instale as dependências:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# Execução

## Notebooks

A ordem recomendada é:

```text
00 → Preparação do dataset analítico
01 → Enriquecimento com dados externos
02 → Análise Exploratória
03 → Modelagem de Machine Learning
```

## Aplicação Streamlit

Na raiz do projeto:

```bash
streamlit run src/app.py
```

Após a inicialização, a aplicação será disponibilizada localmente em:

```text
http://localhost:8501
```

---

# Arquivo de entrada da aplicação

A aplicação aceita:

```text
.csv
.parquet
```

Para executar a inferência, o arquivo deve conter as features esperadas pelo pipeline:

```text
ano
rede
sigla_uf
regiao
pib_per_capita
populacao_2022
area_km2
densidade_demografica
qtd_escolas_censo
prop_escolas_rurais
prop_escolas_agua_potavel
prop_escolas_esgoto_rede
prop_escolas_biblioteca_leitura
prop_escolas_lab_informatica
prop_escolas_internet
prop_escolas_internet_aprendizagem
prop_escolas_banda_larga
prop_escolas_rampas_acessibilidade
media_alunos_turma_fund_ai
media_alunos_docente_fund_ai
media_prop_salas_climatizadas
renda_domiciliar_per_capita_mediana
```

Colunas adicionais, como identificadores e nome do município, podem ser mantidas no arquivo e são utilizadas nas análises e visualizações da aplicação.

Caso a coluna `alfabetizado` esteja disponível, a aplicação também calcula métricas de desempenho para o lote enviado.

---

# Limitações

Algumas limitações devem ser consideradas:

- o ROC-AUC indica capacidade discriminatória moderada;
- muitas features representam características territoriais e contextuais;
- o modelo possui poucas informações individuais diretamente relacionadas ao processo de aprendizagem;
- o threshold de `0.52` aumenta o Recall da classe não alfabetizada, reduzindo o desempenho da classe alfabetizada;
- alunos inseridos em contextos semelhantes podem receber probabilidades próximas;
- municípios e regiões são analisados apenas com base nos registros disponíveis na base enviada;
- os resultados territoriais não representam taxas oficiais de alfabetização;
- Feature Importance e SHAP não estabelecem causalidade;
- o modelo deve ser utilizado como ferramenta de triagem e não como diagnóstico pedagógico definitivo.

---

# Próximos passos

Possíveis evoluções incluem:

- inclusão de novas variáveis socioeconômicas e educacionais;
- incorporação de características mais individualizadas dos alunos;
- validação automatizada dos arquivos enviados à aplicação;
- explicações SHAP individuais dentro do Streamlit;
- monitoramento de drift;
- rastreamento de experimentos com MLflow;
- criação de API com FastAPI;
- integração com banco de dados;
- deploy da aplicação em cloud;
- pipeline CI/CD;
- monitoramento contínuo do modelo.

---

# Conclusão

O projeto implementa um fluxo completo de Ciência de Dados aplicado ao contexto da alfabetização, abrangendo desde a preparação e enriquecimento dos dados até a disponibilização de uma aplicação interativa para consumo das previsões.

O **Random Forest otimizado** foi selecionado como modelo final por apresentar o melhor equilíbrio geral entre as soluções avaliadas.

Com o threshold operacional de **0,52**, o modelo alcançou aproximadamente **78% de Recall para alunos não alfabetizados no conjunto de teste**, alinhando o comportamento da solução ao objetivo de priorização da classe de maior interesse.

As análises com **Feature Importance, SHAP e Learning Curve** contribuíram para avaliar a interpretabilidade, estabilidade e comportamento do modelo.

As probabilidades foram transformadas em um **score de risco**, permitindo estruturar a análise em diferentes níveis:

```text
Aluno
  ↓
Município
  ↓
Região
```

Por fim, a aplicação **Streamlit** transforma o pipeline de Machine Learning em uma solução interativa, permitindo enviar novas bases, gerar previsões, explorar os resultados territorialmente e exportar as classificações obtidas.

Dessa forma, o projeto demonstra um fluxo que vai além do treinamento de modelos, conectando **dados, Machine Learning, interpretabilidade e aplicação prática para apoio à tomada de decisão educacional**.

---

## Autor

Desenvolvido por **Renan Trevelim**.

Projeto desenvolvido para o **Tech Challenge — Fase 3**, com foco em Ciência de Dados, Machine Learning, interpretabilidade e inteligência analítica aplicada à alfabetização no Brasil.