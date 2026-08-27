# Predição e Inteligência Analítica para Alfabetização no Brasil

## Tech Challenge — Fase 3

Projeto desenvolvido com o objetivo de aplicar técnicas de **Ciência de Dados e Machine Learning** sobre a camada Gold construída no Tech Challenge da Fase 2, buscando identificar padrões associados à alfabetização e construir um modelo capaz de apoiar a identificação de alunos potencialmente não alfabetizados.

A solução combina dados educacionais, territoriais, populacionais, econômicos e socioeconômicos, além de informações complementares do Censo Escolar, permitindo analisar a alfabetização sob uma perspectiva multidimensional.

---

## 1. Contexto do problema

A alfabetização na infância representa uma etapa fundamental para o desenvolvimento educacional.

No contexto do **Compromisso Nacional Criança Alfabetizada**, o Indicador Criança Alfabetizada permite acompanhar o percentual de estudantes que atingem o nível esperado de alfabetização ao final do 2º ano do Ensino Fundamental.

Entretanto, o desempenho educacional pode estar associado a diferentes características territoriais, socioeconômicas, populacionais e educacionais.

Este projeto utiliza os dados integrados na camada Gold da Fase 2 para investigar essas relações e desenvolver uma solução de Machine Learning voltada à identificação de alunos potencialmente não alfabetizados.

---

## 2. Objetivo

O principal objetivo é desenvolver um modelo de classificação supervisionada capaz de prever se um aluno pertence à classe:

- **0 — Não alfabetizado**
- **1 — Alfabetizado**

O projeto possui foco especial na identificação da classe **Não alfabetizado**, tratando o modelo como uma ferramenta de **triagem e priorização de alunos que podem demandar acompanhamento pedagógico adicional**.

Além da modelagem preditiva, o projeto busca investigar:

- quais características apresentam maior associação com a alfabetização;
- diferenças entre regiões, estados e municípios;
- variáveis territoriais, socioeconômicas e educacionais relevantes;
- fatores mais importantes para as previsões do modelo;
- capacidade de identificação de alunos potencialmente não alfabetizados.

---

## 3. Base de dados

A base analítica foi construída a partir da camada **Gold** desenvolvida durante o Tech Challenge da Fase 2.

Inicialmente, foram integrados os dados individuais dos alunos com informações municipais, preservando a granularidade necessária para o problema de classificação.

Posteriormente, a base foi enriquecida com fontes externas oficiais.

### Principais grupos de informações

#### Educacionais

- situação de alfabetização;
- rede de ensino;
- presença;
- preenchimento da avaliação;
- informações relacionadas ao caderno;
- indicadores agregados do Censo Escolar.

#### Territoriais

- município;
- Unidade Federativa;
- região;
- área territorial;
- densidade demográfica.

#### Populacionais

- população municipal.

#### Econômicas e socioeconômicas

- PIB per capita;
- rendimento domiciliar per capita mediano.

#### Infraestrutura educacional

- quantidade de escolas;
- proporção de escolas rurais;
- acesso à água potável;
- esgotamento sanitário;
- biblioteca ou sala de leitura;
- laboratório de informática;
- acesso à internet;
- internet para aprendizagem;
- banda larga;
- rampas de acessibilidade;
- proporção de salas climatizadas;
- média de alunos por turma;
- média de alunos por docente.

#### Temporais

- ano da avaliação.

A base enriquecida possui aproximadamente **57 mil registros individuais de alunos**.

---

## 4. Fontes externas utilizadas

O enriquecimento da base foi realizado utilizando fontes oficiais.

### IBGE

Foram incorporadas informações municipais relacionadas a:

- PIB per capita;
- população;
- área territorial;
- densidade demográfica;
- rendimento domiciliar per capita mediano.

Os dados populacionais, territoriais e socioeconômicos foram obtidos principalmente a partir do **Censo Demográfico 2022** e de bases municipais do IBGE.

### INEP — Censo Escolar

Os Microdados do Censo Escolar de 2023 e 2024 foram utilizados para construir indicadores agregados de infraestrutura e organização educacional.

Como o identificador de escola presente na base original não apresentou correspondência direta com o código oficial do Censo Escolar, os dados foram agregados por:

```text
município + ano + rede de ensino
```

Essa estratégia permitiu incorporar contexto educacional sem provocar perda de registros na base individual dos alunos.

> Os indicadores derivados do Censo Escolar representam o contexto agregado do município, ano e rede de ensino, não necessariamente a escola específica frequentada por cada aluno.

---

## 5. Estrutura do projeto

```text
tech-challenge-fase3-literacy-ml/
│
├── data/
│   ├── external/
│   ├── gold/
│   └── silver/
│
├── notebooks/
│   ├── 00_preparacao_dataset_analitico.ipynb
│   ├── 01_enriquecimento_dados_externos.ipynb
│   ├── 02_analise_exploratoria.ipynb
│   └── 03_modelos_machine_learning.ipynb
│
├── models/
│
├── reports/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 6. Organização dos notebooks

### `00_preparacao_dataset_analitico.ipynb`

Responsável pela construção inicial da base analítica.

Principais etapas:

- leitura das bases Gold provenientes da Fase 2;
- integração dos dados individuais dos alunos com informações municipais;
- validação das chaves de relacionamento;
- verificação de duplicidades;
- validação da variável alvo;
- geração do dataset analítico inicial.

---

### `01_enriquecimento_dados_externos.ipynb`

Responsável pela ampliação da base com informações externas.

Principais etapas:

- criação da variável de região;
- integração do PIB municipal;
- integração de população, área e densidade demográfica;
- processamento dos Microdados do Censo Escolar;
- criação de indicadores agregados de infraestrutura educacional;
- integração do rendimento domiciliar per capita mediano;
- validação de cobertura;
- validação de duplicidades;
- geração do dataset Gold enriquecido.

A integração foi realizada utilizando estratégias que preservassem os registros da base original.

---

### `02_analise_exploratoria.ipynb`

Responsável pela Análise Exploratória de Dados — EDA.

Foram avaliados:

- qualidade dos dados;
- valores ausentes;
- duplicidades;
- distribuição da variável alvo;
- distribuições das variáveis numéricas;
- correlação de Spearman;
- associação entre variáveis categóricas e alfabetização;
- Cramér's V;
- diferenças entre regiões e Unidades Federativas;
- padrões municipais;
- investigação do comportamento do caderno 12;
- identificação de potenciais fontes de data leakage;
- hipóteses derivadas da análise;
- definição das variáveis utilizadas na modelagem.

A análise exploratória mostrou forte heterogeneidade territorial e educacional, enquanto as variáveis numéricas apresentaram, isoladamente, associações relativamente baixas com o target.

---

### `03_modelos_machine_learning.ipynb`

Responsável pela construção, comparação, otimização e interpretação dos modelos.

Principais etapas:

- separação entre treino, validação e teste;
- pré-processamento numérico e categórico;
- imputação;
- padronização;
- One-Hot Encoding;
- seleção de features por variância;
- treinamento de modelos baseline;
- otimização de hiperparâmetros;
- comparação entre algoritmos;
- construção de ensembles;
- validação cruzada;
- análise de threshold;
- avaliação final no conjunto de teste;
- Feature Importance;
- interpretabilidade com SHAP.

---

## 7. Análise Exploratória de Dados

A etapa de EDA teve como objetivo compreender a estrutura da base, identificar padrões relevantes e definir decisões para a etapa de modelagem.

### Variáveis numéricas

As distribuições mostraram forte heterogeneidade entre municípios, principalmente em variáveis como:

- PIB per capita;
- população;
- área territorial;
- densidade demográfica;
- quantidade de escolas.

As correlações individuais com a alfabetização apresentaram baixa magnitude, reforçando o caráter multifatorial do problema.

### Variáveis categóricas

Foram identificadas diferenças relevantes entre:

- regiões;
- Unidades Federativas;
- redes de ensino;
- categorias relacionadas ao processo de aplicação da avaliação.

As variáveis `presenca` e `preenchimento_caderno` apresentaram associação elevada com o target, indicando potencial risco de vazamento de informação.

### Análise territorial

A análise por região, Unidade Federativa e município revelou forte heterogeneidade territorial.

Os resultados municipais foram interpretados como padrões observados na amostra, e não como taxas oficiais de alfabetização.

---

## 8. Investigação do caderno 12

Durante o EDA, o `caderno 12` apresentou taxa de alfabetização significativamente inferior às demais categorias.

A investigação mostrou que aproximadamente:

- **79,6% dos alunos estavam ausentes**;
- **79,8% não possuíam prova preenchida**.

Esse comportamento mostrou que a baixa taxa de alfabetização do caderno 12 estava fortemente associada às condições de aplicação da avaliação.

Por esse motivo, a variável `caderno` foi excluída do conjunto final de preditores.

---

## 9. Prevenção de Data Leakage

Durante o EDA foram identificadas variáveis com risco de vazamento de informação.

As principais foram:

- `presenca`;
- `preenchimento_caderno`;
- `proficiencia`;
- `caderno`.

`presenca` e `preenchimento_caderno` apresentaram forte associação com o target e taxa de alfabetização igual a zero entre alunos ausentes ou com prova não preenchida.

A variável `proficiencia` possui proximidade conceitual direta com a definição da situação de alfabetização.

A exclusão dessas variáveis busca garantir que o modelo utilize informações adequadas para um cenário real de previsão.

---

## 10. Variáveis removidas da modelagem

Além das variáveis com potencial data leakage, também foram removidas:

- `id_aluno`;
- `id_escola`;
- `id_municipio`;
- `id_municipio_nome`;
- `serie`;
- `peso_aluno`;
- `meta_municipio_ano`;
- `meta_uf_ano`;
- `percentual_participacao`.

Os identificadores foram retirados por não representarem atributos explicativos adequados.

As variáveis `meta_municipio_ano` e `meta_uf_ano` apresentaram aproximadamente **57% de valores ausentes**, o que exigiria elevado nível de imputação.

A variável `serie` apresentou apenas uma categoria na base.

---

## 11. Variável alvo

A variável utilizada como target do problema de classificação foi:

```text
alfabetizado
```

Representação utilizada:

```text
0 = Não alfabetizado
1 = Alfabetizado
```

---

## 12. Pipeline de Machine Learning

O fluxo geral da modelagem pode ser representado por:

```text
Gold Enriquecido
      │
      ▼
Seleção de Features
      │
      ▼
Treino / Validação / Teste
      │
      ▼
Pré-processamento
      │
      ├── Variáveis Numéricas
      │     ├── Imputação pela mediana
      │     └── StandardScaler
      │
      └── Variáveis Categóricas
            ├── Imputação pela moda
            └── One-Hot Encoding
      │
      ▼
VarianceThreshold
      │
      ▼
Modelos de Machine Learning
      │
      ├── Regressão Logística
      ├── Árvore de Decisão
      ├── KNN
      ├── Random Forest
      └── XGBoost
      │
      ▼
Otimização de Hiperparâmetros
      │
      ▼
Voting / Stacking
      │
      ▼
Seleção do Modelo Final
      │
      ▼
Ajuste de Threshold
      │
      ▼
Teste Final
      │
      ▼
Interpretabilidade com SHAP
```

---

## 13. Pré-processamento

O pré-processamento foi implementado utilizando ferramentas do Scikit-learn.

### Variáveis numéricas

Foram aplicados:

- `SimpleImputer(strategy="median")`;
- `StandardScaler()`.

### Variáveis categóricas

Foram aplicados:

- `SimpleImputer(strategy="most_frequent")`;
- `OneHotEncoder()`.

### Seleção por variância

Após o pré-processamento, foi utilizado:

```python
VarianceThreshold(threshold=0.05)
```

O objetivo foi remover features com baixa variabilidade, principalmente categorias muito raras após o One-Hot Encoding.

---

## 14. Divisão dos dados

Os dados foram separados em três subconjuntos:

```text
Treino
Validação
Teste
```

O conjunto de treino foi utilizado para treinamento e otimização.

O conjunto de validação foi utilizado para:

- comparação dos modelos;
- seleção do modelo final;
- avaliação de diferentes thresholds.

O conjunto de teste permaneceu isolado até a avaliação final.

---

## 15. Modelos avaliados

Durante o projeto foram avaliados diferentes algoritmos:

- Logistic Regression;
- Decision Tree;
- K-Nearest Neighbors;
- Random Forest;
- XGBoost;
- Voting Classifier;
- Stacking Classifier.

Os modelos foram comparados utilizando métricas como:

- Accuracy;
- Precision;
- Recall da classe 0;
- Recall da classe 1;
- F1-score por classe;
- F1-Macro;
- Balanced Accuracy;
- ROC-AUC.

Como o principal objetivo do projeto é identificar alunos potencialmente não alfabetizados, foi dada atenção especial ao:

```text
Recall da classe 0
```

---

## 16. Otimização de hiperparâmetros

Os principais modelos foram submetidos a otimização de hiperparâmetros utilizando `RandomizedSearchCV`.

Foram avaliados principalmente:

- Random Forest;
- XGBoost;
- Regressão Logística.

A otimização permitiu comparar diferentes configurações dos modelos e analisar o equilíbrio entre as duas classes.

---

## 17. Ensembles

Também foram avaliadas estratégias de ensemble.

### Voting Classifier

Foi implementada uma combinação entre:

```text
Random Forest + XGBoost
```

utilizando votação por probabilidades.

### Stacking Classifier

Foram utilizados:

```text
Modelos base:
- Random Forest
- XGBoost

Meta-modelo:
- Regressão Logística
```

Os ensembles apresentaram desempenho competitivo, porém não superaram de forma significativa o Random Forest em equilíbrio geral e capacidade discriminatória.

---

## 18. Comparação dos modelos

Após a otimização, os principais resultados no conjunto de validação foram aproximadamente:

| Modelo | Accuracy | Recall 0 | Recall 1 | F1 Macro | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.5922 | 0.6394 | 0.5428 | 0.5906 | 0.6337 |
| Stacking | 0.5902 | 0.6199 | 0.5592 | 0.5895 | 0.6323 |
| Voting | 0.5873 | 0.6831 | 0.4873 | 0.5822 | 0.6300 |
| XGBoost | 0.5792 | 0.7264 | 0.4255 | 0.5677 | 0.6199 |
| Regressão Logística | 0.5221 | 0.8723 | 0.1567 | 0.4469 | 0.5562 |

O Random Forest apresentou o melhor equilíbrio geral entre as classes.

---

## 19. Modelo selecionado

O modelo selecionado foi o:

```text
Random Forest Tunado
```

A escolha considerou:

- maior F1-Macro entre os principais candidatos;
- maior ROC-AUC;
- melhor equilíbrio entre Recall das duas classes;
- menor complexidade em comparação com ensembles;
- melhor interpretabilidade.

---

## 20. Validação cruzada

Após a seleção do Random Forest, foi realizada validação cruzada estratificada com 5 folds.

Os dados de treino e validação foram utilizados como conjunto de desenvolvimento.

O conjunto de teste permaneceu isolado.

Resultados médios:

| Métrica | Média | Desvio padrão |
|---|---:|---:|
| Accuracy | 0.5901 | 0.0022 |
| Recall classe 0 | 0.6514 | 0.0127 |
| Recall classe 1 | 0.5261 | 0.0147 |
| F1 classe 0 | 0.6187 | 0.0046 |
| F1 classe 1 | 0.5567 | 0.0075 |
| F1 Macro | 0.5877 | 0.0026 |
| ROC-AUC | 0.6290 | 0.0043 |

Os baixos desvios observados indicam desempenho relativamente estável entre diferentes partições dos dados.

---

## 21. Ajuste do threshold

O threshold padrão de classificação é:

```text
0.50
```

Como o objetivo principal do projeto é aumentar a identificação dos alunos potencialmente não alfabetizados, diferentes thresholds foram avaliados utilizando o conjunto de validação.

O threshold operacional escolhido foi:

```text
0.52
```

Na validação, o Recall da classe 0 aumentou aproximadamente de:

```text
63,9% para 78,3%
```

Esse aumento ocorreu acompanhado de redução no Recall da classe alfabetizada.

A decisão representa um trade-off intencional, considerando o modelo como uma ferramenta de triagem.

---

## 22. Resultado final no conjunto de teste

Após a seleção do modelo e do threshold utilizando apenas os dados de desenvolvimento, o conjunto de teste foi utilizado para avaliação final.

Com o Random Forest e threshold `0.52`, foram obtidos aproximadamente:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.58 |
| Recall — Não alfabetizado | **0.78** |
| Recall — Alfabetizado | 0.37 |
| F1 — Não alfabetizado | 0.66 |
| F1 — Alfabetizado | 0.46 |
| ROC-AUC | **0.62** |

No conjunto de teste:

```text
Alunos não alfabetizados: 5.894
Identificados corretamente: 4.616
```

Assim, aproximadamente **78% dos alunos não alfabetizados foram identificados pelo modelo**.

Esse resultado é compatível com a proposta de utilização da solução como ferramenta de triagem.

---

## 23. Matriz de confusão final

A matriz de confusão do modelo final apresentou:

```text
                    Predito
                 Não       Sim

Real Não        4616      1278
Real Sim        3573      2076
```

O modelo foi configurado para aumentar a sensibilidade para alunos não alfabetizados, assumindo maior quantidade de falsos alertas na classe alfabetizada.

---

## 24. Curva ROC

A curva ROC do modelo final foi calculada utilizando as probabilidades produzidas pelo Random Forest.

O modelo apresentou:

```text
ROC-AUC ≈ 0.62
```

Esse valor indica capacidade discriminatória moderada.

---

## 25. Importância das variáveis

A importância interna do Random Forest mostrou forte participação de variáveis territoriais e educacionais.

Entre as principais features apareceram:

- `sigla_uf_CE`;
- `sigla_uf_BA`;
- `qtd_escolas_censo`;
- `prop_escolas_internet_aprendizagem`;
- `prop_escolas_internet`;
- `densidade_demografica`;
- `prop_escolas_biblioteca_leitura`;
- `prop_escolas_agua_potavel`;
- `populacao_2022`;
- `regiao_Norte`.

Essas importâncias representam contribuição preditiva para o modelo e não devem ser interpretadas como relações causais.

---

## 26. Interpretabilidade com SHAP

A biblioteca **SHAP — SHapley Additive exPlanations** foi utilizada para aprofundar a interpretação do Random Forest.

Como a principal classe de interesse do projeto é:

```text
0 = Não alfabetizado
```

as análises SHAP foram direcionadas principalmente para essa classe.

Foram utilizados os seguintes gráficos:

### Summary Plot — Bar

Utilizado para identificar as features de maior importância global.

### Summary Plot — Beeswarm

Utilizado para analisar:

- magnitude do impacto;
- direção da contribuição;
- distribuição dos valores das features.

### Dependence Plot

Utilizado para compreender como diferentes valores de uma feature afetam sua contribuição para a previsão.

### Waterfall Plot

Utilizado para explicar individualmente a previsão de um aluno, mostrando quais variáveis contribuíram positiva ou negativamente para a saída do modelo.

---

## 27. Principais insights do projeto

### Heterogeneidade territorial

Foram observadas diferenças relevantes entre:

- regiões;
- Unidades Federativas;
- municípios.

A variável territorial apresentou forte relevância tanto no EDA quanto na modelagem.

### Infraestrutura educacional

Indicadores relacionados a:

- internet;
- internet para aprendizagem;
- água potável;
- biblioteca;
- acessibilidade;

apresentaram associação com alfabetização e participação relevante no modelo.

### Contexto socioeconômico

A base foi enriquecida com:

```text
pib_per_capita
renda_domiciliar_per_capita_mediana
```

permitindo representar tanto a dimensão econômica municipal quanto aspectos relacionados à renda da população.

### Natureza multifatorial

Nenhuma variável isolada apresentou forte capacidade de explicar a alfabetização.

Os resultados indicam que o fenômeno depende da combinação de diferentes características:

- territoriais;
- socioeconômicas;
- populacionais;
- educacionais.

---

## 28. Análise municipal

Também foram analisados padrões de alfabetização em nível municipal.

Como muitos municípios possuíam poucas observações, foi adotado um número mínimo de registros para reduzir interpretações instáveis.

A análise revelou elevada heterogeneidade entre municípios.

Esses resultados representam padrões observados na amostra e não devem ser interpretados como taxas oficiais municipais.

---

## 29. Tecnologias utilizadas

O projeto foi desenvolvido utilizando principalmente:

- Python;
- Pandas;
- NumPy;
- Scikit-learn;
- XGBoost;
- SHAP;
- SciPy;
- Matplotlib;
- Seaborn;
- Jupyter Notebook;
- Parquet.

---

## 30. Justificativa das principais tecnologias

### Pandas e NumPy

Utilizados para:

- manipulação;
- limpeza;
- integração;
- transformação;
- preparação dos dados.

### Scikit-learn

Utilizado para:

- pré-processamento;
- pipelines;
- modelos;
- validação cruzada;
- otimização;
- seleção de features;
- métricas.

### XGBoost

Utilizado como modelo baseado em boosting para comparação com algoritmos tradicionais.

### SHAP

Utilizado para interpretabilidade global e local.

### Parquet

Utilizado como formato intermediário por oferecer:

- armazenamento colunar;
- compressão;
- eficiência de leitura;
- preservação adequada dos tipos de dados.

---

## 31. Como reproduzir o projeto

### 1. Clone o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd tech-challenge-fase3-literacy-ml
```

### 2. Crie um ambiente virtual

```bash
python -m venv .venv
```

### 3. Ative o ambiente

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux/macOS

```bash
source .venv/bin/activate
```

### 4. Instale as dependências

```bash
pip install -r requirements.txt
```

### 5. Execute os notebooks na ordem

```text
00_preparacao_dataset_analitico.ipynb
        ↓
01_enriquecimento_dados_externos.ipynb
        ↓
02_analise_exploratoria.ipynb
        ↓
03_modelos_machine_learning.ipynb
```

A execução sequencial garante que cada etapa utilize os datasets produzidos anteriormente.

---

## 32. Limitações

O projeto apresenta algumas limitações.

### Variáveis agregadas

Grande parte dos indicadores de infraestrutura educacional representa características agregadas por:

```text
município + ano + rede
```

e não necessariamente a escola específica de cada aluno.

### Ausência de variáveis individuais adicionais

Informações como:

- renda familiar individual;
- escolaridade dos responsáveis;
- estrutura familiar;
- frequência escolar histórica;
- trajetória acadêmica;

poderiam aumentar a capacidade preditiva.

### Poder discriminatório moderado

O ROC-AUC final ficou próximo de:

```text
0.62
```

indicando capacidade discriminatória moderada.

### Trade-off entre classes

A priorização da identificação dos alunos não alfabetizados aumenta a quantidade de falsos alertas na classe alfabetizada.

### Associação não implica causalidade

Os resultados de:

- correlação;
- Feature Importance;
- SHAP;

representam associações e padrões preditivos.

Não devem ser interpretados como evidência direta de causalidade.

---

## 33. Aplicação prática

O modelo pode ser utilizado como ferramenta complementar para:

- identificação de alunos potencialmente em risco;
- priorização de avaliações pedagógicas;
- direcionamento de reforço escolar;
- identificação de contextos territoriais vulneráveis;
- apoio à análise de políticas educacionais;
- apoio à tomada de decisão baseada em dados.

A previsão deve ser interpretada como um **indicador de risco**, e não como diagnóstico definitivo da situação de alfabetização.

---

## 34. Próximos passos

Possíveis evoluções incluem:

- integração de novos indicadores socioeconômicos;
- inclusão de informações sobre formação docente;
- incorporação de dados de vulnerabilidade social;
- clusterização de municípios com contextos semelhantes;
- inclusão de novos anos;
- calibração de probabilidades;
- monitoramento de drift;
- criação de API para inferência;
- criação de dashboard interativo;
- implantação de pipeline de MLOps;
- monitoramento contínuo da performance do modelo.

---

## 35. Conclusão

O projeto apresenta uma solução completa de Ciência de Dados aplicada a dados públicos educacionais.

A camada Gold construída na Fase 2 foi enriquecida com informações territoriais, populacionais, econômicas, socioeconômicas e educacionais.

A análise exploratória mostrou forte heterogeneidade territorial e indicou que a alfabetização possui natureza multifatorial.

Durante a modelagem foram avaliados diferentes algoritmos e estratégias de ensemble.

O **Random Forest Tunado** apresentou o melhor equilíbrio geral e foi selecionado como modelo final.

Com threshold operacional de:

```text
0.52
```

o modelo foi capaz de identificar aproximadamente:

```text
78% dos alunos não alfabetizados
```

no conjunto de teste.

Esse resultado demonstra potencial de aplicação como ferramenta de triagem e priorização de alunos potencialmente em risco.

A utilização de Feature Importance e SHAP também aumentou a transparência do modelo, permitindo compreender quais características mais influenciam suas previsões.

---

## Autores

Projeto desenvolvido para o **Tech Challenge — Fase 3**.

- Renan
- Adicionar demais integrantes do grupo

---

## Referências

- Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira — INEP
- Censo Escolar da Educação Básica
- Instituto Brasileiro de Geografia e Estatística — IBGE
- Censo Demográfico 2022
- Sistema IBGE de Recuperação Automática — SIDRA
- Base dos Dados
- Compromisso Nacional Criança Alfabetizada