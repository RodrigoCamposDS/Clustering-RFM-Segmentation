# RFM Clustering - Segmentação de Clientes

Este projeto realiza a **segmentação de clientes** utilizando a análise **RFM (Recency, Frequency, Monetary)** combinada com **técnicas de clusterização**, incluindo **Fuzzy C-Means e K-Means**. O objetivo é identificar diferentes perfis de clientes para auxiliar estratégias de marketing e retenção.

## 1. Visão Geral

A análise RFM é uma abordagem amplamente utilizada para **segmentar clientes com base no comportamento de compra**. Os três componentes principais são:

- **Recency (R)**: Tempo desde a última compra.
- **Frequency (F)**: Número de compras realizadas em um período.
- **Monetary (M)**: Valor total gasto pelo cliente.

Após o cálculo das métricas RFM, foram aplicadas técnicas de **redução de dimensionalidade (PCA)** e **clusterização** para identificar grupos com padrões de compra distintos.

---

## 2. Estrutura do Projeto

```
project/
│── notebooks/               # Notebooks do Jupyter com as etapas da análise
│   ├── 01-data-preprocessing.ipynb  # Carregamento e tratamento de dados
│   ├── 02-feature-engineering.ipynb  # Criação das variáveis RFM
│   ├── 03-model-cluster.ipynb  # Aplicação de modelos de clusterização
│   ├── 04-evaluation.ipynb  # Avaliação e visualização dos clusters
│── src/                     # Código-fonte do projeto
│   ├── data/                # Dados brutos e processados
│   ├── models/              # Modelos treinados
│   ├── utils/               # Funções auxiliares
│── requirements.txt         # Dependências do projeto
│── README.md                # Documentação do projeto
```

---

## 3. Tecnologias Utilizadas

- **Python**
- **Pandas, NumPy, PySpark** (Manipulação de Dados)
- **Scikit-learn** (Redução de dimensionalidade e clusterização)
- **Fuzzy C-Means** (Clusterização suave)
- **Plotly, Matplotlib, Seaborn** (Visualização dos clusters)

---

## 4. Como Executar o Projeto

### 1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/Clustering-RFM-Segmentation.git
cd Clustering-RFM-Segmentation
```

### 2. Crie um ambiente virtual e instale as dependências:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. Execute os notebooks na sequência:
- `01-data-preprocessing.ipynb` → Pré-processamento dos dados
- `02-feature-engineering.ipynb` → Criação das métricas RFM
- `03-model-cluster.ipynb` → Aplicação de modelos de clusterização
- `04-evaluation.ipynb` → Avaliação dos clusters gerados

---

## 5. Metodologia

1. **Carregamento e Limpeza dos Dados**  
   - Leitura de arquivos Parquet e remoção de valores inconsistentes.  

2. **Engenharia de Features**  
   - Cálculo das métricas **Recency, Frequency e Monetary**.  
   - Normalização dos dados e aplicação do **PCA** para redução de dimensionalidade.  

3. **Clusterização**  
   - Aplicação de **K-Means** e **Fuzzy C-Means** para segmentação dos clientes.  

4. **Avaliação e Interpretação**  
   - Análise dos clusters formados e identificação de perfis de clientes.  
   - Visualização tridimensional dos clusters em gráficos interativos.  

---

## 6. Resultados Obtidos

Os clientes foram segmentados em **cinco grupos distintos**, baseados nas métricas RFM. A análise visual dos clusters permite interpretar padrões de comportamento dos clientes.

Os **clusters identificados** podem ser usados para ações estratégicas, como:

- **Clientes Premium (Vermelho)**: Alta frequência de compras e alto gasto total.
- **Clientes Regulares (Verde)**: Compram com frequência média, mas gastam moderadamente.
- **Clientes Ocasionais (Roxo)**: Compras esporádicas e baixo gasto.
- **Clientes Inativos (Azul)**: Não realizam compras há muito tempo.
- **Clientes Estáveis (Laranja)**: Compram com consistência, mas sem valores extremos.

### **Visualizações dos Clusters**

#### **Clusters em 3D (Fuzzy Clustering)**
A visualização 3D permite identificar a separação entre os grupos no espaço **Recency, Frequency e Monetary**.

![Clusters 3D](Clusters_3D.png)

#### **Clusters em 2D (Recency vs. Frequency)**
Este gráfico destaca a distribuição dos clientes em um plano **Recency vs. Frequency**, reforçando os padrões de compra.

![Clusters 2D](Clusters_2D.png)

#### **Clusters Baseados em Ângulos**
Aqui, os clusters são visualizados em uma projeção polar, explorando diferenças angulares entre os grupos.

![Clusters Ângulos](Clusters_ang.png)

#### **Mapa de Densidade - Recency vs. Frequency**
A análise de densidade revela a concentração de clientes em determinadas áreas do espaço RFM.

![Densidade](densidade.png)

A segmentação permite **ações personalizadas**, como campanhas de retenção, ofertas exclusivas e estratégias de fidelização. A identificação dos clusters possibilita que as equipes de marketing direcionem estratégias específicas para cada grupo de clientes.

---

## 7. Próximos Passos

- Aplicação de **modelos supervisionados** para prever a migração entre clusters ao longo do tempo.  
- Integração com ferramentas de **marketing automatizado** para criar campanhas direcionadas.  
- Análise de impacto de **ofertas personalizadas** com base nos segmentos RFM.

---

## 8. Contato

Para dúvidas ou sugestões, entre em contato:

- **Nome**: Rodrigo Campos
- **GitHub**: [RodrigoCamposDS](https://github.com/RodrigoCamposDS)
- **LinkedIn**: [linkedin.com/in/seu-perfil](https://linkedin.com/in/seu-perfil)

