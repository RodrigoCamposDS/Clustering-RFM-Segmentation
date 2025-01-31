import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution_hist(df, columns=None, bins=30, color="green", figsize=(15, 5)):
    """
    Plota histogramas para visualizar a distribuição das colunas do DataFrame.

    Parâmetros:
    - df: DataFrame Pandas com os dados transformados.
    - columns: Lista de colunas a serem plotadas (default: todas as colunas do DataFrame).
    - bins: Número de bins no histograma (default: 30).
    - color: Cor dos histogramas (default: "green").
    - figsize: Tamanho da figura (default: (15, 5)).
    """
    
    if columns is None:
        columns = df.columns  # Usa todas as colunas se nenhuma for especificada

    plt.figure(figsize=figsize)

    for i, column in enumerate(columns):
        plt.subplot(1, len(columns), i + 1)
        sns.histplot(df[column], kde=True, bins=bins, color=color)
        plt.title(f"Distribuição de {column}")

    plt.tight_layout()
    plt.show()

# Exemplo de uso da função (aplicando ao DataFrame pandas convertido do PySpark)






import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql.functions import col, expr

def plot_box_and_bar_spark(df, col1, col2):
    """
    Função otimizada para plotar boxplots e gráfico de barras para duas colunas contínuas de um DataFrame PySpark.

    Otimizações:
    - Evita `.collect()` desnecessário para grandes datasets.
    - Usa `.describe()` para cálculos estatísticos básicos.
    - Utiliza `.toPandas()` em pequenos DataFrames para performance.
    - Usa `cache()` para evitar recomputação em DataFrames grandes.

    Parâmetros:
    - df: DataFrame PySpark
    - col1: Nome da primeira coluna (str)
    - col2: Nome da segunda coluna (str)
    """

    #  Cachear o DataFrame para evitar recomputação
    df = df.select(col1, col2).cache()

    #  Calcular estatísticas básicas com .describe()
    stats_df = df.describe([col1, col2]).toPandas()

    # Renomear para facilitar leitura
    stats_df.columns = ["Metric"] + [col1, col2]
    stats_df = stats_df[stats_df["Metric"].isin(["mean", "50%"])]
    stats_df["Metric"] = stats_df["Metric"].replace({"mean": "Média", "50%": "Mediana"})

    #  Coletar os valores para o boxplot sem usar RDD
    values_pd = df.toPandas()

    # Criar figura com subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    #  Subplot 1: Boxplots (evita `flatMap`)
    axs[0].boxplot([values_pd[col1], values_pd[col2]], vert=False, patch_artist=True,
                   labels=[col1, col2],
                   boxprops=dict(facecolor="skyblue", color="black"))
    axs[0].set_title(f"Boxplots de {col1} e {col2}")
    axs[0].set_xlabel("Valores")

    #  Subplot 2: Gráfico de Barras
    stats_df.set_index("Metric").astype(float).plot(kind="bar", ax=axs[1], color=["skyblue", "orange"], edgecolor="black")
    axs[1].set_title(f"Comparação de Métricas: {col1} vs {col2}")
    axs[1].set_ylabel("Valores")
    axs[1].set_xlabel("Métricas")
    axs[1].legend(title="Variável")

    plt.tight_layout()
    plt.show()

    #  Remover cache para liberar memória
    df.unpersist()

    # --------------------------------------------------------------------------------------


# 2. Gráfico de Dispersão (Recency vs Frequency)


def plot_scatter(pandas_df, prediction, feature_01, feature_02, model_name):
    import matplotlib.pyplot as plt

    colors = ['red', 'blue', 'green', 'purple']
    plt.figure(figsize=(15, 6))
    for cluster in pandas_df[prediction].unique():
        cluster_data = pandas_df[pandas_df[prediction] == cluster]
        plt.scatter(
            cluster_data[feature_01],
            cluster_data[feature_02],
            s=50,
            alpha=0.6,
            label=f'Cluster {cluster}',
            color=colors[cluster % len(colors)]
        )
    plt.title(f"Visualização dos Clusters {feature_01} vs {feature_02} - {model_name}")
    plt.xlabel(feature_01)
    plt.ylabel(feature_02)
    plt.legend()
    plt.grid(True)
    plt.show()


# 3. Gráfico Polar (Baseado em Ângulos)

import numpy as np
import pandas as pd
import plotly.express as px

def plot_polar_clusters(pandas_df, prediction, dim1, dim2, model_name):
    """
    Plota clusters em um gráfico polar usando Plotly.

    Parâmetros:
    - pandas_df: DataFrame contendo os dados.
    - prediction: Nome da coluna contendo os rótulos dos clusters.
    - dim1, dim2: Nomes das colunas que representam as dimensões usadas na clusterização.
    - model_name: Nome do modelo (usado no título do gráfico).
    """

    # Verifique se as colunas de features existem e crie se necessário
    if "Dim1" not in pandas_df.columns or "Dim2" not in pandas_df.columns:
        pandas_df["Dim1"] = pandas_df[dim1]
        pandas_df["Dim2"] = pandas_df[dim2]

    # Calcular ângulo e magnitude com as colunas adequadas
    pandas_df['angle'] = np.degrees(np.arctan2(pandas_df['Dim2'], pandas_df['Dim1']))  # Convertendo para graus
    pandas_df['magnitude'] = np.sqrt(pandas_df['Dim1']**2 + pandas_df['Dim2']**2)  # Raio (distância da origem)

    # Criar o gráfico polar no Plotly
    fig = px.scatter_polar(
        pandas_df, 
        r="magnitude", 
        theta="angle", 
        color=prediction,
        title=f"Clusters Baseados em Ângulos - {model_name}",
        color_discrete_sequence=px.colors.qualitative.Set1
    )

    # Ajustando layout para melhorar a visualização
    fig.update_layout(
            width=2000,  # Largura do gráfico (padrão: 700)
            height=800,
        polar=dict(
            angularaxis=dict(direction="counterclockwise", showline=False),
            radialaxis=dict(showgrid=True)
        ),
        legend_title="Clusters"
    )

    fig.show()

# 4. Gráfico de Barras (Distribuição de Clusters)


def plot_cluster_distribution(labels,  model_name):
    import numpy as np
    import matplotlib.pyplot as plt

    unique_labels, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(10, 6))
    plt.bar(unique_labels, counts, color="skyblue")
    plt.title(f"Distribuição de Tamanhos dos Clusters - {model_name}")
    plt.xlabel("Cluster")
    plt.ylabel("Número de Pontos")
    plt.grid(axis='y')
    plt.show()

# 5. Gráfico 3D


def plot_3d_clusters(pandas_df, prediction, feature_01, feature_02, feature_03, model_name):
    import plotly.graph_objects as go

    fig = go.Figure()
    for cluster in pandas_df[prediction].unique():
        cluster_data = pandas_df[pandas_df[prediction] == cluster]
        fig.add_trace(go.Scatter3d(
            x=cluster_data[feature_01],
            y=cluster_data[feature_02],
            z=cluster_data[feature_03],
            mode='markers',
            marker=dict(size=6),
            name=f' {cluster}'
        ))
    fig.update_layout(
        title=f"Clusters em 3D ({model_name})",
        scene=dict(
            xaxis=dict(title=feature_01),
            yaxis=dict(title=feature_02),
            zaxis=dict(title=feature_03)
        ),
        showlegend=True,
        width=2000,
        height=800
    )
    fig.show()

# 6. Heatmap de Similaridade entre Clusters

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql import Window

def plot_heatmap_similarity(spark_df, label_col, model_name):
    """
    Gera um heatmap de similaridade entre clusters utilizando PySpark.
    
    Parâmetros:
    - spark_df: DataFrame PySpark contendo as labels dos clusters.
    - label_col: Nome da coluna com as labels dos clusters.
    - model_name: Nome do modelo para o título do gráfico.
    """
    # Contar a quantidade de elementos em cada cluster
    cluster_counts = (
        spark_df.groupBy(label_col)
        .agg(F.count("*").alias("count"))
        .orderBy(label_col)
    )
    
    # Obter os resultados como uma lista para criar a matriz de similaridade
    cluster_counts_list = cluster_counts.collect()
    labels = [row[label_col] for row in cluster_counts_list]
    counts = [row["count"] for row in cluster_counts_list]
    
    # Criar a matriz de similaridade
    similarity_matrix = np.zeros((len(labels), len(labels)))
    for i, count in enumerate(counts):
        similarity_matrix[i, i] = count  # Autossimilaridade na diagonal
    
    # Plotar o heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        similarity_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".0f",
        square=True,
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title(f"Heatmap de Similaridade entre Clusters - {model_name}")
    plt.xlabel("Clusters")
    plt.ylabel("Clusters")
    plt.show()





# 7. Mapa de Densidade (Frequency vs Monetary)


import plotly.graph_objects as go
import numpy as np

def plot_density_map(pandas_df, x_feature, y_feature, prediction, sample_fraction=0.1):
    """
    Plota um gráfico de densidade 2D (contour plot) interativo usando Plotly, sem histogramas laterais.

    Parâmetros:
    - pandas_df: DataFrame contendo os dados.
    - x_feature: Nome da variável para o eixo X.
    - y_feature: Nome da variável para o eixo Y.
    - prediction: Nome da coluna com os clusters (usado como rótulo).
    - sample_fraction: Fração da amostra a ser usada para reduzir o volume de dados.
    """

    # Reduzindo o dataset para melhorar performance
    sample_df = pandas_df.sample(frac=sample_fraction, random_state=42)

    # Criando o gráfico de densidade sem histogramas
    fig = go.Figure()

    # Adicionando o mapa de densidade
    fig.add_trace(go.Histogram2dContour(
        x=sample_df[x_feature], 
        y=sample_df[y_feature], 
        colorscale="Blues",
        contours=dict(showlabels=False),  # Remove os números das isolinhas
        showscale=True,  # Mostrar escala de cores
    ))

    # Adicionando os pontos individuais
    fig.add_trace(go.Scatter(
        x=sample_df[x_feature], 
        y=sample_df[y_feature], 
        mode="markers", 
        marker=dict(size=3, color="black", opacity=0.5),
        name="Pontos"
    ))

    # Ajustando layout
    fig.update_layout(
        title=f"Densidade de {x_feature} vs {y_feature} - {prediction}",
        xaxis_title=x_feature,
        yaxis_title=y_feature,
        width=2000,
        height=800
    )

    fig.show()



# 8. Boxplot e Violinplot

import plotly.express as px

def plot_box_prediction(pandas_df, prediction, feature, model_name):
    """
    Plota um boxplot interativo usando Plotly.

    Parâmetros:
    - pandas_df: DataFrame contendo os dados.
    - prediction: Nome da coluna contendo os rótulos dos clusters.
    - feature: Nome da variável numérica a ser plotada no eixo Y.
    - model_name: Nome do modelo (usado no título do gráfico).
    """

    # Criar o gráfico de boxplot no Plotly
    fig = px.box(
        pandas_df, 
        x=prediction, 
        y=feature, 
        color=prediction, 
        points="all",  # Mostra todos os pontos individuais
        hover_data=pandas_df.columns,  # Exibe detalhes ao passar o mouse
        title=f"Boxplot dos Clusters ({feature} por Cluster) - {model_name}",
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    # Ajustar layout para melhorar a visualização
    fig.update_layout(
        xaxis_title="Clusters",
        yaxis_title=feature,
        width=2000,  # Largura do gráfico
        height=800,  # Altura do gráfico
        legend_title="Clusters"
    )

    fig.show()


import plotly.express as px

def plot_violin_prediction(pandas_df, prediction, feature, model_name):
    """
    Plota um gráfico de violino interativo usando Plotly.

    Parâmetros:
    - pandas_df: DataFrame contendo os dados.
    - prediction: Nome da coluna contendo os rótulos dos clusters.
    - feature: Nome da variável numérica a ser plotada no eixo Y.
    - model_name: Nome do modelo (usado no título do gráfico).
    """

    # Criar o gráfico de violino no Plotly
    fig = px.violin(
        pandas_df, 
        x=prediction, 
        y=feature, 
        color=prediction, 
        box=True,  # Adiciona boxplot dentro do violin
        points="all",  # Mostra todos os pontos individuais
        hover_data=pandas_df.columns,  # Permite visualizar mais informações ao passar o mouse
        title=f"Violinplot dos Clusters ({feature} por Cluster) - {model_name}",
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    # Ajustar layout para melhorar visualização
    fig.update_layout(
        xaxis_title="Clusters",
        yaxis_title=feature,
        width=2000,  # Ajusta largura do gráfico
        height=800,  # Ajusta altura do gráfico
        legend_title="Clusters"
    )

    fig.show()


# 9. Métricas de Avaliação por Cluster

import matplotlib.pyplot as plt

def plot_cluster_metrics(clusters, silhouette_scores, davies_bouldin_scores, model_name):
    # Silhouette Score
    plt.figure(figsize=(10, 6))
    plt.bar(clusters, silhouette_scores, color='skyblue', label="Silhouette Score")
    plt.title(f"Silhouette Score por Cluster - {model_name}")
    plt.xlabel("Cluster")
    plt.ylabel("Silhouette Score")
    plt.grid(axis='y')
    plt.legend()
    plt.tight_layout()  # Ajusta o layout automaticamente
    plt.show()

    # Davies-Bouldin Score
    plt.figure(figsize=(10, 6))
    plt.bar(clusters, davies_bouldin_scores, color='orange', label="Davies-Bouldin Score")
    plt.title(f"Davies-Bouldin Score por Cluster - {model_name}")
    plt.xlabel("Cluster")
    plt.ylabel("Davies-Bouldin Score")
    plt.grid(axis='y')
    plt.legend()
    plt.tight_layout()  # Ajusta o layout automaticamente
    plt.show()

def plot_2d_clusters(pandas_df, predictions, feature_01, feature_02, model_name):
    import plotly.graph_objects as go

    fig = go.Figure()
    for cluster in pandas_df[predictions].unique():
        cluster_data = pandas_df[pandas_df[predictions] == cluster]
        fig.add_trace(go.Scatter(
            x=cluster_data[feature_01],
            y=cluster_data[feature_02],
            mode='markers',
            marker=dict(size=6),
            name=f' {cluster}'
        ))
    fig.update_layout(
        title=f"Clusters em 2D ({model_name})",
        xaxis=dict(title=feature_01),
        yaxis=dict(title=feature_02),
        showlegend=True,
        width=2000,
        height=800
    )
    fig.show()



import plotly.express as px

def plot_fuzzy_probability_histogram(pandas_df, column_name="Probability", nbins=30, width=800, height=500):
    """
    Plota um histograma interativo das probabilidades de associação fuzzy usando Plotly.

    Parâmetros:
    - pandas_df: DataFrame contendo os dados.
    - column_name: Nome da coluna que contém as probabilidades (default: "Probability").
    - nbins: Número de bins no histograma (default: 30).
    - width: Largura do gráfico (default: 800).
    - height: Altura do gráfico (default: 500).
    """

    # Criando o histograma interativo com Plotly
    fig = px.histogram(
        pandas_df, 
        x=column_name, 
        nbins=nbins,  # Define o número de bins
        marginal="rug",  # Adiciona rug plot na margem
        opacity=0.75,  # Transparência para melhor visualização
        histnorm="probability density",  # Normalização correta da densidade
        title=f"Distribuição das Probabilidades de Associação Fuzzy ({column_name})"
    )

    # Ajustando layout
    fig.update_layout(
        xaxis_title="Probabilidade",
        yaxis_title="Densidade",
        width=width,
        height=height
    )

    # Mostrar gráfico
    fig.show()




import matplotlib.pyplot as plt
import seaborn as sns

def plot_rfm_relationships(df_spark, variables):
    """
    Função para gerar gráficos exploratórios de dispersão e relações entre variáveis RFM.

    Parâmetros:
    - df_spark: DataFrame PySpark contendo as colunas "Recency", "Frequency" e "Monetary".
    - variables: Lista com os nomes das colunas a serem analisadas.
    """

    #  Converter PySpark DataFrame para Pandas
    df_pandas = df_spark.select(*variables).toPandas()

    #  Garantir que o DataFrame Pandas não esteja vazio antes de plotar
    if df_pandas.empty:
        print(" Erro: O DataFrame está vazio! Verifique os dados antes de plotar.")
        return

    #  Configuração do estilo dos gráficos
    sns.set(style="whitegrid")

    #  Criar matriz de dispersão (Pairplot)
    g = sns.pairplot(df_pandas, vars=variables, diag_kind="kde", corner=True)
    g.fig.suptitle("Relação entre Recency, Frequency e Monetary", y=1.02, fontsize=16)
    plt.show()





import matplotlib.pyplot as plt
import statsmodels.api as sm

def plot_qq_pyspark(df_spark, variables):
    """
    Função otimizada para gerar QQ-Plots das variáveis selecionadas de um DataFrame PySpark,
    exibindo os gráficos lado a lado.

    Parâmetros:
    - df_spark: DataFrame PySpark contendo as colunas desejadas.
    - variables: Lista de colunas para gerar os QQ-Plots.
    """

    #  Converter PySpark DataFrame para Pandas
    df_pandas = df_spark.select(variables).toPandas()

    #  Criar subplots lado a lado
    fig, axes = plt.subplots(1, len(variables), figsize=(6 * len(variables), 6))

    # Caso seja apenas uma variável, transformar em lista para subplots
    if len(variables) == 1:
        axes = [axes]

    #  Gerar QQ-Plots para cada variável no subplot correspondente
    for i, var in enumerate(variables):
        sm.qqplot(df_pandas[var], line='s', ax=axes[i])
        axes[i].set_title(f"QQ-Plot de {var}")
        axes[i].grid()

    plt.tight_layout()  # Ajusta o espaçamento dos gráficos
    plt.show()




import matplotlib.pyplot as plt
import seaborn as sns

def plot_histograms(df_spark, features):
    """
    Plota histogramas para múltiplas variáveis de um DataFrame PySpark.

    Parâmetros:
    - df_spark: DataFrame PySpark contendo os dados.
    - features: Lista de colunas a serem plotadas.
    """
    
    # Converter PySpark DataFrame para Pandas
    df_pandas = df_spark.select(features).toPandas()

    #  Criar subplots lado a lado
    fig, axes = plt.subplots(1, len(features), figsize=(15, 5))

    for i, feature in enumerate(features):
        sns.histplot(df_pandas[feature], kde=True, bins=30, ax=axes[i])
        axes[i].set_title(f"Distribuição de {feature}")

    plt.tight_layout()  # Ajusta o espaçamento dos gráficos
    plt.show()



import matplotlib.pyplot as plt

def plot_pca_loadings(loadings_df, original_columns, figsize=(10, 6)):
    """
    Plota um gráfico de vetores (Loadings) para os dois primeiros componentes principais do PCA.

    Parâmetros:
    - loadings_df: DataFrame contendo os loadings do PCA.
    - original_columns: Lista de nomes das variáveis originais.
    - figsize: Tamanho da figura (default=(10, 6)).
    
    Retorna:
    - Exibe um gráfico com os vetores representando os loadings do PCA.
    """
    plt.figure(figsize=figsize)

    # Criar um grid para os vetores
    for i in range(len(original_columns)):
        plt.arrow(0, 0, 
                  loadings_df.iloc[0, i],  # PC1
                  loadings_df.iloc[1, i],  # PC2
                  head_width=0.03, 
                  head_length=0.05, 
                  fc='red', 
                  ec='black')

        plt.text(loadings_df.iloc[0, i] * 1.1, 
                 loadings_df.iloc[1, i] * 1.1, 
                 original_columns[i], 
                 color='blue', 
                 fontsize=12)

    # Adicionar círculos para representar a escala do gráfico
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    plt.grid()
    plt.xlabel('PC1 (Componente Principal 1)')
    plt.ylabel('PC2 (Componente Principal 2)')
    plt.title('Gráfico de Vetores (Loadings) do PCA')
    plt.show()




