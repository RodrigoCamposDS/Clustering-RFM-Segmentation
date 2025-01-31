import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

def calcular_metricas(data_matrix, labels):
    """
    Função para calcular métricas de avaliação de clusters.
    """
    # Verificar se há mais de um cluster válido
    if len(set(labels)) > 1:
        silhouette_scores = silhouette_score(data_matrix, labels, metric='euclidean')
        davies_bouldin_scores = davies_bouldin_score(data_matrix, labels)
        print(f"Silhouette Score: {silhouette_scores:.4f}")
        print(f"Davies-Bouldin Score: {davies_bouldin_scores:.4f}")
        return silhouette_scores, davies_bouldin_scores
    else:
        print("Não foi possível calcular as métricas - número insuficiente de clusters.")
        return None, None
    


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import resample

def calcular_matriz_similaridade(data_matrix, sample_size=0.1):
    """
    Função para calcular uma matriz de similaridade (ou distância) entre os clusters
    utilizando uma amostragem para otimização.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Amostrar o número de pontos (sample_size pode ser quantidade ou porcentagem)
    if sample_size < 1:  # Se for fração, calcular porcentagem
        sample_size = int(len(data_matrix) * sample_size)
    
    if sample_size < len(data_matrix):
        np.random.seed(42)  # Para resultados consistentes
        sampled_indices = np.random.choice(len(data_matrix), sample_size, replace=False)
        data_matrix = data_matrix[sampled_indices]
    
    similarity_matrix = cosine_similarity(data_matrix)
    print("Matriz de Similaridade calculada com amostragem.")
    return similarity_matrix


from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
   


# ---- MÉTRICAS INTERNAS ---- #
def calcular_metricas_internas(data_matrix, labels):
    if len(set(labels)) > 1:  # Verifica se há mais de um cluster
        print(f"Índice de Silhouette (Euclidean): {silhouette_score(data_matrix, labels, metric='euclidean'):.4f}")
        print(f"Índice de Silhouette (Manhattan): {silhouette_score(data_matrix, labels, metric='manhattan'):.4f}")
        print(f"Índice de Silhouette (Cosine): {silhouette_score(data_matrix, labels, metric='cosine'):.4f}")
        print(f"Davies-Bouldin: {davies_bouldin_score(data_matrix, labels):.4f}")
        print(f"Calinski-Harabasz: {calinski_harabasz_score(data_matrix, labels):.4f}")
    else:
        print("Clusters inválidos para métricas internas.")
        