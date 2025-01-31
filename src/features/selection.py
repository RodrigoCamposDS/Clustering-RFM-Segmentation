from pyspark.sql import functions as F

def calcular_limites_outliers(df, coluna):
    """
    Calcula os limites inferior e superior para outliers usando o IQR (Interquartile Range).
    """
    # Calcular o 1º e 3º quartis
    q1, q3 = df.approxQuantile(coluna, [0.25, 0.75], 0.05)  # 5% de precisão

    # Calcular o IQR
    iqr = q3 - q1
    
    # Limites inferior e superior para outliers
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    
    return lower_limit, upper_limit