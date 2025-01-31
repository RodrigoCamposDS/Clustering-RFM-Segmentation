from pyspark.ml.feature import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

def apply_scaler(df, scaler_type="standard", input_col="features", output_col="scaledFeatures"):
    """
    Função para aplicar diferentes escaladores no PySpark.

    Args:
        df (DataFrame): DataFrame PySpark com os dados de entrada.
        scaler_type (str): Tipo de escalador ("standard", "minmax", "maxabs", "robust").
        input_col (str): Nome da coluna de entrada (features).
        output_col (str): Nome da coluna de saída (scaledFeatures).

    Returns:
        DataFrame: DataFrame transformado com a coluna escalada.
    """
    print(f"Aplicando {scaler_type} scaler...")
    
    if scaler_type == "standard":
        scaler = StandardScaler(inputCol=input_col, outputCol=output_col, withStd=True, withMean=True)
    elif scaler_type == "minmax":
        scaler = MinMaxScaler(inputCol=input_col, outputCol=output_col)
    elif scaler_type == "maxabs":
        scaler = MaxAbsScaler(inputCol=input_col, outputCol=output_col)
    elif scaler_type == "robust":
        scaler = RobustScaler(inputCol=input_col, outputCol=output_col)
    else:
        raise ValueError("Scaler type not recognized. Choose from: 'standard', 'minmax', 'maxabs', 'robust'.")

    # Ajustar o escalador e transformar os dados
    scaler_model = scaler.fit(df)
    scaled_df = scaler_model.transform(df)

    print(f"Escalonamento com {scaler_type} completo.")
    scaled_df.select(output_col).show(truncate=False)

    return scaled_df