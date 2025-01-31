# from pyspark.ml.feature import VectorAssembler
# from pyspark.sql import SparkSession
# from scalers import apply_scaler  # Importar a função criada
# from src.data.load_data import load_data
# # ---- INICIAR SPARK SESSION ---- #
# spark = SparkSession.builder.appName("DataScalingPipeline").getOrCreate()

# # ---- CARREGAR OS DADOS ---- #
# # Exemplo de carga de dados

# df_tranformed
# columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
# df = spark.createDataFrame(df_tranformed, columns)

# # ---- CRIAR O VETOR DE FEATURES ---- #
# assembler = VectorAssembler(inputCols=["Recency", "Frequency", "Monetary"], outputCol="features")
# df_features = assembler.transform(df_tranformed)

# # ---- APLICAR SCALERS ---- #
# # StandardScaler
# df_standard_scaled = apply_scaler(df_features, scaler_type="standard")
# df_standard_scaled.select("features", "scaledFeatures").show(truncate=False)

# # MinMaxScaler
# df_minmax_scaled = apply_scaler(df_features, scaler_type="minmax")
# df_minmax_scaled.select("features", "scaledFeatures").show(truncate=False)

# # MaxAbsScaler
# df_maxabs_scaled = apply_scaler(df_features, scaler_type="maxabs")
# df_maxabs_scaled.select("features", "scaledFeatures").show(truncate=False)

# # RobustScaler
# df_robust_scaled = apply_scaler(df_features, scaler_type="robust")
# df_robust_scaled.select("features", "scaledFeatures").show(truncate=False)