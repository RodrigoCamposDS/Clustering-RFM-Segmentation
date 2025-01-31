from pyspark.sql import SparkSession
import os

# Função para inicializar a SparkSession
def create_spark_session(app_name="LoadFeatures"):
    """
    Inicializa uma SparkSession com configurações básicas.
    
    Parâmetros:
    - app_name: Nome da aplicação Spark.
    
    Retorna:
    - spark: SparkSession configurada.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryo.registrationRequired", "false") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    return spark

# Função para obter o caminho do arquivo Parquet
def get_parquet_path():
    """
    Obtém o caminho absoluto do arquivo Parquet salvo.
    
    Retorna:
    - input_path: Caminho completo para o arquivo Parquet.
    """
    # Obter o caminho absoluto do diretório do projeto
    project_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), '..'))
    
    # Caminho completo para a pasta src/data/processed_data
    src_path = os.path.join(project_path, 'RID184082_Desafio07/project/src/data/processed_data')
    
    # Caminho do arquivo Parquet
    input_path = os.path.join(src_path, 'df.parquet')
    return input_path

# Função principal para carregar o DataFrame
def load_parquet_to_df(spark, input_path):
    """
    Carrega o DataFrame a partir de um arquivo Parquet.
    
    Parâmetros:
    - spark: SparkSession inicializada.
    - input_path: Caminho do arquivo Parquet.
    
    Retorna:
    - df: DataFrame PySpark carregado.
    """
    df = spark.read.parquet(input_path)
    return df

