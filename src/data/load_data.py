import pandas as pd
import os
import chardet
from pyspark.sql import SparkSession

def load_data(filename):
    """
    Função para carregar o dataset usando PySpark.
    """
    # Verificar se já existe uma SparkSession
    spark = SparkSession.builder.getOrCreate()  # Isso usa a sessão existente, se houver

    # Caminho absoluto do diretório raiz do projeto
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # Caminho completo até a pasta raw
    raw_path = os.path.join(base_dir, 'src', 'data', 'raw')

    # Concatena o caminho com o nome do arquivo
    file_path = os.path.join(raw_path, filename)

    # Carregar o arquivo CSV com PySpark
    df = spark.read.csv(file_path, header=True, inferSchema=True)

    return df