# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

pip install pyspark
pip install requests
pip install newsapi-python

from newsapi import NewsApiClient
import re
import requests
import json

pip install nltk
pip install spacy
pip install stopwords
nltk.download('punkt')
nltk.download('stopwords')

url = ('http://newsapi.org/v2/top-headlines?'
       'country=us&')
api_key = '2350f033f0694b55af423991fdbdd532'  
params = {
    'pageSize': 1000, 
    'apiKey': api_key
}

all_articles = []

page = 50
while True:
    params['page'] = page
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        articles = data['articles']
        if len(articles) == 0:
            break
        all_articles.extend(articles)
        page += 1
    else:
        break
for item in articles:
    if 'source' in item:
        del item['source']

# Loại bỏ dữ liệu None
cleaned_data = []
for item in articles:
    cleaned_item = {k: v for k, v in item.items() if v is not None} 
    cleaned_data.append(cleaned_item)
print(cleaned_data)

#Loaibodulieutrunglap
for item in cleaned_data:
    if item not in cleaned_data:
        cleaned_data.append(item)
print(cleaned_data)

from pyspark.sql import SparkSession
import json

# Khởi tạo phiên Spark
spark = SparkSession.builder.appName("SparkSql").getOrCreate()

# Tạo DataFrame từ dữ liệu
df = spark.createDataFrame(cleaned_data)
df_cleaned = df.na.drop()
df_cleaned.show()

# Khởi tạo phiên Spark
spark = SparkSession.builder.appName("SparkSQL to PostgreSQL").config("spark.jars", "path/to/postgresql-connector.jar").getOrCreate()

# Đọc dữ liệu từ PostgreSQL vào DataFrame
df = spark.read.format("jdbc").option("url", "jdbc:postgresql://localhost:5432/db_name").option("dbtable", "table_name").option("user", "username").option("password", "password").load()

# Thực hiện các phép biến đổi trên DataFrame nếu cần thiết

# Lưu trữ DataFrame vào PostgreSQL
df.write.format("jdbc").option("url", "jdbc:postgresql://localhost:5432/db_name").option("dbtable", "table_name").option("user", "username").option("password", "password").save()

# Dừng phiên Spark
spark.stop()

    descriptions = [article.description for article in df_cleaned]
    # Phân tích và xử lý từng mô tả tin tức
    for idx, description in enumerate(descriptions, start=1):
        print(f"Phân tích tin tức {idx}:")
        
        # Tách từ
        tokens = word_tokenize(descriptions)
        
        # Loại bỏ stop words
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        
        #Loại bỏ dấu 
        result_string = re.sub(r'[^\w\s]', '', filtered_tokens)

        # Tính toán tần suất từ
        fdist = FreqDist(result_string)
        print(fdist.most_common(5))
        print("\n")

from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

# Khởi tạo phiên Spark
spark = SparkSession.builder.appName("TF-IDF Example").getOrCreate()

# Chia từ trong cột "description"
tokenizer = Tokenizer(inputCol="description", outputCol="words")
wordsData = tokenizer.transform(df_cleaned)

# Ánh xạ từ thành vectors sử dụng TF
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)

# Tính toán IDF
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Hiển thị kết quả
rescaledData.select("description", "features").show(truncate=False)

# Dừng phiên Spark
spark.stop()
