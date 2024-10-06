# Databricks notebook source
# MAGIC %md
# MAGIC # Using Databricks Vector Search
# MAGIC

# COMMAND ----------

# DBTITLE 1,Install Libs
# MAGIC %pip install --upgrade --force-reinstall databricks-vectorsearch mlflow==2.16.2 langchain-databricks==0.1.0
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Configure Parameters
# We need to set these correctly first
import re
username = spark.sql("SELECT current_user()").first()['current_user()'].split('@')[0]
username_processed = re.sub(r'[^\w]', '_', username)

# UC location
source_catalog = username_processed
source_schema = "rag_ai_app"
index_name = "gold_chunked_pdfs_index"

vs_endpoint = <>

# COMMAND ----------

# DBTITLE 1,Setup Index Connection
# given these chunks Databricks Vector Search will manage the vector sync and we can focus on making sure that the chunking is working
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
# vs_endpoint
vsc.get_endpoint(
  name=vs_endpoint
)

vs_index_fullname = f"{source_catalog}.{source_schema}.{index_name}"
index = vsc.get_index(endpoint_name=vs_endpoint,index_name=vs_index_fullname)

# COMMAND ----------

# DBTITLE 1,Search Index
my_query = "Tell me about tuning LLMs"

results = index.similarity_search(
  columns=["chunked_text"],
  # vs_index_fullname,
  query_text = my_query,
  num_results = 3
  )

# COMMAND ----------

# Explore the results
results

# COMMAND ----------

# pulling the top result
print(results['result']['data_array'][0][0])

# COMMAND ----------

# DBTITLE 1,Lets make the vector store index a tool
# Run the printed SQL Statement in a SQL Serverless Session
print(f"""
  CREATE OR REPLACE FUNCTION {source_catalog}.{source_schema}.vector_index_sesarch(
    question STRING COMMENT "this is index about how to better align LLM responses with human preferences through various techniques like finetuning and enhanced RAG Architectures"
    )
  RETURNS STRING
  LANGUAGE SQL
  COMMENT 'This is a search for anything regarding your vector store index' 
  RETURN 
    SELECT string(collect_set(chunked_text)) 
    FROM vector_search(index => "{source_catalog}.{source_schema}.{index_name}", query => question, num_results => 5);
""")

# We can now find this function and use it from within playground
