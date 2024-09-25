# Databricks notebook source
username = spark.sql("SELECT current_user()").first()['current_user()'].replace('@vocareum.com','')

# COMMAND ----------

import random


random.seed(username)

# assign vs search endpoint by username
vs_endpoint_prefix = "vs_endpoint_"
vs_endpoint_fallback = "vs_endpoint_fallback"
vs_endpoint_name = vs_endpoint_prefix+str(random.randint(1,9))
print(f"Vector Endpoint name: {vs_endpoint_name}. In case of any issues, replace variable `vs_endpoint_name` with `vs_endpoint_fallback` in demos and labs.")


# COMMAND ----------


