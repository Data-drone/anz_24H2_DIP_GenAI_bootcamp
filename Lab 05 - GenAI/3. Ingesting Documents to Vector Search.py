# Databricks notebook source
# MAGIC %md
# MAGIC # Transforming files to vector indexes
# MAGIC
# MAGIC When building a vector index, we do not use documents as they are.\
# MAGIC We break it up into sections or chunks. \
# MAGIC This allows for us to do fine grained searches within the text and to locate specific paragraphs.\
# MAGIC

# COMMAND ----------

# DBTITLE 1,Setup Libraries
# MAGIC %pip install --upgrade --force-reinstall databricks-vectorsearch mlflow==2.16.2 langchain-text-splitters langchain-databricks==0.1.0 pypdf==5.0.1
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC # Building Our ETL Pipeline
# MAGIC
# MAGIC Building a chunking pipeline is just like building a standard Data Engineering Pipeline to process table data.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Configure Parameters
# We will create the following source Delta table.
import re
username = spark.sql("SELECT current_user()").first()['current_user()'].split('@')[0]
username_processed = re.sub(r'[^\w]', '_', username)

# UC location
source_catalog = username_processed
source_schema = "rag_ai_app"

# table naming
source_volume = "source_files"
raw_table = "bronze_raw_pdfs"
silver_table = "silver_parsed_pdfs"
gold_table = "gold_chunked_pdfs" 

embedding_endpoint_name = "databricks-gte-large-en"

######### IMPORTANT
###### You will be given a specific Vector Store Endpoint to use
vs_endpoint = <>

# COMMAND ----------

# MAGIC %md
# MAGIC # Bronze Layer - Ingestion
# MAGIC
# MAGIC The first step is to ingest the files into a Delta Table \
# MAGIC Loading it into a Delta table makes it easier for users on the platform to discover the interact with the data \
# MAGIC We will store it in bronze layer as a blob.
# COMMAND ----------

# DBTITLE 1,Loading the raw files
# import urllib
# file_uri = 'https://arxiv.org/pdf/2203.02155.pdf'
volume_path = f'/Volumes/{source_catalog}/{source_schema}/{source_volume}/'
#file_path = f"{volume_path}2203.02155.pdf"
# urllib.request.urlretrieve(file_uri, file_path)

raw_files_df = (
    spark.read.format("binaryFile")
    .option("recursiveFileLookup", "true")
    .option("pathGlobFilter", f"*.pdf")
    .load(volume_path)
)

# Save to a table
raw_files_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f'{source_catalog}.{source_schema}.{raw_table}'
)

# COMMAND ----------

# DBTITLE 1,Reviewing the Table
# reload to get correct lineage in UC
raw_files_df = spark.read.table(f'{source_catalog}.{source_schema}.{raw_table}')

# For debugging, show the list of files, but hide the binary content
display(raw_files_df.drop("content"))


# COMMAND ----------

# MAGIC %md
# MAGIC # Silver Layer - Parsing the files
# MAGIC
# MAGIC First step is to decode the file and extract the text. \
# MAGIC Each document will get extracted into one big long string.

# COMMAND ----------

# DBTITLE 1,Import Libs
from pypdf import PdfReader
from typing import TypedDict, Dict
import warnings
import io 
from pyspark.sql.types import StructType, StringType, StructField, MapType, ArrayType
import pyspark.sql.functions as func

# COMMAND ----------

# DBTITLE 1,Setup Function
class ParserReturnValue(TypedDict):
    doc_parsed_contents: Dict[str, str]
    parser_status: str


def parse_bytes_pypdf(
    raw_doc_contents_bytes: bytes,
) -> ParserReturnValue:
    try:
        pdf = io.BytesIO(raw_doc_contents_bytes)
        reader = PdfReader(pdf)

        parsed_content = [page_content.extract_text() for page_content in reader.pages]
        output = {
            "num_pages": str(len(parsed_content)),
            "parsed_content": "\n".join(parsed_content),
        }

        return {
            "doc_parsed_contents": output,
            "parser_status": "SUCCESS",
        }
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return {
            "doc_parsed_contents": {"num_pages": "", "parsed_content": ""},
            "parser_status": f"ERROR: {e}",
        }

# COMMAND ----------

# DBTITLE 1,Build Silver Table
parser_udf = func.udf(
    parse_bytes_pypdf,
    returnType=StructType(
        [
            StructField(
                "doc_parsed_contents",
                MapType(StringType(), StringType()),
                nullable=True,
            ),
            StructField("parser_status", StringType(), nullable=True),
        ]
    ),
)

parsed_files_staging_df = raw_files_df.withColumn("parsing", parser_udf("content")).drop("content")

parsed_files_df = parsed_files_staging_df.withColumn("doc_parsed_contents", func.col("parsing.doc_parsed_contents")).drop("parsing")

parsed_files_df.write.mode("overwrite").option("overwriteSchema", "true")\
  .saveAsTable(f'{source_catalog}.{source_schema}.{silver_table}')

print(f"Parsed {parsed_files_df.count()} documents.")

display(parsed_files_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup the Gold Layer - Chunking
# MAGIC
# MAGIC With this codebase, we will chunk the documents and make them ready for loading into a Vector Index. \
# MAGIC The best way to chunk will depend on the type of docment, it's formating and the information contained within.\
# MAGIC

# COMMAND ----------

# DBTITLE 1,Setting up my chunking functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ChunkerReturnValue(TypedDict):
    chunked_text: str
    chunker_status: str

def chunk_parsed_content_langrecchar(
    doc_parsed_contents: str, chunk_size: int, chunk_overlap: int) -> ChunkerReturnValue:
    
    # RecursiveCharacterTextSplitter is a basic format. There are many possible approaches
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=chunk_overlap,
      length_function=len,
      is_separator_regex=False,
    )

    chunks = text_splitter.split_text(doc_parsed_contents)

    return {
            "chunked_text": [doc for doc in chunks],
            "chunker_status": "SUCCESS",
        }

# COMMAND ----------

# DBTITLE 1,Setting up udf
from functools import partial

chunker_udf = func.udf(
    partial(
        chunk_parsed_content_langrecchar,
        chunk_size=1000,
        chunk_overlap=150
    ),
    returnType=StructType(
        [
            StructField("chunked_text", ArrayType(StringType()), nullable=True),
            StructField("chunker_status", StringType(), nullable=True),
        ]
    ),
)

# COMMAND ----------

# Run the chunker
chunked_files_df = parsed_files_df.withColumn(
    "chunked",
    chunker_udf("doc_parsed_contents.parsed_content"),
)

chunked_files_df = chunked_files_df.select(
    "path",
    func.explode("chunked.chunked_text").alias("chunked_text"),
    func.md5(func.col("chunked_text")).alias("chunk_id")
)

# Write to Delta Table
chunked_files_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f'{source_catalog}.{source_schema}.{gold_table}'
)

print(f"Produced a total of {chunked_files_df.count()} chunks.")

# Display without the parent document text - this is saved to the Delta Table
display(chunked_files_df)

# COMMAND ----------

# Prep table for indexing
spark.sql(f"""ALTER TABLE {source_catalog}.{source_schema}.{gold_table}
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true)""")  

# COMMAND ----------

# MAGIC %md
# MAGIC # Discussion
# MAGIC
# MAGIC We have now chunked and setup the files\
# MAGIC Now that the table is ready, we can create a vector index on top of it
# MAGIC
