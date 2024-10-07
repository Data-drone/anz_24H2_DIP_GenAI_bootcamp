# Databricks notebook source
# MAGIC %md
# MAGIC # Setting up Our Data Source
# MAGIC
# MAGIC So we saw before that adding information context is key to making LLMs able to answer questions specific to your organisation \
# MAGIC Or to add knowledge of recent events that may not be in it's training data. \
# MAGIC
# MAGIC The key to finding information, our document library for LLMs, is the Vector Search Engine \
# MAGIC Rather than just relying on keyword search like old school document databases, Vector Search Engines can also look at the similarity of phrases \
# MAGIC This allows us to look more at the meaning of sentences rather than just keyword search.

# COMMAND ----------

# DBTITLE 1,Configuration Parameters

# extract current username
import re
username = spark.sql("SELECT current_user()").first()['current_user()'].split('@')[0]
username_processed = re.sub(r'[^\w]', '_', username)


######### Edit these to customise location (optional)
db_catalog = username_processed
db_schema = 'rag_ai_app'
volume_name = 'source_files'
##########################################

# Get the workspace url to make a shortcuts
from dbruntime.databricks_repl_context import get_context
ctx = get_context()
volume_folder = f'/Volumes/{db_catalog}/{db_schema}/{volume_name}/'
vol_url = f"https://{ctx.browserHostName}/explore/data/volumes/{db_catalog}/{db_schema}/{volume_name}"

# COMMAND ----------

# DBTITLE 1,Setup Catalog and Schema

# During the lab we will do this with click ops.
#spark.sql(f"CREATE CATALOG IF NOT EXISTS {db_catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {db_catalog}.{db_schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {db_catalog}.{db_schema}.{volume_name}")

print(f"Load files manually with this link: {vol_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Files with Code
# MAGIC
# MAGIC If you don't have any pdfs that you want to load manually then you can use this code to load some from the internet\
# MAGIC Just uncomment the last cell. 

# COMMAND ----------

# DBTITLE 1,Files to Load
# docs to load
pdfs = {'2203.02155.pdf':'https://arxiv.org/pdf/2203.02155.pdf',
        '2302.09419.pdf': 'https://arxiv.org/pdf/2302.09419.pdf',
        'Brooks_InstructPix2Pix_Learning_To_Follow_Image_Editing_Instructions_CVPR_2023_paper.pdf': 'https://openaccess.thecvf.com/content/CVPR2023/papers/Brooks_InstructPix2Pix_Learning_To_Follow_Image_Editing_Instructions_CVPR_2023_paper.pdf',
        '2303.10130.pdf':'https://arxiv.org/pdf/2303.10130.pdf',
        '2302.06476.pdf':'https://arxiv.org/pdf/2302.06476.pdf',
        '2302.06476.pdf':'https://arxiv.org/pdf/2302.06476.pdf',
        '2303.04671.pdf':'https://arxiv.org/pdf/2303.04671.pdf',
        '2209.07753.pdf':'https://arxiv.org/pdf/2209.07753.pdf',
        '2302.07842.pdf':'https://arxiv.org/pdf/2302.07842.pdf',
        '2302.07842.pdf':'https://arxiv.org/pdf/2302.07842.pdf',
        '2204.01691.pdf':'https://arxiv.org/pdf/2204.01691.pdf'}

# COMMAND ----------

# DBTITLE 1,Download file script
import os
import requests
user_agent = "me-me-me"

def load_file(file_uri: str, file_name: str, library_folder: str) -> None:
    """

    This function is designed to loop through the provided urls and load them to a particular folder

    """
    
    # Create the local file path for saving the PDF
    local_file_path = os.path.join(library_folder, file_name)

    # Download the PDF using requests
    try:
        # Set the custom User-Agent header
        headers = {"User-Agent": user_agent}

        response = requests.get(file_uri, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            # Save the PDF to the local file
            with open(local_file_path, "wb") as pdf_file:
                pdf_file.write(response.content)
            print(f"PDF {file_name} downloaded successfully.")
        else:
            print(f"Failed to download PDF. Status code: {response.status_code}")
    except requests.RequestException as e:
        print("Error occurred during the request:", e)


# COMMAND ----------

# DBTITLE 1,Download the files

########## Uncomment to run (Optional)
#for pdf in pdfs.keys():
#    load_file(pdfs[pdf], pdf, volume_folder)