# Databricks notebook source
# MAGIC %md # Building your first RAG app
# MAGIC
# MAGIC This notebook covers how to build a full customised application \
# MAGIC Building a full application is how you are going to approach the task IRL \
# MAGIC This gives you:
# MAGIC - Control
# MAGIC - Governance
# MAGIC - Logic customisation
# MAGIC
# MAGIC As compared to just using the playground with tools.
# MAGIC
# MAGIC Databricks provides a review application out of the box when you deploy with databricks-agents as documented in the next Notebook. \
# MAGIC That, however, requires that our langchain is developed to accept standard ChatML format.

# COMMAND ----------

# DBTITLE 1,Install Libs
# MAGIC %pip install -U  mlflow==2.16.2 langchain==0.2.16 langchain_community==0.2.17 langchain-databricks==0.1.0
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Setup Params 
import re
import mlflow
#username = spark.sql("SELECT current_user()").first()['current_user()'].split('@')[0]
#username_processed = re.sub(r'[^\w]', '_', username)

#### Enter your own naming as required
db_catalog = 'brian_test'
db_schema = 'rag_ai_app'
vs_index_name = "gold_chunked_pdfs_index"

endpoint_name = 'brian_endpoint'
vs_index_fullname = f"{db_catalog}.{db_schema}.{vs_index_name}"

# temp need to change later
embedding_model = "databricks-gte-large-en"
chat_model = "databricks-meta-llama-3-1-70b-instruct"

# COMMAND ----------

# DBTITLE 1,Load Instruction Finetuned Model
from langchain_databricks import ChatDatabricks

llm_model = ChatDatabricks(
            target_uri='databricks',
            endpoint=chat_model,
            temperature=0.1
        )

# Lets quickly test the model
llm_model.invoke("Hello")

# COMMAND ----------

# DBTITLE 1,Load Embedding Model
from langchain_databricks import DatabricksEmbeddings

embeddings = DatabricksEmbeddings(endpoint=embedding_model)

embeddings.embed_query("Embed my test question")

# COMMAND ----------

# DBTITLE 1,Load DB Vector Store
from databricks.vector_search.client import VectorSearchClient
from langchain_databricks import DatabricksVectorSearch

vsc = VectorSearchClient()
index = vsc.get_index(endpoint_name=endpoint_name,
                      index_name=vs_index_fullname)

index.similarity_search(columns=["chunked_text"],query_text="Tell me about tuning LLMs")

# COMMAND ----------

# DBTITLE 1,Build Retrieval Chain
retriever = DatabricksVectorSearch(
    vs_index_fullname,  
    columns=["path"]
).as_retriever()

retriever.invoke("What is a RAG?")

# COMMAND ----------

# MAGIC %md
# MAGIC # Setting up the chain
# MAGIC
# MAGIC The most common chat format is the ChatML format as defined originally by OpenAI.
# MAGIC
# MAGIC A Chatml conversation looks like this:
# MAGIC ```
# MAGIC [
# MAGIC   {
# MAGIC     "role": "system",
# MAGIC     "content": "You are a helpful assistant."
# MAGIC   },
# MAGIC   {
# MAGIC     "role": "user",
# MAGIC     "content": "What's the weather like today?"
# MAGIC   },
# MAGIC   {
# MAGIC     "role": "assistant",
# MAGIC     "content": "I'm sorry, but I don't have access to real-time weather information. To get accurate weather details for today, you would need to check a reliable weather service or app that provides up-to-date forecasts for your specific location."
# MAGIC   },
# MAGIC   {
# MAGIC     "role": "user",
# MAGIC     "content": "Can you tell me a joke instead?"
# MAGIC   },
# MAGIC   {
# MAGIC     "role": "assistant",
# MAGIC     "content": "Sure! Here's a light-hearted joke for you: Why don't scientists trust atoms? Because they make up everything!"
# MAGIC   }
# MAGIC ]
# MAGIC ```
# MAGIC
# MAGIC It is a list with a series of dictionaries inside. \
# MAGIC Each dictionary has a `role` and `content` keys\
# MAGIC role starts with `system``at the top, this is your system message\
# MAGIC The rest alternates between `assistant` the bot and `user` the person/
# MAGIC
# MAGIC There are no default langchain parsers for this at this time so we need to write our own for our chain \
 
# COMMAND ----------

# DBTITLE 1,ChatML Message Format Formatters
from langchain_core.messages import HumanMessage, AIMessage

def extract_user_query_string(chat_messages_array):
    # We need to sseparate out the last message, text only
    return chat_messages_array[-1]["content"]

def extract_chat_history(chat_messages_array):
    # we also need to extract out the historical messages
    return chat_messages_array[:-1]

def format_chat_history_for_prompt(chat_messages_array):
    # We also need to reformat the messages for the prompt template
    history = extract_chat_history(chat_messages_array)
    formatted_chat_history = []
    if len(history) > 0:
        for chat_message in history:
            if chat_message["role"] == "user":
                formatted_chat_history.append(
                    HumanMessage(content=chat_message["content"])
                )
            elif chat_message["role"] == "assistant":
                formatted_chat_history.append(
                    AIMessage(content=chat_message["content"])
                )
    return formatted_chat_history


# COMMAND ----------

# DBTITLE 1,Setup Prompt Templates
# We need to setup a cpl of prompt templates one will be for incorporating chat history into our line of questions for the vector search
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate

query_rewrite_template = """Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natural language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.

Chat history: {chat_history}

Question: {question}"""

query_rewrite_prompt = PromptTemplate(
    template=query_rewrite_template,
    input_variables=["chat_history", "question"]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (  # System prompt contains the instructions
            "system",
            "ou are a helpful assistant designed to help customers undertake research on RAG models",
        ),
        # If there is history, provide it.
        # Note: This chain does not compress the history, so very long converastions can overflow the context window.
        MessagesPlaceholder(variable_name="formatted_chat_history"),
        # User's most current question
        ("user", "{question}"),
    ]
)
# COMMAND ----------

# DBTITLE 1,Build Chain with Prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from operator import itemgetter

def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)


# Build Rag Chain
# For consistency we use input as the key of the input json as well.
# The LCEL code then remaps it to user_input to sent to our basic_template
rag_chain = (
     {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_chat_history),
        "formatted_chat_history": itemgetter("messages") | RunnableLambda(format_chat_history_for_prompt),
    }
    | RunnablePassthrough()
    | {
        "context": RunnableBranch(  # Only re-write the question if there is a chat history
            (
                lambda x: len(x["chat_history"]) > 0,
                query_rewrite_prompt | llm_model | StrOutputParser(),
            ),
            itemgetter("question"),
        )
        | retriever
        | RunnableLambda(format_docs),
        "formatted_chat_history": itemgetter("formatted_chat_history"),
        "question": itemgetter("question"),
    }
    | generation_prompt
    | llm_model
    | StrOutputParser()
)


rag_chain.invoke({"messages": [{"role": "user", "content": "Why do I need RAG techniques?"}]})

# COMMAND ----------

# Set the model for productionisation
mlflow.models.set_model(model=rag_chain)

# Setting the retriever for the agent framework
mlflow.models.set_retriever_schema(
    primary_key="chunk_id",
    text_column="chunked_text",
    doc_uri="path"
    )  # Review App uses `doc_uri` to display chunks from the same document in a single view


# COMMAND ----------

# MAGIC %md
# MAGIC We have now got our chain developed and ready to deploy \
# MAGIC The next Notebook will take this chain and deploy it