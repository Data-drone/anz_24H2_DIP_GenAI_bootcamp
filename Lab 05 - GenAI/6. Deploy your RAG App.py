# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Building and Logging A Parameterised Retriever Model File Chain
# MAGIC
# MAGIC We will do a model file chain with more advanced configs
# MAGIC (WIP - This is not ready yet)

# COMMAND ----------

# DBTITLE 1,Run Pip Install
# MAGIC %pip install -U databricks-agents databricks-sdk mlflow==2.16.2 langchain==0.2.16 langchain_community==0.2.17 langchain-databricks==0.1.0 tiktoken
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Config Management
import re
import mlflow
import os
from databricks import agents
username = spark.sql("SELECT current_user()").first()['current_user()']
username_processed = username.split('@')[0]
username_processed = re.sub(r'[^\w]', '_', username_processed)

#### Enter your own naming as required
db_catalog = username_processed
db_schema = 'rag_ai_app'
model_name = 'my_first_rag_chain'

experiment_name = 'first_rag_app'

mlflow.set_experiment(f'/Users/{username}/{experiment_name}')

# COMMAND ----------
# DBTITLE 1,Setup Evals
import pandas as pd

# We use a couple of test prompts to see how well they perform
evaluations = pd.DataFrame(
    {'request': [
        'What is a RAG?',
        'In what ways can RAGs go wrong?',
        'Why did the chicken cross the road?'
    ]}
)

# Eval Function for mlflow evals
def eval_pipe(inputs):

        def invoke_chain(prompt):
            return chain.invoke(input={"messages": [
             {"role": "user", "content": prompt}
        ]})

        answers = inputs['request'].apply(invoke_chain)
        #answer = chain.invoke(context="", data=inputs)
        return answers.tolist()

# COMMAND ----------

# DBTITLE 1,Log and version our model
with mlflow.start_run(run_name='Rag_chain'):

    mlflow.set_tag("type", "chain")

    logged_chain_info = mlflow.langchain.log_model(
        lc_model=os.path.join(
            os.getcwd(), '5. Building your RAG App'
        ),  # Chain code file e.g., /path/to/the/chain.py
        artifact_path="chain",  # Required by MLflow
        input_example={"messages": [
             {"role": "user", "content": "Why do I need RAG techniques?"}
        ]},  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
        registered_model_name=f'{db_catalog}.{db_schema}.retrieval_chain_model_file'
    )

    chain = mlflow.langchain.load_model(logged_chain_info.model_uri)

    # Setting it to Databricks Agent sets it up to use our custom designed
    # LLM-as-a-Judge stack
    results = mlflow.evaluate(eval_pipe,
                          data=evaluations,
                          model_type='text')

# COMMAND ----------

# DBTITLE 1,Setup Review App Instruction
instructions_to_reviewer = f"""## Instructions for Testing the Initial Proof of Concept (PoC)

Your inputs are invaluable for the development team. By providing detailed feedback and corrections, you help us fix issues and improve the overall quality of the application. We rely on your expertise to identify any gaps or areas needing enhancement.

1. **Variety of Questions**:
   - Please try a wide range of questions that you anticipate the end users of the application will ask. This helps us ensure the application can handle the expected queries effectively.

2. **Feedback on Answers**:
   - After asking each question, use the feedback widgets provided to review the answer given by the application.
   - If you think the answer is incorrect or could be improved, please use "Edit Answer" to correct it. Your corrections will enable our team to refine the application's accuracy.

3. **Review of Returned Documents**:
   - Carefully review each document that the system returns in response to your question.
   - Use the thumbs up/down feature to indicate whether the document was relevant to the question asked. A thumbs up signifies relevance, while a thumbs down indicates the document was not useful.

Thank you for your time and effort in testing our app. Your contributions are essential to delivering a high-quality product to our end users."""

print(instructions_to_reviewer)

# COMMAND ----------

# DBTITLE 1,Deploy Review App
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
from databricks.sdk.errors import NotFound, ResourceDoesNotExist

w = WorkspaceClient()

# Deploy to enable the Review APP and create an API endpoint
deployment_info = agents.deploy(model_name=f'{db_catalog}.{db_schema}.retrieval_chain_model_file', 
                                model_version=logged_chain_info.registered_model_version)

browser_url = mlflow.utils.databricks_utils.get_browser_hostname()
print(f"\n\nView deployment status: https://{browser_url}/ml/endpoints/{deployment_info.endpoint_name}")

# Add the user-facing instructions to the Review App
agents.set_review_instructions(f'{db_catalog}.{db_schema}.retrieval_chain_model_file', 
                               instructions_to_reviewer)

# Wait for the Review App to be ready
print("\nWaiting for endpoint to deploy.  This can take 15 - 20 minutes.", end="")
while w.serving_endpoints.get(deployment_info.endpoint_name).state.ready == EndpointStateReady.NOT_READY or w.serving_endpoints.get(deployment_info.endpoint_name).state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
    print(".", end="")
    time.sleep(30)

print(f"\n\nReview App: {deployment_info.review_app_url}")
