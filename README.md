# Databricks 2024 H2 Lab

This is a Lab to show how to work with Large Language Models on Databricks.

Run through the notebooks in order from 1 to 6.

NOTE:
- We assume access to Databricks Pay Per Token models which are currently just in US Regions.
- Each Notebook contains a configuration block, make sure that the catalog / schema / volume and table flags there are set correctly before you run each notebook

## Individual Notebook Tips

**Notebook 1** can be run totally from within Databricks Playground by copying the prompts into the Playground UI.
See: https://docs.databricks.com/en/large-language-models/ai-playground.html

**Notebook 2** is optional. You can create all the required resources via the UI.
See:
- Create Catalog - https://docs.databricks.com/en/catalogs/create-catalog.html#create-a-catalog
- Create Schema - https://docs.databricks.com/en/schemas/create-schema.html#create-a-schema
- Create Volume - https://docs.databricks.com/en/volumes/utility-commands.html#create-a-volume

You will need to find some PDF files to upload. PDF files should be primarily text based with minimal diagrams and images

**Notebook 4** assumes you have created the Vector Search Index via the UI
See: https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-endpoint-using-the-ui

