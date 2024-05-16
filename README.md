# What's in a Name? Auditing Large Language Models for Race and Gender Bias

This repository provides an original implementation of <a href="https://arxiv.org/abs/2402.14875" target="_blank">What's in a Name? Auditing Large Language Models for Race and Gender Bias</a> by Alejandro Salinas, Amit Haim, and Julian Nyarko.

# 1. Setup

## Requirements
All requirements can be found in requirements.txt. Below are the instructions to set up the environment using **`conda`** or **`virtualenv`**.

### Using Conda
1. Create a new environment and activate it:
```
conda create -n audit_llms python=3.11.3
conda activate audit_llms
```
2. Clone the repository and install dependencies:
```
git clone https://github.com/AlexSalinas99/audit_llms.git
cd audit_llms
pip install -r requirements.txt
```

### Using Virtualenv
1. Create a new environment and activate it:
```
python -m virtualenv -p python3.11.3 audit_llms
source audit_llms/bin/activate
```
2. Clone the repository and install dependencies:
```
git clone https://github.com/AlexSalinas99/audit_llms.git
cd audit_llms
pip install -r requirements.txt
```

# 2. Usage

## API Keys
For closed source models (<a href="https://openai.com/api/" target="_blank">OpenAI</a>, <a href="https://docs.mistral.ai/api/" target="_blank">Mistral-large</a>, <a href="https://ai.google.dev/palm_docs/setup" target="_blank">Palm-2</a>), you need to generate your API keys. We also used <a href="https://replicate.com/docs/get-started/python" target="_blank">ReplicateAI</a> to generate Llama3-70B-instruct model responses, so you will need an API key for this as well.

## Data Files

The **`data`** folder includes the following files:

* **just_prompts.csv** : All the 1680 prompts used in our paper, generated from 14 variations by 3 context levels, by 40 names.

* **raw_responses.csv** (forthcoming): All raw responses from all models.

* **cleaned_responses.csv** (forthcoming): All cleaned responses from all models.

## Notebooks

The **`notebooks`** folder includes the following files:

1. **generate_responses_from_models.ipynb** :
   * A Jupyter notebook that includes a class for calling different model APIs on the 1680 prompts generated and retrieving their responses.
   * Supported models: **`gpt-4-1106-preview`**, **`gpt-3.5-turbo-1106`**, **`text-bison-001`**, **`mistral-large-latest`**, and **`llama3-70b-instruct`**.

2. **prompt_generation.ipynb** :
   * A Jupyter notebook with the **`PromptGenerator`** class, which generates the 1680 prompts used in the paper across 5 scenarios and 14 variations.
   * Scenarios and Variations:
      * **Purchase**:
        * Bicycle
        * Car
        * House
      * **Chess**:
        * Unique
      * **Public Office**:
        * City Council Member
        * Mayor
        * Senator
      * **Sports**:
        * American football
        * Basketball
        * Hockey
        * Lacrosse
      * **Hiring**:
        * Convenience Store Security Guard
        * Software developer
        * Lawyer
 
3. **cleaning_general.ipynb** :
   * A Jupyter notebook to automate the extraction of relevant text from the model's responses.
   * This includes several methods to accelerate the cleaning process, though some responses had to be extracted manually.

4. **visualization_general.ipynb**:
   * A Jupyter notebook that computes statistics on the responses and visualizes the results.
   * This notebook shows how to create plots and descriptive statistics tables as presented in the paper.

5. **rankings_stereotypes.ipynb** (forthcoming):
   * A Jupyter notebook that calculates the cosine similarity between the original prompt embeddings and the stereotype prompt embeddings (with a positive/negative stereotype replacing the name)
   * It performs a Wilcox-sum (Mann-Whitney U) ranking test and a chi-square test.

# 3. Contributing
We welcome contributions! Please submit issues or pull requests if you have suggestions or improvements.
