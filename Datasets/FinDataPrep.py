
'''
Finance Specific Datasets: FinGPT/fingpt-fiqa_qa | bavest/fin-llama-dataset | nihiluis/financial-advisor-100 | gbharti/finance-alpaca
Unify the data set into the following format:

    Below is an instruction that describes a task. Write a response that appropriately completes the request.
    ### Question:
    What athlete created the 'beast quake' for the...
    ### Answer:
    Marshawn Lynch
    ###end

Combine the general datasets, then combine them into one for finetuning
'''
from datasets import load_dataset
import pandas as pd
import re

def format_row(row, question_key, ans_key):
    """
    Formats a single row from the DataFrame into a question and answer format.
    
    Parameters:
    row (pd.Series): A row of the DataFrame.
    question_key (str): The key/column name for the question in the DataFrame.
    ans_key (str): The key/column name for the answer in the DataFrame.
    
    Returns:
    str: The formatted question and answer text.
    """
    intro = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
    question_block = f"### Question:\n{row[question_key]}\n"
    answer_block = f"### Answer:\n{row[ans_key]}"
    outro = "\n###end"
    return intro + question_block + answer_block + outro

#-----------------------Preprocessing FinGPT/fingpt-fiqa_qa--------------------------------------------
def Preprocessing_fingpt(dataset):
    df = pd.DataFrame(dataset)
    df['text'] = df.apply(lambda x: format_row(x, 'input', 'output'), axis=1)
    df = df[['text']]
    return df


#------------------------preprocess bavest/fin-llama-dataset-------------------------------------------
def preprocess_finllama(dataset):
    df = pd.DataFrame(dataset)
    df['text'] = df.apply(lambda x: format_row(x, 'instruction', 'output'), axis=1)
    df = df[['text']]
    return df

#-----------------------preprocess nihiluis/financial-advisor-100---------------------------------------------
def preprocess_advisor(dataset):
    df = pd.DataFrame(dataset)
    df['text'] = df.apply(lambda x: format_row(x, 'question', 'answer'), axis=1)
    df = df[['text']]
    return df
#------------------------preprocess gbharti/finance-alpaca--------------------------------------------
def preprocess_alpaca(dataset):
    df = pd.DataFrame(dataset)
    df['text'] = df.apply(lambda x: format_row(x, 'instruction', 'output'), axis=1)
    df = df[['text']]
    return df


#--------------------------------------------------------------------
# Process fingpt:
fingpt = load_dataset("FinGPT/fingpt-fiqa_qa", split='train')
fingpt_processed = Preprocessing_fingpt(fingpt)
print(len(fingpt_processed))

#Process Alpaca:
alpaca = load_dataset("gbharti/finance-alpaca", split='train')
alpaca_processed = preprocess_alpaca(alpaca)
print(len(alpaca_processed))
print(alpaca_processed.iloc[0])

# Process Guanaco
fin_llama = load_dataset("bavest/fin-llama-dataset", split='train')
fin_llama_processed = preprocess_finllama(fin_llama)
print(len(fin_llama_processed))
print(fin_llama_processed.iloc[0])

#Process Advice
ad = load_dataset("nihiluis/financial-advisor-100", split='train')
ad_processed = preprocess_advisor(ad)
print(len(ad_processed))
print(ad_processed.iloc[0])


combined = pd.concat([fingpt_processed, alpaca_processed, fin_llama_processed, ad_processed], ignore_index=True)
print(len(combined))
combined.to_csv('FinQA.csv', index=False)