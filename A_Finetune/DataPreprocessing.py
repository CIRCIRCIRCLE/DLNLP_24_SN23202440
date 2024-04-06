
'''
General Datasets: databricks/databricks-dolly-15k | yahma/alpaca-cleaned | timdettmers/openassistant-guanaco

Unify the data set into the following format:
    ### Question:
    What athlete created the 'beast quake' for the...
    ### Answer:
    Marshawn Lynch

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

#-----------------------Preprocessing_Dolly--------------------------------------------
def Preprocessing_Dolly(dataset):
    df = pd.DataFrame(dataset)
    df['text'] = df.apply(lambda x: format_row(x, 'instruction', 'response'), axis=1)
    df = df[['text']]
    return df


#------------------------preprocess_alpaca--------------------------------------------
def preprocess_alpaca(dataset):
    df = pd.DataFrame(dataset)
    df['text'] = df.apply(lambda x: format_row(x, 'instruction', 'output'), axis=1)
    df = df[['text']]
    return df

#-----------------------preprocess_guanaco---------------------------------------------
def format_row_dialog(text):
    """Formats a text containing dialogues into a question and answer format."""
    parts = re.split(r'### Human: |### Assistant: ', text)
    parts = [part.strip() for part in parts if part]
    
    formatted_text = ""
    for i in range(0, len(parts), 2):
        question = parts[i]
        answer = parts[i + 1] if i + 1 < len(parts) else ""
        formatted_text += f"### Question:\n{question}\n\n### Answer:\n{answer}\n"

    intro = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
    outro = "\n###end"
    return intro + formatted_text + outro

def preprocess_guanaco(dataset):
    """Preprocesses the given dataset."""
    texts = [example['text'] for example in dataset]
    formatted_texts = [format_row_dialog(text) for text in texts]
    df = pd.DataFrame({'text': formatted_texts})
    
    return df



#--------------------------------------------------------------------
# Process Dolly:
dolly = load_dataset("databricks/databricks-dolly-15k", split='train')
dolly_processed = Preprocessing_Dolly(dolly)
print(len(dolly_processed))
print(dolly_processed.iloc[0])

#Process Alpaca:
alpaca = load_dataset("yahma/alpaca-cleaned", split='train')
alpaca_processed = preprocess_alpaca(alpaca)
print(len(alpaca_processed))
print(alpaca_processed.iloc[0])

# Process Guanaco
guanaco = load_dataset("timdettmers/openassistant-guanaco", split='train')
guanaco_processed = preprocess_guanaco(guanaco)
print(len(guanaco_processed))
print(guanaco_processed.iloc[0])

combined = pd.concat([dolly_processed, alpaca_processed, guanaco_processed], ignore_index=True)
print(len(combined))
combined.to_csv('../Datasets/generalQA.csv', index=False)