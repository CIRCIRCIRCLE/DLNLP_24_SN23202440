from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# ======================================================================================================================
# Task A
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

def clean_text(text):
    text = re.sub(r'###endofgeneration', '', text)
    return text

repo_name = "circircircle/FinDPO-Phi2"
model = AutoModelForCausalLM.from_pretrained(repo_name)
tokenizer = AutoTokenizer.from_pretrained(repo_name)
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda:1") 
model = model.to(device)
model.eval()

SAMPLE_PROMPTS = [
    "What are some signs that the stock market might crash?",
    "What's the impact of NLP in the finance area?",
    "Can you give me some investment advice?",
    "Do I need a business credit card?",
    "How do insurance funds work?",
    "Should we invest some of our savings to protect against inflation?",
    "What are the best restaurants in LONDON?",
    "What are some good situation action outcome questions to ask a Data engineer?"
]


for question in SAMPLE_PROMPTS:
    input_text = '###Question:\n' + question + '\n###Answer:\n'
    # Tokenize the input text, and prepare the inputs and attention mask
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Move the tokenized inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # Use the generated attention mask
        max_length=256,
        num_return_sequences=1,
        repetition_penalty=1.2,
        temperature=0.4,
        top_k=30,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned_ans = clean_text(answer)
    print(cleaned_ans)


# ======================================================================================================================
# Task B
# cd B_RAG
# python app.py
