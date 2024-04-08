from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

def clean_text(text):
    text = re.sub(r'###endoftext', '', text)
    text = re.sub(r'Write an instruction.*', '', text)
    text = re.sub(r'Generate a response.*', '', text)
    text = re.sub(r'###endofquestionandanswer*', '', text)
    text = re.sub(r'###endofresponse*', '', text)
    text = re.sub(r'###endofgeneration|>', '', text)
    text = re.sub(r'<|question|', '', text)
    text = re.sub(r'##Requirement:', '', text) 
    text = re.sub(r'### Task:', '', text) 
    text = re.sub(r'\n+', '\n', text).strip()
    return text

repo_name = "circircircle/FinQA-phi2"
model = AutoModelForCausalLM.from_pretrained(repo_name)
tokenizer = AutoTokenizer.from_pretrained(repo_name)

# Set the tokenizer's pad token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
tokenizer.padding_side = "left"

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
    outputs = model.generate(
        tokenizer(input_text, return_tensors="pt").to(device)['input_ids'],
        #attention_mask=inputs["attention_mask"],
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
    #print(f"Q: {question}\nA: {answer}\n")
    print(cleaned_ans)
