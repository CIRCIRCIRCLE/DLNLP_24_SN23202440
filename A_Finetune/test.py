from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

repo_name = "circircircle/GeneralQA-phi2"
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
    inputs = tokenizer(question, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    inputs = inputs.to(device)
  
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=200,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Q: {question}\nA: {answer}\n")
