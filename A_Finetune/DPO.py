import os
import gc
import torch

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from trl import DPOTrainer
import bitsandbytes as bnb

#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#torch.cuda.set_device(device)

#Load data and model
peft_model_name = "circircircle/FinQA-phi2"
new_model ="FinDPO-Phi2" 

tokenizer = AutoTokenizer.from_pretrained(peft_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

#############################################################################
#Dataset Preparation
#############################################################################
def chatml_format(example):
    # Format system
    if len(example['system']) > 0:
        message = {"role": "system", "content": example['system']}
        system = tokenizer.apply_chat_template([message], tokenize=False)
    else:
        system = ""

    # Format instruction
    message = {"role": "user", "content": example['question']}
    prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

    # Format chosen answer
    chosen = example['chosen'] + "<|im_end|>\n"

    # Format rejected answer
    rejected = example['rejected'] + "<|im_end|>\n"

    return {
        "prompt": system + prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

# Load dataset
dataset = load_dataset("Intel/orca_dpo_pairs")['train']

# Save columns
original_columns = dataset.column_names

# Format dataset
dataset = dataset.map(
    chatml_format,
    remove_columns=original_columns
)

data = dataset.to_pandas()
print(data.head())
print(dataset[1])

#############################################################################
# Train the model with DPO
#############################################################################
'''
At a high level, we need to initialize the DPOTrainer with the model we wish to train, 
a reference model (ref_model) which we will use to calculate the implicit rewards for the preferred and rejected responses. 
The parameter 'beta' refers to the hyperparameter of the implicit reward, 
typically set in the range of 0.1 to 0.5. Note that as beta approaches 0, we tend to ignore the reference model.
'''
#LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
)

#Load a pretrained model to fine-tune
model = AutoModelForCausalLM.from_pretrained(
    peft_model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True
)
model.config.use_cache = False

#Reference model
ref_model = AutoModelForCausalLM.from_pretrained(
    peft_model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True
)

#Tokenizer
tokenizer = AutoTokenizer.from_pretrained(peft_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Initialize Training arguments
# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    max_steps=100,
    save_strategy="no",
    logging_steps=1,
    output_dir=new_model,
    optim="paged_adamw_32bit",
    warmup_steps=100,
    bf16=True,
    report_to="wandb",
)


#Initialize DPO trainer
dpo_trainer = DPOTrainer(
    model,
    ref_model = None,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    beta=0.1,
    max_prompt_length=512,
    max_length=1024,
)

# # Fine-tune model with DPO
dpo_trainer.train()


################################################################################
#Save and upload the model
#################################################################################
# Save artifacts
dpo_trainer.model.save_pretrained("final_checkpoint")
tokenizer.save_pretrained("final_checkpoint")

# Flush memory
del dpo_trainer, model
gc.collect()
torch.cuda.empty_cache()

base_model = AutoPeftModelForCausalLM.from_pretrained(
    peft_model_name, 
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(peft_model_name)

# Merge base model with the adapter
model = PeftModel.from_pretrained(base_model, "final_checkpoint")
model = model.merge_and_unload()

output_merged_dir = "results/dpo/final_merged_checkpoint"
os.makedirs(output_merged_dir, exist_ok=True)

# Save model and tokenizer
model.save_pretrained(output_merged_dir, safe_serialization=True)
tokenizer.save_pretrained(output_merged_dir)

# Push them to the HF Hub
model.push_to_hub('circircircle/FinDPO-Phi2')
tokenizer.push_to_hub('circircircle/FinDPO-Phi2')
