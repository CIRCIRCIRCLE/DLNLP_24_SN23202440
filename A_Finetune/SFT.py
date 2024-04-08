import argparse
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
from huggingface_hub import notebook_login
seed = 42
set_seed(seed)

from huggingface_hub import login
import os

'''
make sure to inter your HuggingFace token in the env using 'export HF_ACCESS_TOKEN=hf_**********'
run the code using the script:
    python sft2.py --model_name "circircircle/GeneralQA-phi2" --dataset "circircircle/FinQA" --repo_name "circircircle/FinQA-phi2"

    model_name represents the pretrained model; the repo_name represents the trained model.
    
'''
access_token = os.getenv("HF_ACCESS_TOKEN")

if access_token is not None:
    login(token=access_token, add_to_git_credential=True)
else:
    print("Hugging Face access token not found. Please set it as an environment variable.")
    
##############################################################################
# dataset preparation
##############################################################################
def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing text
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )

def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset: str):
    """
    Tokenize the dataset
    """
    print("Preprocessing dataset...")
    _preprocessing_function = partial(preprocess_batch, tokenizer=tokenizer, max_length=max_length)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["text"],  
    )
    

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    # Split datasets
    train_val_split = dataset["train"].train_test_split(test_size=0.1)
    datasets = {
        "train": train_val_split["train"],
        "val": train_val_split["test"]
    }

    return datasets

#############################################################
#load model
#############################################################
def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = f'{40960}MB'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length

##########################################################
#QLoRA configuration
##########################################################
def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config


def create_peft_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
        r=256,           # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=modules,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    return config


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

#############################################################
#Launch training
################################################################
def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )

def train(model, tokenizer, dataset_train, dataset_val, output_dir):
    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # Get lora module names
    modules = find_all_linear_names(model)

    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(modules)
    model.add_adapter(peft_config)
    model = get_peft_model(model, peft_config)

    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)

    # Training parameters
    trainer = Trainer(
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=30,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit",
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs

    ### SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
    # Verifying the datatypes before training
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)

    do_train = True

    # Launch training
    print("Training...")

    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)

    # Saving model
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with QLoRA and BitsAndBytes.')
    # Parse arguments
    parser.add_argument('--model_name', type=str, default='microsoft/phi-2', help='Model name or path')
    parser.add_argument('--dataset', type=str, default='circircircle/FinQA', help='Dataset to use for training')
    parser.add_argument('--repo_name', type=str, default='circircircle/FinQA-phi2', help='Repository name for Hugging Face Hub')

    
    args = parser.parse_args()

    # Use arguments
    model_name = args.model_name  #'microsoft/phi-2'  'circircircle/GeneralQA-phi2' 
    dataset_name = args.dataset   #'circircircle/generalQA'   'circircircle/FinQA'
    repo_name = args.repo_name    #'circircircle/GeneralQA-phi2'  'circircircle/FinQA-phi2'

    #-----------------------------------------------------------------------------------
    # Load dataset from Hugging Face
    dataset = load_dataset(dataset_name)
    print(f'Number of prompts: {len(dataset)}')
    print(f'Column names are: {dataset.column_names}')

    # Load model
    bnb_config = create_bnb_config()
    model, tokenizer = load_model(model_name, bnb_config)

    ## Preprocess dataset
    max_length = get_max_length(model)
    datasets = preprocess_dataset(tokenizer, max_length, seed, dataset)

    # Train and save the model params
    output_dir = "results/phi2/final_checkpoint"
    train(model, tokenizer, datasets["train"], datasets["val"], output_dir)

    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)

    model = model.merge_and_unload()
    output_merged_dir = "results/phi2/final_merged_checkpoint"
    os.makedirs(output_merged_dir, exist_ok=True)
    model.save_pretrained(output_merged_dir, safe_serialization=True)

    # save tokenizer for easy inference
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_merged_dir)
    
    # push the fintuned model to huggingface
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)