# Import necessary packages for the fine-tuning
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    TrainerCallback,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import logging

# Configure logging to write to file
output_dir = "/content/drive/My Drive/LLAMA"

log_file_path = os.path.join(output_dir, "training.log")
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# The model that we want to train from the Hugging Face hub
model_name = "NousResearch/Llama-2-7b-hf"

# The Alpaca dataset to use
dataset_name = "tatsu-lab/alpaca"

new_model = os.path.join(output_dir, "llama-2-7b-alpaca")

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 8

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.05

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Number of training epochs
num_train_epochs = 3

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 2

# Batch size per GPU for evaluation
per_device_eval_batch_size = 2

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule (constant a bit better than cosine)
lr_scheduler_type = "constant"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 1000

# Log every X updates steps
logging_steps = 10

logging_dir = os.path.join(output_dir, "logs")

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = 256

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}

# Step 1 : Load dataset (you can process it here)
dataset = load_dataset(dataset_name, split="train")

dataset[0]

# Step 2 :Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Step 3 :Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Step 4 :Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Step 5 :Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Step 6 :Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
)

# Configure logging
hyperparams = {
    "model_name": model_name,
    "dataset_name": dataset_name,
    "new_model": new_model,
    "lora_r": lora_r,
    "lora_alpha": lora_alpha,
    "lora_dropout": lora_dropout,
    "use_4bit": use_4bit,
    "bnb_4bit_compute_dtype": bnb_4bit_compute_dtype,
    "bnb_4bit_quant_type": bnb_4bit_quant_type,
    "use_nested_quant": use_nested_quant,
    "num_train_epochs": num_train_epochs,
    "fp16": fp16,
    "bf16": bf16,
    "per_device_train_batch_size": per_device_train_batch_size,
    "per_device_eval_batch_size": per_device_eval_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "gradient_checkpointing": gradient_checkpointing,
    "max_grad_norm": max_grad_norm,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "optim": optim,
    "lr_scheduler_type": lr_scheduler_type,
    "max_steps": max_steps,
    "warmup_ratio": warmup_ratio,
    "group_by_length": group_by_length,
    "save_steps": save_steps,
    "logging_steps": logging_steps,
    "max_seq_length": max_seq_length,
    "packing": packing,
    "output_dir": output_dir,
    "name": "Rochan Madhusudhana",
    "gtid": "903535969",
}


class LoggingCallback(TrainerCallback):
    def __init__(self):
        self.hyperparams_logged = False
        self.hyperparams = hyperparams

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.hyperparams_logged:
            with open(os.path.join(output_dir, "training.log"), "a") as log_file:
                log_file.write(f"Hyperparameters: {self.hyperparams}\n")
                log_file.write(f"Name: {self.hyperparams['name']}\n")
                log_file.write(f"GTID: {self.hyperparams['gtid']}\n")
            self.hyperparams_logged = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % 10 == 0:  # Log every 10 steps
            with open(os.path.join(output_dir, "training.log"), "a") as log_file:
                log_file.write(f"Step: {state.global_step}, Loss: {logs['loss']}\n")


# Step 7 :Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
    callbacks=[LoggingCallback()],
)

# Step 8 :Train model
# TODO: If you want to resume training from a checkpoint, set resume_from_checkpoint=True
trainer.train(resume_from_checkpoint=False)

# Step 9 :Save trained model
trainer.save_model(new_model)
