# Load required libraries
from peft import PeftModel
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

# The base model and our updated weights
base_model_name = "NousResearch/Llama-2-7b-hf"
new_model_name = "rochanm/llama-2-7b-alpaca"

device_map = {"": 0}

# Initialize the model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model_name)
model = model.merge_and_unload()

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Initialize the pipeline
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


def process_file(input_file_name: str, output_file_name: str) -> None:
    with open(input_file_name, "r") as test_file, open(
        output_file_name, "w"
    ) as output_file:
        for line in test_file:
            line = line.strip()
            if line:  # Check if line is not empty
                generated_text = text_generator(line, max_length=50)[0][
                    "generated_text"
                ]
                output_file.write(generated_text + "\n")


if __name__ == "__main__":
    process_file(
        input_file_name="test.txt",
        output_file_name="test-output.txt",
    )
