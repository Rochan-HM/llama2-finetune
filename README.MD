# LLaMA-2 Fine Tuning

## Description
This assignment aims to replicate the results of the Stanford Alpaca utilizing LoRA.

## Requirements
This project has been testing on the following Python versions:
- Python 3.10+

The core requirements are listed in the `requirements.txt` file. They can be installed with the following command:
```bash
pip install -r requirements.txt
```

The core requirements will then install the appropriate dependencies depending on the environment. For example, if you are using a GPU, the GPU version of PyTorch will be installed.


# Training

All the training code is located in the `train.py` file. The training code can be run with the following command:
```bash
python train.py
```

As per the requirements, the model was fine-tuned for 3 epochs (78003 steps).

Please note that some configuration (like the `output_dir`) can be changed in the `training.py` file. Also, please note that `resume_from_checkpoint=False` by default in the `train.py` file. This means that the training will start from scratch. If you want to resume training from a checkpoint, please change this to `True`.

The following base LLaMA-2 model was used: [`NousResearch/Llama-2-7b-hf`](https://huggingface.co/NousResearch/Llama-2-7b-hf). It is a freely available model on the HuggingFace model hub. The alternative official Meta model [`meta-llama/Llama-2-7b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) can also be used. However, it requires one to obtain a license from Meta.

The official dataset from the Stanford Alpaca project was used from HuggingFace. It is available here: [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca/viewer/tatsu-lab--alpaca/train?p=520)

## Log File

A log file was manually created to track the training progress. It is located in the `training.log` file. It contains my name, GTID, the hyperparameters used, and the loss every 10 steps.

## Training Weights

The LoRA weights from this project can be found on Hugging Face Models: [`rochanm/llama-2-7b-alpaca`](https://huggingface.co/rochanm/llama-2-7b-alpaca/tree/main)

## Inference

The inference code is located in the `inference.py` file. The inference code can be run with the following command:
```bash
python inference.py
```

A `process_file()` function is defined in the `inference.py` file. This function can optionally take in an `input_file_name` and `output_file_name` to process a file. If these are not provided, the function will default to using the `test.txt` and `test-output.txt` files in the root of the project.

Sample input and output files are provided in the root of the project.

## Google Colab Link

The training was done on Google Colab. The link to the Colab notebook is here:
https://colab.research.google.com/drive/1bMB0ApuYdriY_zHLhaw_v6yig3lbA4ru?usp=sharing


## Credits

This code was heavily inspired by:
https://github.com/ashishpatel26/LLM-Finetuning/blob/main/2.Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.ipynb
