# RLHF
This is a repository showcasing finetuning of small LLMs like GPT-2 using reinforcement learning from human feedback (RLHF) to perform pun generation.


## Project Structure
```
src # Source directory, housing all source code
  |_configs.py #Configurations
  |_dataset.py #Dataset Definitions
  |_evaluate.py #Evaluates Generation with ChatGPT
  |_gpt.py #GPT-2 + LoRA implementation
  |_loss.py #Loss functions
  |_test.ipynb #Testing Code
  |_tokenizer.py #Tokenizers
  |_train_dpo.py #DPO Training Script
  |_train_ppo.py #PPO Training Script
  |_train_rm.py #Reward Model Training Script
  |_train_sft.py #SFT Training Script
  |_trainers.py #Training Loops + Other Trainers

res # Resouces directory, housing training data and results
  | //TODO Add files in here when they are populated

requirements.txt # library requirements to be installed with either pip or conda. Note: pytorch installation method may vary by platform

```

## Getting Started
Switch onto the jp/RLHF_pipeline_base branch on this repository
//TODO Add installation and execution instructions

## Youtube Video
//TODO Add link to unlisted youtube video

## Written Report
//TODO Add link to finalized report

## Acknowledgements 
//TODO Add sources (Minimum 10)