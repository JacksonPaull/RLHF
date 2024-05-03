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

requirements.txt # library requirements to be installed with either pip or conda. Note: pytorch installation method may vary by platform

```

## Getting Started
1. Switch onto the jp/RLHF_pipeline_base branch on this repository
2. Install requirements through pip install -r requirements.txt
3. Train the SFT model with python train_sft.py --n experiment_name -b batch_size
4. Train the reward model with python train_rm.py -b batch_size -n experiment_name -p "./runs/path/to/sft/weights"
5. Start RLHF with the reward and SFT models with python train_rm.py -b batch_size -n experiment_name -a "./runs/path/to/sft/weights" -c "./runs/path/to/reward_model/weights" -s naive

## Youtube Video
//TODO Add link to unlisted youtube video

## Written Report
//TODO Add link to finalized report

## Acknowledgements 
[1] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn
Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jack-
son Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Her-
nandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine
Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann,
and Jared Kaplan. Training a helpful and harmless assistant with reinforcement learning from
human feedback, 2022.
[2] Ting-Rui Chiang and Yun-Nung Chen. Relating neural text degeneration to exposure bias,
2021.
[3] He He, Nanyun Peng, and Percy Liang. Pun generation with surprise, 2019.
[4] Zhenyu Hou, Yilin Niu, Zhengxiao Du, Xiaohan Zhang, Xiao Liu, Aohan Zeng, Qinkai Zheng,
Minlie Huang, Hongning Wang, Jie Tang, and Yuxiao Dong. Chatglm-rlhf: Practices of aligning
large language models with human feedback, 2024.
[5] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang,
Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021.
[6] Yanjia Li. minchatgpt: A minimum example of aligning language models with rlhf similar to
chatgpt. https://github.com/ethanyanjiali/minChatGPT, 2023.
[7] Elham Madjidi and Christopher Crick. A survey on neural text generation and degeneration.
8
Under review for the Reinforcement Learning Conference (RLC)
[8] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language
models are unsupervised multitask learners. 2019.
[9] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al.
Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.
[10] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and
Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model,
2023.
[11] Michael Santacroce, Yadong Lu, Han Yu, Yuanzhi Li, and Yelong Shen. Efficient rlhf: Reducing
the memory usage of ppo, 2023.
[12] Michael Santacroce, Yadong Lu, Han Yu, Yuanzhi Li, and Yelong Shen. Efficient rlhf: Reducing
the memory usage of ppo, 2023.
[13] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal
policy optimization algorithms, 2017.
[14] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal
policy optimization algorithms, 2017.
[15] Simeng Sun, Dhawal Gupta, and Mohit Iyyer. Exploring the impact of low-rank adaptation on
the performance, efficiency, and regularization of rlhf, 2023.
[16] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT
Press, Cambridge, MA, 1998.
[17] Zhiwei Yu, Jiwei Tan, and Xiaojun Wan. A neural approach to pun generation. In Iryna
Gurevych and Yusuke Miyao, editors, Proceedings of the 56th Annual Meeting of the Associ-
ation for Computational Linguistics (Volume 1: Long Papers), pages 1650–1660, Melbourne,
Australia, July 2018. Association for Computational Linguistics.
[18] Chen Zheng, Ke Sun, Hang Wu, Chenguang Xi, and Xun Zhou. Balancing enhancement,
harmlessness, and general capabilities: Enhancing conversational llms with direct rlhf, 2024.