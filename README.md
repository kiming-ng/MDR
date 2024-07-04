# MDR
Code for NAACL 2024 paper: [MDR: Model-Specific Demonstration Retrieval at Inference Time for In-Context Learning](https://aclanthology.org/2024.naacl-long.235/).

## Environment Setup
```bash
cd MDR
bash install.sh 
```

## Preparation
Follow the instructions in [UPRISE](https://github.com/microsoft/LMOps/tree/main/uprise#1-download-retriever-and-prompt-pool) to download pre-trained retriever and pre-constructed demonstration pool. 

After downloading, encode the demonstration pool with the demonstration encoder:
```bash
bash ./scripts/gen_demonstration_embeds.sh
```

## Quick Start
Download [demonstration_pool_GPTNeo.json](https://drive.google.com/file/d/1m4ls7Unl36-NaGCLyKPAUtqeMAZJcb6J/view?usp=drive_link) to `./demonstration_pools`. Then run the provided shell to evaluate MDR on different tasks with GPTNeo-2.7B and get to know the demonstration retrieval process:

```bash
bash ./scripts/run_GPTNeo_2.7B.sh
```

You can change the variable `DEMONSTRATION_POOL` to `path_to_demonstration_pool` (downloaded from UPRISE) to see how MDR calculate eigenvalue and loss for each sample in test dataset given specific inference model.

## Evaluation MDR on any tasks and models
Customize your scripts to support different tasks and models based on the parameters:
- `LLM`: you can specify the LLM name here (in huggingface format);
- `DEMONSTRATION_POOL`: since the calculation of eigenvalue and loss has a one-to-one correspondence with the model, you should create different demonstration pool files for different models (just copy the downloaded demonstration pool file and rename it);
- `TASKS`: MDR support 20+ datasets, you can specify the task name to evaluate according to the task definition in `./DPR/dpr/utils/tasks.py`;
  

## Acknowledgement
This repository is built using the [UPRISE](https://github.com/microsoft/LMOps/tree/main/uprise) codebase.