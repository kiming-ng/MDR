from typing import Any, Dict
from transformers import AutoTokenizer
import torch
import pandas as pd
import json
import pandas as pd
from datasets import Dataset
import json
import re
from src.utils.dataset_utils import pad2sameLen
from dpr.utils.tasks import task_map

def remove_double_space(string):
    return re.sub("[ ]{2,}", " ", string)


class ScorerDatasetReader(torch.utils.data.Dataset):
    def __init__(
        self,
        model_name=None,
        cache_dir=None,
        max_length=2048,
        prompt_list=None
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, model_max_length=max_length
        )        

        # prompt_pool
        self.prompt_pool = prompt_list
        
        def get_instance(entry):
            if entry.meta_data['task_name'] == 'multirc':
                entry.meta_data['idx'] = 0
            if entry.meta_data['task_name'] == 'hellaswag':
                entry.meta_data['label'] = int(entry.meta_data['label'])
            if entry.meta_data['task_name'] == 'natural_questions':
                entry.meta_data['answer'] = entry.meta_data['answer'][0]
            examples = [entry.meta_data]
            yield from examples

        def get_dataset(data):
            for entry in data:
                yield from get_instance(entry)

        df = pd.DataFrame(list(get_dataset(self.prompt_pool)))
        self.dataset = Dataset.from_pandas(df)

    def shard(self, accelerator):
        self.dataset = self.dataset.shard(
            num_shards=accelerator.num_processes, index=accelerator.process_index
        )

    def __getitem__(self, index):       
        return self.text_to_instance(self.dataset[index])

    def __len__(self):
        return len(self.dataset)

    def get_fields(self, entry):
        task = task_map.cls_dic[entry['task_name']]()
        test_input_strs = task.get_input_strs(entry)
        test_questions = [input for input in test_input_strs]
        test_answer_strs = task.get_answers(entry)
        test_label = task.get_label(entry)
        # demonstration = f'{question}{answer}'
        
        return test_questions, test_answer_strs, test_label
    
    def text_to_instance(self, entry):
        self.task = task_map.cls_dic[entry['task_name']]()
        if self.task.class_num == 1:  # text completion question
            self.tokenizer.padding_side = "left"
            return self.text_to_instance_completion(entry)
        else:
            return self.text_to_instance_choice(entry)

    def text_to_instance_choice(self, entry):
        """
        multiple-choice question
        """
        test_questions, test_answers, test_label = self.get_fields(entry)  

        input_ids_list = []
        input_atten_mask_list = []
        input_loss_mask_list = []
        # locate_prompt = []        

        for i in range(len(test_questions)):
            enc_text = remove_double_space(test_questions[i] + test_answers[i])
            enc_answer = remove_double_space(test_answers[i])
            tokenized_example = self.tokenizer.encode_plus(
                enc_text,
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt",
            )
            tokenized_answer = self.tokenizer.encode_plus(
                enc_answer,
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt",
            )

            answer_mask = tokenized_answer.attention_mask.squeeze()
            if len(answer_mask.shape) == 0:
                answer_mask = torch.tensor([1]).to(answer_mask)

            input_ids = tokenized_example.input_ids.squeeze()
            input_atten_mask = tokenized_example.attention_mask.squeeze()
            input_loss_mask = torch.nn.functional.pad(
                answer_mask, (input_ids.shape[-1] - answer_mask.shape[-1], 0)
            )

            input_ids_list.append(input_ids)
            input_atten_mask_list.append(input_atten_mask)
            input_loss_mask_list.append(input_loss_mask)

        return {
            "input_ids": pad2sameLen(
                input_ids_list, pad_idx=self.tokenizer.pad_token_id
            ),
            "input_atten_mask": pad2sameLen(input_atten_mask_list, pad_idx=0),
            "input_loss_mask": pad2sameLen(input_loss_mask_list, pad_idx=0),
            "labels": torch.tensor([test_label]),
            "metadata": entry,
            # "locate_prompt": torch.tensor(locate_prompt)
        }

    def text_to_instance_completion(self, entry: Dict[str, Any]):
        """
        text completion question
        """
        test_questions, test_answers, test_label = self.get_fields(entry)

        input_ids_list = []
        input_atten_mask_list = []
        
        input_ids_list_eigen = []
        input_atten_mask_list_eigen = []
        input_loss_mask_list_eigen = []
        
        for i in range(len(test_questions)): # len(test_questions) = 1 for completion question
            enc_text = remove_double_space(test_questions[i]).strip() 
            tokenized_example = self.tokenizer.encode_plus(
                enc_text,
                truncation=False,
                return_tensors="pt",
                add_special_tokens=False,
            )

            input_ids = tokenized_example.input_ids.squeeze()
            input_atten_mask = tokenized_example.attention_mask.squeeze()

            input_ids_list.append(input_ids)
            input_atten_mask_list.append(input_atten_mask)
            
            # for eigen
            enc_text_eigen = remove_double_space(test_questions[i] + test_answers[i])
            enc_answer_eigen = remove_double_space(test_answers[i])
            tokenized_example_eigen = self.tokenizer.encode_plus(
                enc_text_eigen,
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt",
            )
            tokenized_answer_eigen = self.tokenizer.encode_plus(
                enc_answer_eigen,
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt",
            )

            answer_mask_eigen = tokenized_answer_eigen.attention_mask.squeeze()
            if len(answer_mask_eigen.shape) == 0:
                answer_mask_eigen = torch.tensor([1]).to(answer_mask_eigen)

            input_ids_eigen = tokenized_example_eigen.input_ids.squeeze()
            input_atten_mask_eigen = tokenized_example_eigen.attention_mask.squeeze()
            input_loss_mask_eigen = torch.nn.functional.pad(
                answer_mask_eigen, (input_ids_eigen.shape[-1] - answer_mask_eigen.shape[-1], 0)
            )

            input_ids_list_eigen.append(input_ids_eigen)
            input_atten_mask_list_eigen.append(input_atten_mask_eigen)
            input_loss_mask_list_eigen.append(input_loss_mask_eigen)

        entry["temp_label"] = test_label  # pass label for the next step
        return {
            "input_ids": pad2sameLen(
                input_ids_list, pad_idx=self.tokenizer.pad_token_id, left_pad=True
            ),
            "input_atten_mask": pad2sameLen(
                input_atten_mask_list, pad_idx=0, left_pad=True
            ),
            "input_ids_eigen":pad2sameLen(
                input_ids_list_eigen, pad_idx=self.tokenizer.pad_token_id, left_pad=True
            ),
            "input_atten_mask_eigen":pad2sameLen(
                input_atten_mask_list_eigen, pad_idx=self.tokenizer.pad_token_id, left_pad=True
            ),
            "input_loss_mask_eigen":pad2sameLen(
                input_loss_mask_list_eigen, pad_idx=self.tokenizer.pad_token_id, left_pad=True
            ),
            "metadata": entry,
        }
