import tqdm
import json
import random
import os
import sys 
sys.path.append("..")
from dpr.utils.tasks import task_map
import torch
from torch.utils.data import DataLoader
from src.data.collators import DataCollatorWithPaddingAndCuda
import hydra.utils as hu 
import hydra
from omegaconf import OmegaConf
from src.utils.cache_util import BufferedJsonWriter, BufferedJsonReader
from src.utils.metric import metric_dict
from accelerate import Accelerator
import glob
import logging
from transformers import  AutoModelForCausalLM
logger = logging.getLogger(__name__)


class Scorer:
    def __init__(self,cfg, model, accelerator, prompt_list) -> None:
        self.dataset_reader = hu.instantiate(config=cfg.dataset_reader, prompt_list=prompt_list)
        self.dataset_reader.shard(accelerator)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer,device=accelerator.device)
        self.dataloader = DataLoader(self.dataset_reader,batch_size=cfg.batch_size,collate_fn=co)
        self.dataset_reader.tokenizer.pad_token_id = self.dataset_reader.tokenizer.eos_token_id
        self.accelerator = accelerator        
        self.model = model.eval()
        self.cfg = cfg
        self.tokenizer=self.dataset_reader.tokenizer
        self.max_length=cfg.dataset_reader.max_length #used for text completion task,
        self.generate_max_len=cfg.generate_max_len # max seq len to be generated

        
    def choice_losses(self,input_ids,input_atten_mask,loss_mask,labels,locate_prompt):
        bsz, option_num, seq_len = input_ids.shape
        self.model.zero_grad()
        output =self.model(input_ids=input_ids.reshape(bsz*option_num, seq_len), 
                              attention_mask=input_atten_mask.reshape(bsz*option_num, seq_len),\
                                output_hidden_states=True,return_dict=True)

        logits = output.logits.reshape(bsz, option_num, seq_len, -1) # (bsz, option_num, seq_len, hidden_dim)
        logits = logits[:,:, :-1, :] # (bsz, option_num, seq_len-1, hidden_dim)
        targets = input_ids[:,:,1:].unsqueeze(-1) # (bsz,option_num, seq_len-1, 1)
        logit_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1) # (bsz, option_num, seq_len-1, hidden_dim)
        loss_mask=loss_mask[:,:,1:] # (bsz, option_num, seq_len-1)
        loss= -torch.gather(logit_probs, -1, targets).squeeze(-1) * loss_mask  # (bsz, option_num, seq_len-1) 
        loss = loss.sum(-1) / loss_mask.sum(-1) # (bsz, option_num)
        preds= torch.argmin(loss,dim=-1)

        # calculate max eigen of fisher matrix
        max_eigenvalue_list = []
        for i in range(loss.shape[0]):
            j = preds[i]
            sel_layer = len(output.hidden_states)-1 # select last layer
            gradients = torch.autograd.grad(loss[i,j],output.hidden_states[sel_layer],retain_graph=True)
            gradients = gradients[0].reshape(bsz, option_num, seq_len, gradients[0].shape[2])
            if locate_prompt is not None:
                tot = locate_prompt[i]
                fisher = gradients[i, j, :tot, :] # Take only the prompt part 
            else:
                fisher = gradients[i, j, :, :] # Take the full input sequence 
            J = torch.cat([e.flatten() for e in fisher])
            eigen = 0.0
            for i in range(J.size(0)):
                eigen+=J[i]**2
            assert eigen != 0.0 or eigen != 0
            max_eigenvalue_list.append(float(eigen))
        
        normed_loss = torch.nn.functional.normalize(loss, p=1,dim=-1)
        labels_losses = torch.gather(normed_loss, -1, labels).squeeze(-1).tolist()
        accurate_list=(preds==labels.squeeze(-1)).int().tolist()
        
        return  {
                "labels_losses": labels_losses,
                "loss": labels_losses,
                "accurate_list": accurate_list,
                "preds": preds.tolist(),
                "max_eigenvalue": max_eigenvalue_list
                }

    def completion_losses(self,input_ids,input_atten_mask,labels,input_ids_eigen,input_atten_mask_eigen,input_loss_mask_eigen,locate_prompt):
        self.model.zero_grad()
        answer_start = int(input_atten_mask.shape[-1]) 
        input_ids_eigen=input_ids_eigen.squeeze(1)
        output = self.model(input_ids=input_ids_eigen,
                            attention_mask=input_atten_mask_eigen.squeeze(1),                            
                            output_hidden_states=True,return_dict=True)
        res = self.model.generate(input_ids=input_ids.squeeze(1), #remove the dim for option_num
                                    attention_mask=input_atten_mask.squeeze(1),
                                    eos_token_id=self.dataset_reader.tokenizer.encode("\n")[0],
                                    pad_token_id=self.dataset_reader.tokenizer.pad_token_id,
                                    max_length=min(self.max_length,answer_start+self.generate_max_len),
                                    do_sample=False)
        
        loss_mask=input_loss_mask_eigen.squeeze(1)
        # For completion loss
        logits = output.logits[:,:-1, :]   # (bsz, seq_len-1, hidden_dim)
        targets = input_ids_eigen[:,1:].unsqueeze(-1) # (bsz,seq_len-1, 1)
        logit_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1) # (bsz, seq_len-1,hidden_dim)
        loss_mask=loss_mask[:,1:] # (bsz, seq_len-1)
        loss= -torch.gather(logit_probs, -1, targets).squeeze(-1) * loss_mask  #  (bsz, seq_len-1) 
        loss = loss.sum(-1) / loss_mask.sum(-1) # (bsz)
        
        # calculate max eigen of fisher matrix
        max_eigenvalue_list = []
        for i in range(loss.shape[0]):
            sel_layer = len(output.hidden_states)-1
            gradients = torch.autograd.grad(loss[i],output.hidden_states[sel_layer],retain_graph=True)
            gradients = gradients[0]
            if locate_prompt is not None:
                tot = locate_prompt[i]
                fisher = gradients[i, :tot, :] # Take only the prompt part 
            else:
                fisher = gradients[i, :, :] # Take the full input sequence                    
            J = torch.cat([e.flatten() for e in fisher])  
            eigen = 0.0
            for a in range(J.size(0)):
                eigen+=J[a]**2
            max_eigenvalue_list.append(float(eigen))
                        
        pred_ids=res[:,answer_start:]
        preds=[]
        for i in range(len(pred_ids)):
            pred=self.dataset_reader.tokenizer.decode(pred_ids[i],skip_special_tokens=True)
            # avoid empty prediction to avoid errors when calculating Rouge metric scores
            if '\n' not in pred: pred+='\n' 
            preds.append(pred)
        compute_metric=metric_dict[self.dataset_reader.task.metric]
        scores=compute_metric(preds=preds, labels=labels, return_list=True)
        return  {
                "labels_losses": [1-score for score in scores],
                "accurate_list": scores,
                "preds": preds,
                "loss": loss.tolist(),
                "max_eigenvalue": max_eigenvalue_list
                }
    
    
    def forward(self):
        
        if self.accelerator.is_main_process:
            dataloader = tqdm.tqdm(self.dataloader)
        else:
            dataloader = self.dataloader

        metadata_list = []
        for i,entry in enumerate(dataloader): # len(prompt_list)
            metadata = entry.pop("metadata")
            if task_map.cls_dic[metadata[0]['task_name']]().class_num==1:
                    one_shot_res=self.completion_losses(
                    input_ids=entry.input_ids,
                    input_atten_mask=entry.input_atten_mask,
                    labels=[x.pop('temp_label') for x in metadata],
                    input_ids_eigen=entry.input_ids_eigen,
                    input_atten_mask_eigen=entry.input_atten_mask_eigen,
                    input_loss_mask_eigen=entry.input_loss_mask_eigen,
                    locate_prompt = None
                )
            else:
                one_shot_res=self.choice_losses(
                    input_ids=entry.input_ids,
                    input_atten_mask=entry.input_atten_mask,
                    loss_mask=entry.input_loss_mask,
                    labels=entry.labels,
                    locate_prompt = None
                )
            one_shot_labels_losses=one_shot_res["labels_losses"]
            one_shot_losses=one_shot_res["loss"]
            for i in range(len(metadata)):
                metadata[i]['pred']=one_shot_res["preds"][i]
                metadata[i]['labels_loss']=one_shot_labels_losses[i]
                metadata[i]['loss']=one_shot_losses[i]
                metadata[i]['one_shot_acc']=one_shot_res["accurate_list"][i]
                metadata[i]['max_eigenvalue']=one_shot_res["max_eigenvalue"][i]
                metadata[i]['eigen']= one_shot_res["max_eigenvalue"][i]
            metadata_list.append(metadata[i])
        
        return metadata_list

    