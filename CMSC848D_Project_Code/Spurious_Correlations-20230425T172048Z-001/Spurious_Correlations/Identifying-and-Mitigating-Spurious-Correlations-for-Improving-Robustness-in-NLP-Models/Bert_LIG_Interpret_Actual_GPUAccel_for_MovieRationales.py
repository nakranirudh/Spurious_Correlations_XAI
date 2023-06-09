#!/usr/bin/env python
# coding: utf-8

# Imports 

import utils.data_structure as dataproj
import numpy as np
np.set_printoptions(precision=16,threshold=np.inf)
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForQuestionAnswering, BertConfig
from torch.utils.data.dataloader import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead
from transformers import GlueDataTrainingArguments as DataTrainingArguments
import lime
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
import pickle
# Imports coming from the Distributed Parallel
import os
import torch.distributed as dist
from torch import Tensor
from torch.multiprocessing import Process
# Imports - Captum
from captum.attr import Lime
from captum._utils.models.linear_model import SkLearnLasso
from torch.nn.parallel import DistributedDataParallel as DDP
from captum.attr import LayerIntegratedGradients as LIG
from captum._utils.models.linear_model import SkLearnLasso

# Global Variables

USE_CUDA = True
WORLD_SIZE = 3
model_path = 'randellcotta/distilbert-base-uncased-finetuned-yelp-polarity'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load tokenizer
tokenizer=AutoTokenizer.from_pretrained(model_path)
list_of_attrs = []
count=0


def run(rank, size, inp_batch,label_batch):
	'''Main Multiprocessing function for running on batch'''

	global list_of_attrs
	global count
	model = AutoModelForSequenceClassification.from_pretrained(model_path)
	model.to(device)
	# print(label_batch)
	def fun_predict_captum(inp_batch):
		out_neg = model(inp_batch)
		out_neg=out_neg[0].to(device)
		classfn_out_neg=torch.softmax(out_neg,dim=1)
		return classfn_out_neg

	def summarize_attributions(attributions):
	    attributions = attributions.sum(dim=-1).squeeze(0)
	    attributions = attributions / torch.norm(attributions)
	    # print(attributions)
	    return attributions

	# Move model and input to device with ID rank if USE_CUDA is True
	if USE_CUDA:
		model = model.cuda(rank)
		# Uncomment line below to wrap with DistributedDataParallel
		model = DDP(model, device_ids=[rank])

	for i in range(500):
		inp_batch = list(inp_batch)
		label_batch=list(label_batch)
		curr_inp_batch = inp_batch[i]
		curr_lab_batch = label_batch[i]
		# print('Label',curr_lab_batch)
		cuda_device= f'cuda:{rank}'
		tok_input_batch = tokenizer(curr_inp_batch,truncation=True,add_special_tokens=False,return_tensors='pt').to(cuda_device)
		# print(curr_inp_batch)
		# print(tok_input_batch)
		out_neg = model(tok_input_batch.input_ids)
		out_neg=out_neg[0].to(device)
       	#lime = Lime(fun_predict_captum,interpretable_model=SkLearnLasso(alpha=0.000001))
       	#attr = lime.attribute(tok_input_batch.input_ids, target=curr_lab_batch, n_samples=1000,show_progress=True).detach().cpu() 
       	# LxA attribution
		print("Running the attribution : ",count, "\n")
		count+=1
		# print(model)
		lig = LIG(fun_predict_captum,model.module.distilbert.embeddings)
		attr = lig.attribute(tok_input_batch.input_ids, target=curr_lab_batch,internal_batch_size=10).detach().cpu()
		attr=summarize_attributions(attr).to(torch.float32)
		attr=torch.tensor([attr.tolist()]).to(torch.float32)
		list_of_attrs.append(([rank],[i],[curr_lab_batch],attr))

def init_process(rank, size, fn, inp_batch,label_batch, backend='gloo'):
    '''Initialize processes and call run'''

    global list_of_attrs
    os.environ['MASTER_ADDR'] = '10.229.55.68'
    os.environ['MASTER_PORT'] = '25400' # Figure out later
    dist.init_process_group(backend, rank=rank, world_size=size)
    # print(label_batch)
    fn(rank, size, inp_batch,label_batch)
    # print("list of attrs in init_process: ", list_of_attrs,"\n")
	# Store the data into pickle files
    filename=f'Attributions/LIG/Big/Movie_Rationales/list_LIG_500samples_GPU_trainidx_trainlabel_attributes_{rank}'
    with open(filename+'.pkl','wb') as f:
	    pickle.dump(list_of_attrs,f)
    # # print("We have ran run")
    dist.destroy_process_group()



def main():
	'''Main function which calls init_processes'''

	global list_of_attrs
	# df_yelp=dataproj.get_yelp()
	# df_sst2=dataproj.get_sst2()
	df_movie_rationales=dataproj.get_movie_rationales()

	# choose dataset between yelp,sst,movierationales
		# if choosing text and labels for movierationales
			 # review is the text field , evidences is human annotations
			 # movie rationales is smaller (NOTE) 
	df_chosen=df_movie_rationales.copy()
	# print(df_chosen.shape)

	# print(df_chosen.head(10))

	# Split into 1/3rds for each GPU
	split_ratio=int(df_chosen.shape[0]/3)
	split_1=split_ratio
	split_2=split_ratio*2
	split_3=split_ratio*3
	# print(split_ratio)
	# print(split_1)
	# print(split_2)
	# print(split_3)

	size = WORLD_SIZE
	processes = []

	batch_chunks = (df_chosen['review'][:split_1], df_chosen['review'][split_1:split_2], df_chosen['review'][split_2:split_3])
	label_chunks = (df_chosen['label'][:split_1], df_chosen['label'][split_1:split_2], df_chosen['label'][split_2:split_3])
    #print(len(batch_chunks[0]))
	for rank in range(size):
	    p = Process(target=init_process, args=(rank, size, run, batch_chunks[rank],label_chunks[rank]))
	    p.start()
	    # print(p)
	    # print(p.is_alive())
	    processes.append(p)
	    # print(processes)

	for p in processes:
	    p.join()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    torch.cuda.empty_cache()
    main()

# # Debug

# # Reading in pickle files
# import pickle
# filename=f'Attributions/LIME/SST/list_100samples_GPU_trainidx_trainlabel_attributes_0.pkl'
# with open(filename, 'rb') as f:
#     x = pickle.load(f)
# print(x)
