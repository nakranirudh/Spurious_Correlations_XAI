import pickle
import torch
from utils.data_structure import get_sst2, get_movie_rationales, get_IMDB
from transformers import AutoTokenizer

# Load the 3 split data
# parse through and generate the dictionaries
# Save in current directory

size_idx = 1
dataset_idx = 1
tech_idx = 0

techniques = ["LIME/"]
datasets = ["Movie_Rationales", "SST", "Yelp"]
data_sizes = ["100", "7500"]
folder_name = ["", "Big/"]

gpu0_file = "Attributions/" + techniques[tech_idx] + folder_name[size_idx] + datasets[dataset_idx] + "/list_" + data_sizes[size_idx] + "samples_GPU_trainidx_trainlabel_attributes_0.pkl"
gpu1_file = "Attributions/" + techniques[tech_idx] + folder_name[size_idx] + datasets[dataset_idx] + "/list_" + data_sizes[size_idx] + "samples_GPU_trainidx_trainlabel_attributes_1.pkl"
gpu2_file = "Attributions/" + techniques[tech_idx] + folder_name[size_idx] + datasets[dataset_idx] + "/list_" + data_sizes[size_idx] + "samples_GPU_trainidx_trainlabel_attributes_2.pkl"

print(gpu0_file)
print(gpu1_file)
print(gpu2_file)

with open(gpu0_file, 'rb') as f:
    data0 = pickle.load(f)

with open(gpu1_file, 'rb') as f:
    data1 = pickle.load(f)

with open(gpu2_file, 'rb') as f:
    data2 = pickle.load(f)

sst_df = get_sst2()
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

word2tuple = {} # contains a mapping from word 2 tuple {word : ([list(scores), list(idx)])}
senIdx2word = {} # contains a mapping from idx 2 word


split_ratio=int(sst_df.shape[0]/3)
split_1=split_ratio
split_2=split_ratio*2
split_3=split_ratio*3

split_idxs = [0, split_1, split_2, split_3]
all_data = [data0, data1, data2]

# Process each dataset split
for i in range(len(all_data)):
	curr_data = all_data[i]
	curr_split_idx = split_idxs[i]
	# Process each sentence from data split i
	for j in range(len(curr_data)):
		# Extract curr example 
		curr_example = curr_data[j]
		curr_GPU = curr_example[0][0]
		curr_sentence_idx = curr_example[1][0]
		curr_label = curr_example[2][0]	
		curr_saliency = curr_example[3][0].tolist()

		# tokenize the current current sentence
		true_dataset_idx = curr_split_idx + j # due to the way that we chunk the dataset into 3
		curr_input_ids = tokenizer(sst_df['text'][true_dataset_idx],truncation=True,add_special_tokens=False,return_tensors='pt')['input_ids']
		indices = curr_input_ids[0].detach().tolist()
		curr_tokens = tokenizer.convert_ids_to_tokens(indices)

		if len(curr_tokens) != len(curr_saliency):
			print("there is a mismatch between the sizes of the salient tokens and tokenized strings")

		for tok_idx,tok in enumerate(curr_tokens):
			# populate the word2tuple dict
			if tok not in word2tuple:
				# If the word is not in dict, then map the word 2 tuple (list(score),list(sen idx))
				word2tuple[tok] = ([curr_saliency[tok_idx]],[curr_sentence_idx])

			else:
				#otherwise just append
				score_list,sen_idx_list = word2tuple[tok]
				score_list.append(curr_saliency[tok_idx])
				sen_idx_list.append(curr_sentence_idx)
				word2tuple[tok] = (score_list,sen_idx_list)

			if true_dataset_idx not in senIdx2word:
				# if sen idx is not in dict, then map the idx 2 list(word)
				senIdx2word[true_dataset_idx] = [tok]
			elif tok not in senIdx2word[true_dataset_idx]:
				# only add token to the list if not already in the list
				tok_list = senIdx2word[true_dataset_idx]
				tok_list.append(tok)
				senIdx2word[true_dataset_idx] = tok_list

# print(word2tuple['the'])
print(word2tuple['remains'])
print(senIdx2word[0])





