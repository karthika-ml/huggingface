import torch
from huggingface_hub import login
from datasets import load_dataset, DatasetDict, Dataset
import numpy as np

# Github token
login("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")

# Use GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# load dataset
imdb_dataset = load_dataset("imdb")

# this has almost 50,000 samples. 
# In the interest of saving run-time, I truncate

# truncate to 1000 samples
# define subsample size
N = 1000 
# # generate indexes for random subsample
rand_idx = np.random.randint(24999, size=N) 

# # extract train and test data
x_train = imdb_dataset['train'][rand_idx]['text']
y_train = imdb_dataset['train'][rand_idx]['label']

x_test = imdb_dataset['test'][rand_idx]['text']
y_test = imdb_dataset['test'][rand_idx]['label']

# # create new dataset
dataset = DatasetDict({'train':Dataset.from_dict({'label':y_train,'text':x_train}),
                             'validation':Dataset.from_dict({'label':y_test,'text':x_test})})

dataset.save_to_disk('./my_dataset')

# then load the dataset
# dataset =load_dataset('/Users/karthi/Projects/chatbot_projects/finetune_llm/my_imdb_dataset')
dataset.push_to_hub("karthika-ml/my_truncated_imdb")
print(dataset)
