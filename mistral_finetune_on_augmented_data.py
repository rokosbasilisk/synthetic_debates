#!/usr/bin/env python
# coding: utf-8

# In[1]:


from accelerate import FullyShardedDataParallelPlugin, Accelerator
from datasets import load_dataset
from datetime import datetime
from peft import LoraConfig, get_peft_model
from peft import PeftModel, prepare_model_for_kbit_training
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
import transformers
import random


# In[2]:


base_model_id = "mistralai/Mistral-7B-v0.1"
data_path = "data/debates/debate_data_v2.json"
data_aug_path = "data/debates/augmented_debates.json"

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token


# In[3]:


## dataset prep


# In[4]:


debate_data = json.loads(open(data_path,"r").read())


# In[5]:


augmented_debate_data = json.loads(open(data_aug_path,"r").read())


# In[6]:


debate_prefix = f"""<s>[INST] You are a debate completion agent. Given a debate transcript, your task is to continue the discourse based on the current speaker and the preceding context. The debate begins with a "HUMAN_JUDGE," followed by two adversaries, Alice and Bob, who continue to argue.
The debate's topic is: "<TOPIC>"
During each turn of the debate, Alice or Bob presents concise arguments. They specify whether they are arguing for "Yes" or "No" based on the question posed in the debate.[/INST]"""


# In[7]:


def get_debate_str(debate):
    
    get_role_token = lambda role_str: "|<" + role_str.upper() + ">|"
    
    debate_str = """|<START_DEBATE>|"""
    for msg in debate:
        role = get_role_token(msg["name"])
        if role!="|<HUMAN_JUDGE>|":
            debate_str += role+msg["content"]+role
    debate_str+= "|<END_DEBATE>|"
    
    return debate_str

def process_debate_data(debate_data,is_aug=False):
    
    debates_list = []

    for index, (topic, debate) in enumerate(debate_data.items()):
        if is_aug:
            for example in debate["right_examples"]:
                if example:
                    transcript = debate_prefix.replace("<TOPIC>",topic) + example
                    debates_list.append({"topic":topic,"transcript":transcript})
            
        else:
            transcript = debate_prefix.replace("<TOPIC>",topic) + get_debate_str(debate["debate_messages"])
            debates_list.append({"topic":topic,"transcript":transcript})
        
    return debates_list


# In[8]:


debates_processed = []

for item in process_debate_data(debate_data):
    json_line = json.dumps(item)
    debates_processed.append(json_line)

for item in process_debate_data(augmented_debate_data, is_aug=True):
    json_line = json.dumps(item)
    debates_processed.append(json_line)

debates_processed = sorted(debates_processed)

with open("data/debates_data_v3.jsonl", 'w') as jsonl_file:
    for item in debates_processed:
        jsonl_file.write(item + '\n')


# In[9]:


len(debates_processed)


# In[10]:


train_dataset = load_dataset('json', data_files='data/debates_data_v3.jsonl', split='train[:150]')
eval_dataset = load_dataset('json', data_files='data/debates_data_v3.jsonl', split='train[150:]')


# In[11]:


def tokenize_prompt(example):
    
    prompt = example["transcript"]
    result = tokenizer(prompt, truncation=True, max_length=1024, padding="max_length")
    result["labels"] = result["input_ids"].copy()
    
    return result


# In[12]:


tokenized_train_dataset = train_dataset.map(tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(tokenize_prompt)        


# In[13]:


model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="balanced", use_safetensors=False)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


# In[14]:


config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)


# In[15]:


if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True


# In[16]:


fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
model = accelerator.prepare_model(model)


# In[17]:


project = "debate-finetune-v3"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=500,
        per_device_train_batch_size=12,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        num_train_epochs=20,
        learning_rate=2e-5, 
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=20,              
        logging_dir="./logs",        
        save_strategy="epoch",
        evaluation_strategy="epoch", 
        do_eval=True,                
        report_to="wandb",           
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)


# In[ ]:


model.config.use_cache = False  
trainer.train()


# In[ ]:


ft_model = PeftModel.from_pretrained(model, "mistral-debate-finetune-v3/checkpoint-130/")


# In[ ]:


def eval_model(idx):
    
    topic,transcript = list(debate_data.keys())[idx], get_debate_str(list(debate_data.values())[idx]['debate_messages'])
    
    input_prefix = debate_prefix.replace("<TOPIC>",topic)+"|<BOB>|".join(transcript.split("|<BOB>|")[:2])
    model_input = tokenizer(input_prefix, return_tensors="pt").to("cuda")
    
    ft_model.eval()

    with torch.no_grad():
        model_trascript = tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=800)[0], skip_special_tokens=True)

    return input_prefix, transcript, model_trascript


# In[ ]:


input_prefix, transcript, model_transcript = eval_model(-1)


# In[ ]:


print(f"model-result: {model_transcript} \n\n\nground-truth: {transcript}")


# In[ ]:




