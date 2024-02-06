#!/usr/bin/env python
# coding: utf-8


import openai
import jsonlines
import os
import time
from tqdm import tqdm
import json


openai.api_key = os.environ["OPENAI_API_KEY"]


data_path = "debates_data.jsonl"
debate_data = []

with jsonlines.open(data_path) as reader:
    for line in reader:
        debate_data.append(line)


def get_debate_result(debate_message):
    
    topic = debate_message["topic"]
    
    debate_transcript = debate_message["transcript"]

    prompt = f"""
    Here's the corrected version:
    "For the given debate topic {topic}, observe the sample debate: {debate_transcript}. You have to determine who won the debate—Alice or Bob. You should never answer 'neither'; always state who is better. Additionally, provide a one-line review explaining why you made that choice. Your response should follow this format:
    Winner: Alice
    Reason: Alice's arguments are more in line with the topic, especially in the first and last places ...etc."""
    completion = openai.chat.completions.create(model="gpt-3.5-turbo-0125", messages=[{"role": "user","content":prompt}, ], temperature=0.3)
    
    result = completion.choices[0].message.content
    
    return {"transcript":debate_transcript, "result":result}

debate_results = {}
for debate in tqdm(debate_data):
    debate_results[debate["topic"]] = get_debate_result(debate)

with open("data/results/debate_results.json","w") as f:
    f.write(json.dumps(debate_results))

