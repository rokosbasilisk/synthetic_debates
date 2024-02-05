#!/usr/bin/env python
# coding: utf-8


import openai
import jsonlines
import os
import time
from tqdm import tqdm



openai.api_key = os.environ["OPENAI_API_KEY"]



data_path = "debates_data.jsonl"
debate_data = [] # your list with json objects (dicts)

with jsonlines.open(data_path) as reader:
    for line in reader:
        debate_data.append(line)


def augment_debate(debate_message):
    
    topic = debate_message["topic"]
    
    debate_transcript = debate_message["transcript"]

    prompt = f"""
    For the given debate topic {topic}, observe the sample debate: {debate_transcript}. 
    Generate two more debates on the same topic, maintaining the structure of starting and ending tokens. 
    Each of these debates should be between Alice and Bob, taking three turns each. 
    Ensure that you alter the arguments slightly without deviating from the topic. 
    Also, provide two more debates on the same topic, following the structure of the previous debate, but introducing issues such as irrelevant arguments, lack of consistency, failure to address counterarguments, misinterpretation of the topic, skipping turns, failure to conclude, inconsistency in tone, and redundant arguments. 
    These issues can undermine the structure and effectiveness of the discussion. 
    Debates with mistakes should start with a special header: "|<WRONG_EXAMPLE>|" and the correct ones should start with "|<RIGHT_EXAMPLE>|". 
    NOTE: when Alice's turn using |<ALICE>| and use |<BOB>| for Bob's turn, |<START_DEBATE>| and |<END_DEBATE>| to denote start and end of debate.
    """
    completion = openai.chat.completions.create(model="gpt-3.5-turbo-0125", messages=[{"role": "user","content":prompt}, ], temperature=0.3)
    
    generated_debates = completion.choices[0].message.content
    
    return generated_debates


augmented_debates = {}
for debate in tqdm(debate_data):
    augmented_debates[debate["topic"]] = augment_debate(debate)



def extract_between_strings(input_string, start, end):
    start_index = input_string.find(start)
    end_index = input_string.find(end, start_index + len(start))

    if start_index != -1 and end_index != -1:
        result = input_string[start_index + len(start):end_index]
        return start+result+end
    else:
        return None

augmented_debates_updated = {}

for topic,debate in augmented_debates.items():
    debate_ = debate.replace("|<WRONG_EXAMPLE>|","|<EXAMPLE>||<WRONG>|").replace("|<RIGHT_EXAMPLE>|","|<EXAMPLE>||<RIGHT>|")
    transcripts = debate_.split("|<EXAMPLE>|")
    
    right_examples = []
    wrong_examples = []
    
    for transcript in transcripts:
        if len(transcript)>0:
            if transcript.startswith("|<RIGHT>|"):
                right_examples.append(extract_between_strings(transcript,"|<START_DEBATE>|","|<END_DEBATE>|"))
            if transcript.startswith("|<WRONG>|"):
                wrong_examples.append(extract_between_strings(transcript,"|<START_DEBATE>|","|<END_DEBATE>|"))
                
    augmented_debates_updated[topic] = {"right_examples":right_examples, "wrong_examples":wrong_examples}


with open("augmented_debates.json","w") as f:
    f.write(json.dumps(augmented_debates_updated))

