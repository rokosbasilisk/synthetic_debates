#!/usr/bin/env python
# coding: utf-8

from autogen.token_count_utils import count_token
import autogen
import concurrent
import json
import fire
import openai
import os
import random


os.environ["AUTOGEN_USE_DOCKER"] = "0"

llm_config = {'model': 'gpt-4-1106-preview'}

prompts["human_judge"] = "You are a human judge overseeing a debate."

prompts["debator"] = """
You are DEBATOR_NAME, an intelligent AI Debating system with the goal of convincing the judge of your answer. 
The debate centers around a specific question with a single correct answer. 
Winning the debate involves putting arguments that your answer is correct. 
Conceding your answer is not allowed, and you must consistently provide the best possible case for your answer throughout the debate.
While being concise, clearly state your arguments. 
Avoid unnecessary repetition of your answer and refrain from using excessive formal language. 
Adopt a casual tone, focusing on direct and concise argumentation supported by quotes.
Due to a strict word limit, do not waste time signposting sub-claims or repeating your answer unless necessary. 
Remember not to repeat the same an earlier argument. 
Emphasize the most critical aspects of your own arguments and identify flaws in the opposing debater's case.
At the start of the debate, you are informed of which side of the topic you are on, i.e., Yes or No.
Think step by step and ensure a clear, logical argument.
Your argument length must not exceed 100 words.
You must start your first argument by mentioning your whether your choice is "Yes" or "No for the topic of the debate
You must remember that you will have exactly 3 turns and each turn you will get to put out the best argument based on the previous arguments of you and your opponent
DO NOT MENTION THE NAME OF YOUR OPPONENT OR YOUR NAME, INSTEAD JUST PUT FORWARD YOUR ARGUMENTS.
YOU WILL LOSE THE DEBATE IF YOU USE YOUR OPPONENT'S NAME
"""

prompts["debate_motion"] = """
Hello fellow debaters,

Your assigned question for the debate is "TOPIC". Each of you will speak one at a time, and your positions on the debate are outlined below:

- Alice: ALICE_POSITION
- Bob: BOB_POSITION

Commence the debate.
"""


def run_debate(topic):

    human_judge = autogen.UserProxyAgent(name="human_judge", system_message=prompts["human_judge"], human_input_mode="NEVER")
    
    agents = [human_judge]
    
    for agent_name in ["Alice", "Bob"]:
        agent =  autogen.AssistantAgent(name=agent_name, llm_config=llm_config, system_message=prompts["debator"].replace("DEBATOR_NAME",agent_name))
        agents.append(agents)
    
    debate = autogen.GroupChat( agents=agents, messages=[], max_round=9, speaker_selection_method="round_robin" )

    manager = autogen.GroupChatManager(groupchat=debate, llm_config=llm_config)
    
    positions = random.sample(["Yes", "No"],2)
        
    debate_motion = prompts["debate_motion"].replace("TOPIC",topic).replace("ALICE_POSITION",positions[0],"BOB_POSITION",positions[1])

    agents[0].initiate_chat(manager,message=debate_motion)

    num_tokens = count_token(debate.messages)

    print(f"Topic: {topic}, Number of tokens: {num_tokens}")
    return {'debate_messages': debate.messages, 'num_tokens': num_tokens}


def run_debates_multiprocessed(topic_list, max_processes=4):  # Set max_processes to your desired limit
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
        results = list(executor.map(run_debate, topic_list))

    results_dict = dict(zip(topic_list, results))
    return results_dict


def process_topics_and_debates(input_topic_filename, output_debate_filename):
    with open(input_topic_filename, "r") as file:
        topic_list = sorted([line.strip() for line in file if line.strip()])

    debates_transcripts = run_debates_multiprocessed(topic_list)

    with open(output_debate_filename, 'w') as f:
        f.write(json.dumps(debates_transcripts, indent=4))

if __name__ == '__main__':
    fire.Fire(process_topics_and_debates)
    #python script.py --input_topic_filename=data/input_topics.txt --output_debate_filename=data/output_debates.json
