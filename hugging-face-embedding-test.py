# Setting up the api key
""" import environ
env = environ.Env()
environ.Env.read_env() """

import os
from dotenv import load_dotenv
load_dotenv()


hf_token = os.getenv('HUGGING_FACE_TOKEN')

# #API_KEY = env('OPENAI_API_KEY')
# API_KEY = env('HUGGING_FACE_TOKEN')

#hf_token= env('HUGGING_FACE_TOKEN') #"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"

model_id = "sentence-transformers/all-MiniLM-L6-v2"
import requests

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

def query(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()

""" texts = ["How do I get a replacement Medicare card?",
    "What is the monthly premium for Medicare Part B?",
    "How do I terminate my Medicare Part B (medical insurance)?",
    "How do I sign up for Medicare?",
    "Can I sign up for Medicare Part B if I am working and have health insurance through an employer?",
    "How do I sign up for Medicare Part B if I already have Part A?",
    "What are Medicare late enrollment penalties?",
    "What is Medicare and who can get it?",
    "How can I get help with my Medicare Part A and Part B premiums?",
    "What are the different parts of Medicare?",
    "Will my Medicare premiums be higher because of my higher income?",
    "What is TRICARE ?",
    "Should I sign up for Medicare Part B if I have Veterans' Benefits?"] """

texts2 = ["Sachin Ramesh Tendulkar, BR AO born 24 April 1973 is an Indian former international cricketer who captained the Indian national team. He is widely regarded as one of the greatest batsmen in the history of cricket.[4] He is the all-time highest run-scorer in both ODI and Test cricket with more than 18,000 runs and 15,000 runs, respectively.[5] He also holds the record for receiving the most man-of-the-match awards in international cricket.[6] Sachin was a Member of Parliament, Rajya Sabha by nomination from 2012 to 2018.[7][8]",
"Tendulkar took up cricket at the age of eleven, made his Test match debut on 15 November 1989 against Pakistan in Karachi at the age of sixteen, and went on to represent Mumbai domestically and India internationally for over 24 years.[9] In 2002, halfway through his career, Wisden ranked him the second-greatest Test batsman of all time, behind Don Bradman, and the second-greatest ODI batsman of all time, behind Viv Richards.[10] The same year, Tendulkar was a part of the team that was one of the joint-winners of the 2002 ICC Champions Trophy. Later in his career, Tendulkar was part of the Indian team that won the 2011 Cricket World Cup, his first win in six World Cup appearances for India.[11] He had previously been named Player of the Tournament at the 2003 World Cup.",
"Tendulkar has received several awards from the government of India: the Arjuna Award (1994), the Khel Ratna Award (1997), the Padma Shri (1998), and the Padma Vibhushan (2008).[12][13] After Tendulkar played his last match in November 2013, the Prime Minister's Office announced the decision to award him the Bharat Ratna, India's highest civilian award.[14][15] He was the first sportsperson to receive the reward and, as of 2023, is the youngest recipient.[16][17][18] In 2010, Time included Tendulkar in its annual list of the most influential people in the world.[19] Tendulkar was awarded the Sir Garfield Sobers Trophy for cricketer of the year at the 2010 International Cricket Council (ICC) Awards.[20]",
"Having retired from ODI cricket in 2012,[21][22] he retired from all forms of cricket in November 2013 after playing his 200th Test match.[23] Tendulkar played 664 international cricket matches in total, scoring 34,357 runs.[24] In 2013, Tendulkar was included in an all-time Test World XI to mark the 150th anniversary of Wisden Cricketers' Almanack, and he was the only specialist batsman of the post–World War II era, along with Viv Richards, to get featured in the team.[25] In 2019, he was inducted into the ICC Cricket Hall of Fame.[26] On 24 April 2023, the Sydney Cricket Ground unveiled a set of gates named after Tendulkar and Brian Lara on the occasion of Tendulkar's 50th birthday and the 30th anniversary of Lara's inning of 277 at the ground.[27][28][29]"
]

import json
output = query(texts2)

#print (output); """

""" with open("sample2.json", "w") as outfile:
    json.dump(output, outfile)
 """

import pandas as pd
embeddings = pd.DataFrame(output)
embeddings.to_csv("embeddings2.csv",index=False)

import torch
from datasets import load_dataset

#faqs_embeddings = load_dataset('nileshsawant1/SampleEmbedding') #'json',data_files="sample2.json"); #(embeddings) #('namespace/repo_name')
#faqs_embeddings = load_dataset("embeddings.csv")
faqs_embeddings = load_dataset("csv", data_files="embeddings2.csv")


#dataset_embeddings = torch.from_numpy(faqs_embeddings["train"].to_pandas().to_numpy()).to(torch.float)
dataset_embeddings = torch.from_numpy(faqs_embeddings["train"].to_pandas().to_numpy()).to(torch.float)


#question = ["How can Medicare help me?"]
question = ["When was sachin born?"]

output = query(question)

query_embeddings = torch.FloatTensor(output)

from sentence_transformers.util import semantic_search

hits = semantic_search(query_embeddings, dataset_embeddings, top_k=5)

print([texts2[hits[0][i]['corpus_id']] for i in range(len(hits[0]))])
