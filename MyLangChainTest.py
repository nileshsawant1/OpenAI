import os
from dotenv import load_dotenv, find_dotenv
print (load_dotenv(find_dotenv()))

print (os.getenv('OPENAI_API_KEY'))

from langchain.llms import OpenAI
llm = OpenAI(model_name="text-davinci-003")
llm("explain large language models in one sentence")

