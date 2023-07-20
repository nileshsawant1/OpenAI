
from langchain import SQLDatabase, SQLDatabaseChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain import HuggingFaceHub, LLMChain

# Setting up the api key
import environ
env = environ.Env()
environ.Env.read_env()

#API_KEY = env('OPENAI_API_KEY')
API_KEY = env('HUGGING_FACE_TOKEN')

#print (API_KEY)
              
db = SQLDatabase.from_uri(f"postgresql+psycopg2://postgres:{env('DB_PASSWORD')}@localhost:5432/Northwind",)

# setup llm
#llm = ChatOpenAI(temperature=0, openai_api_key=API_KEY, model_name='gpt-3.5-turbo')
llm = HuggingFaceHub(repo_id="mrm8488/t5-base-finetuned-wikiSQL",huggingfacehub_api_token=API_KEY)

# Create query instruction
QUERY = """
Given an input question, first create a syntactically correct postgresql query to run, then look at the results of the query and return the answer.
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

{question}
"""

# Setup the database chain
db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)

""" memory = ConversationBufferMemory()
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, memory=memory)
 """
def get_prompt():
    print("Type 'exit' to quit")

while True:
    prompt = input("Enter a prompt: ")

    if prompt.lower() == 'exit': 
        print('Exiting...')
        break
    else:
        try:
            question = QUERY.format(question=prompt)
            print(db_chain.run(question))
        except Exception as e:
            print(e)

get_prompt()