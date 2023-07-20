from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate

import environ
env = environ.Env()
environ.Env.read_env()

API_KEY = env('HUGGING_FACE_TOKEN')

#print (API_KEY)

prompt = PromptTemplate(
    input_variables=["question"],
    template="translate english to SQL: {question}"
)

hugging_face_llm = HuggingFaceHub(repo_id="mrm8488/t5-base-finetuned-wikiSQL",huggingfacehub_api_token=API_KEY)
hugging_face_chain  = LLMChain(llm=hugging_face_llm, prompt=prompt, verbose=True)


while True:
    prompt = input("Enter a prompt: ")

    if prompt.lower() == 'exit':
        print('Exiting...')
        break
    else:
        try:
            #question = QUERY.format(question=prompt)
            print (hugging_face_chain.run(prompt))
        except Exception as e:












            
            print(e)



#print (hugging_face_chain.run(prompt))

#print (hugging_face_chain.run("what is the average age of the respondents using mobile device"))
#print (hugging_face_chain.run("Total number of respondents"))
