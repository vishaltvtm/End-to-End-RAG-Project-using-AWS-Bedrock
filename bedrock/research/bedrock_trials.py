from langchain_community.llms import Bedrock
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import boto3
import streamlit as st

# Initialize the Bedrock client
def get_bedrock_client():
    return boto3.client(
        service_name='bedrock-runtime',
        region_name='us-west-2',  # Change to your desired region
        aws_access_key_id=st.secrets["aws_access_key_id"],
        aws_secret_access_key=st.secrets["aws_secret_access_key"]
    )

# Function to create a Bedrock LLM instance
def create_bedrock_llm(model_id):
    client = get_bedrock_client()
    return Bedrock(
        model_id=model_id,
        client=client
    )

# Function to run a trial with the Bedrock LLM
def run_bedrock_trial(model_id, prompt_template, input_text):
    llm = create_bedrock_llm(model_id)
    prompt = PromptTemplate(input_variables=["input"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    
    response = chain.run(input=input_text)
    return response

# Streamlit app to run Bedrock trials
def run_bedrock_trials():
    st.title("Bedrock LLM Trials")
    
    model_id = st.text_input("Enter Bedrock Model ID", "amazon.titan-tg1-large")
    prompt_template = st.text_area("Enter Prompt Template", "What is the capital of {input}?")
    input_text = st.text_input("Enter Input Text", "France")
    
    if st.button("Run Trial"):
        if model_id and prompt_template and input_text:
            try:
                response = run_bedrock_trial(model_id, prompt_template, input_text)
                st.success(f"Response: {response}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please fill in all fields.")
if __name__ == "__main__":
    run_bedrock_trials()
    