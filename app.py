import streamlit as st
import requests
import json

LLM_MAPPING = {
    "healthcare": "microsoft/phi-3-medium-128k-instruct:free",
    "mathematics": "meta-llama/llama-3.1-405b-instruct:free",
    "programming": "huggingfaceh4/zephyr-7b-beta:free",
    "creative writing": "mistralai/mistral-7b-instruct:free",
    "science": "qwen/qwen-2-7b-instruct:free",
    "education": "meta-llama/llama-3.1-405b-instruct:free",  
    "general knowledge": "huggingfaceh4/zephyr-7b-beta:free",  
    "travel": "mistralai/mistral-7b-instruct:free",  
    "entertainment": "qwen/qwen-2-7b-instruct:free",
    "finance": "microsoft/phi-3-medium-128k-instruct:free",
    "technology": "meta-llama/llama-3.1-405b-instruct:free",
    "shopping": "huggingfaceh4/zephyr-7b-beta:free",
    "history": "meta-llama/llama-3.1-405b-instruct:free",
    "geography": "qwen/qwen-2-7b-instruct:free",
    "art": "mistralai/mistral-7b-instruct:free",
    "music": "mistralai/mistral-7b-instruct:free",
    "sports": "huggingfaceh4/zephyr-7b-beta:free",
    "fitness": "google/gemini-exp-1114",
    "food ": "mistralai/mistral-7b-instruct:free",
    "childcare": "huggingfaceh4/zephyr-7b-beta:free",
    "language": "meta-llama/llama-3.1-405b-instruct:free",
    "business": "microsoft/phi-3-medium-128k-instruct:free",
    "marketing": "huggingfaceh4/zephyr-7b-beta:free",
    "job": "meta-llama/llama-3.1-405b-instruct:free",
    "diy": "huggingfaceh4/zephyr-7b-beta:free",
    "dating": "google/gemini-exp-1114",
    "psychology": "google/gemini-exp-1114",
    "law": "microsoft/phi-3-medium-128k-instruct:free",
    "environment": "meta-llama/llama-3.1-405b-instruct:free",
    "astronomy": "qwen/qwen-2-7b-instruct:free",
    "fashion": "mistralai/mistral-7b-instruct:free",
    "gaming": "qwen/qwen-2-7b-instruct:free",
    "mythology": "mistralai/mistral-7b-instruct:free",
    "religion": "mistralai/mistral-7b-instruct:free",
    "pets": "google/gemini-exp-1114"
}


OPENROUTER_API_KEY = "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"
YOUR_APP_NAME = "AiGator"  

def detect_intent(query):
    data = {
        "model": "google/gemma-2-9b-it:free",  
        "messages": [
            {
                "role": "user",
                "content": f"""Classify the intent i.e the domain of the user query from this list: 
                (healthcare, mathematics, programming, creative writing, science, education, 
                general knowledge, travel, entertainment, finance, technology, shopping, history, 
                geography, art, music, sports, fitness, food, childcare, language, business, 
                marketing, job, diy, dating, psychology, law, environment, astronomy, fashion, 
                gaming, mythology, religion, pets),
                reply with only one word from the given list above: '{query}'"""
            }
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        #"X-Title": f"{YOUR_APP_NAME}"
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"].strip().lower()
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

def fetch_response(query, llm_name):
    data = {
        "model": llm_name,
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ],
        "max_tokens": 300,
        "temperature": 0.7
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"].strip()
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

st.title("AiGator - Smart Router")

user_query = st.text_input("Enter your query:")

if st.button("Ask"):
    if user_query:
        with st.spinner("Detecting intent..."):
            try:
                intent = detect_intent(user_query)
                st.success(f"Detected Intent: {intent.capitalize()}")

                selected_llm = LLM_MAPPING.get(intent)
                if not selected_llm:
                    st.error(f"No LLM found for intent: {intent}")
                else:
                    st.info(f"Using Model: {selected_llm}")

                    with st.spinner("Fetching response from LLM..."):
                        response = fetch_response(user_query, selected_llm)
                        st.write("### Response")
                        st.write(response)

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a query.")