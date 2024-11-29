import streamlit as st
import requests
import json

LLM_MAPPING = {
    "healthcare": "microsoft/phi-3-medium-128k-instruct:free",
    "mathematics": "meta-llama/llama-3.1-8b-instruct",
    "programming": "huggingfaceh4/zephyr-7b-beta:free",
    "creative writing": "mistralai/mistral-7b-instruct:free",
    "science": "qwen/qwen-2-7b-instruct:free",
    "reasoning": "meta-llama/llama-3.1-8b-instruct",
    "education": "meta-llama/llama-3.1-405b-instruct:free",  
    "general knowledge": "huggingfaceh4/zephyr-7b-beta:free",  
    "travel": "mistralai/mistral-7b-instruct:free",  
    "entertainment": "qwen/qwen-2-7b-instruct:free",
    "finance": "microsoft/phi-3-medium-128k-instruct:free",
    "technology": "google/gemma-2-9b-it:free",
    "shopping": "huggingfaceh4/zephyr-7b-beta:free",
    "history": "meta-llama/llama-3.1-8b-instruct",
    "geography": "qwen/qwen-2-7b-instruct:free",
    "art": "mistralai/mistral-7b-instruct:free",
    "music": "mistralai/mistral-7b-instruct:free",
    "sports": "huggingfaceh4/zephyr-7b-beta:free",
    "fitness": "meta-llama/llama-3.1-70b-instruct:free",
    "food": "mistralai/mistral-7b-instruct:free",
    "childcare": "huggingfaceh4/zephyr-7b-beta:free",
    "language": "meta-llama/llama-3.2-3b-instruct:free",
    "business": "microsoft/phi-3-medium-128k-instruct:free",
    "marketing": "huggingfaceh4/zephyr-7b-beta:free",
    "job": "meta-llama/llama-3.1-405b-instruct:free",
    "diy": "huggingfaceh4/zephyr-7b-beta:free",
    "dating": "mistralai/mistral-7b-instruct:free",
    "psychology": "meta-llama/llama-3.1-405b-instruct",
    "law": "microsoft/phi-3-medium-128k-instruct:free",
    "environment": "meta-llama/llama-3.1-405b-instruct:free",
    "astronomy": "qwen/qwen-2-7b-instruct:free",
    "fashion": "mistralai/mistral-7b-instruct:free",
    "gaming": "qwen/qwen-2-7b-instruct:free",
    "mythology": "mistralai/mistral-7b-instruct:free",
    "religion": "mistralai/mistral-7b-instruct:free",
    "pets": "meta-llama/llama-3.1-70b-instruct:free"
}

SECONDARY_LLM_MAPPING = {
    "healthcare": "meta-llama/llama-3.1-70b-instruct:free",
    "mathematics": "qwen/qwen-2-7b-instruct:free",
    "programming": "google/gemma-2-9b-it:free",
    "creative writing": "meta-llama/llama-3.1-8b-instruct",
    "science": "meta-llama/llama-3.1-405b-instruct:free",
    "reasoning": "qwen/qwen-2-7b-instruct:free",
    "education": "mistralai/mistral-7b-instruct:free",
    "general knowledge": "meta-llama/llama-3.2-3b-instruct:free",
    "travel": "qwen/qwen-2-7b-instruct:free",
    "entertainment": "mistralai/mistral-7b-instruct:free",
    "finance": "meta-llama/llama-3.1-8b-instruct",
    "technology": "meta-llama/llama-3.1-405b-instruct:free",
    "shopping": "mistralai/mistral-7b-instruct:free",
    "history": "qwen/qwen-2-7b-instruct:free",
    "geography": "meta-llama/llama-3.1-8b-instruct",
    "art": "huggingfaceh4/zephyr-7b-beta:free",
    "music": "meta-llama/llama-3.2-3b-instruct:free",
    "sports": "qwen/qwen-2-7b-instruct:free",
    "fitness": "meta-llama/llama-3.2-3b-instruct:free",
    "food": "meta-llama/llama-3.1-8b-instruct",
    "childcare": "mistralai/mistral-7b-instruct:free",
    "language": "mistralai/mistral-7b-instruct:free",
    "business": "meta-llama/llama-3.2-3b-instruct:free",
    "marketing": "meta-llama/llama-3.1-405b-instruct:free",
    "job": "meta-llama/llama-3.1-8b-instruct",
    "diy": "qwen/qwen-2-7b-instruct:free",
    "dating": "meta-llama/llama-3.2-3b-instruct:free",
    "psychology": "mistralai/mistral-7b-instruct:free",
    "law": "meta-llama/llama-3.1-8b-instruct",
    "environment": "meta-llama/llama-3.2-3b-instruct:free",
    "astronomy": "meta-llama/llama-3.1-405b-instruct:free",
    "fashion": "meta-llama/llama-3.2-3b-instruct:free",
    "gaming": "huggingfaceh4/zephyr-7b-beta:free",
    "mythology": "meta-llama/llama-3.1-70b-instruct:free",
    "religion": "meta-llama/llama-3.1-8b-instruct",
    "pets": "qwen/qwen-2-7b-instruct:free"
}

OPENROUTER_API_KEY = "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"

def detect_intent(query):
    data = {
        "model": "google/gemma-2-9b-it:free",  
        "messages": [
            {
                "role": "user",
                "content": f"""Classify the intent i.e the domain of the user query from this list: 
                (healthcare, mathematics, programming, creative writing, science, reasoning, education, 
                general knowledge, travel, entertainment, finance, technology, shopping, history, 
                geography, art, music, sports, fitness, food, childcare, language, business, 
                marketing, job, diy, dating, psychology, law, environment, astronomy, fashion, 
                gaming, mythology, religion, pets). Reply with only one word from the list above: '{query}'"""
            }
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        response_data = response.json()
        intent = response_data["choices"][0]["message"]["content"].strip().lower()
        return intent
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

def fetch_response(query, llm_name, secondary_llm=None):
    data = {
        "model": llm_name,
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            response_data = response.json()
            if "choices" in response_data and response_data["choices"]:
                return response_data["choices"][0]["message"]["content"].strip()
            else:
                st.error("Unexpected response format from Primary LLM. Trying Secondary LLM...")
                if secondary_llm:
                    return fetch_response(query, secondary_llm)
                else:
                    return None
        elif response.status_code != 200 and secondary_llm:
            st.warning("Rate Limit Exceeded for Primary LLM. Trying Secondary LLM...", secondary_llm)
            return fetch_response(query, secondary_llm)
        else:
            st.error(f"Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        st.error(f"Exception occurred: {e}")
        if secondary_llm:
            st.info("Trying Secondary LLM due to primary failure...")
            return fetch_response(query, secondary_llm)
        return None

st.title("AiGator - Smart Router")

user_query = st.text_input("Enter your query:")

if st.button("Ask"):
    if user_query:
        with st.spinner("Detecting intent..."):
            try:
                intent = detect_intent(user_query)
                st.success(f"Detected Intent: {intent.capitalize()}")

                primary_llm = LLM_MAPPING.get(intent, "meta-llama/llama-3.1-70b-instruct:free")
                secondary_llm = SECONDARY_LLM_MAPPING.get(intent)

                st.info(f"Using Primary LLM: {primary_llm}")

                with st.spinner("Fetching response from LLM..."):
                    response = fetch_response(user_query, primary_llm, secondary_llm)
                    st.write("### Response")
                    if response:
                        st.write(response)
                    else:
                        st.error("Unable to fetch response. Try again later.")

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a query.")
