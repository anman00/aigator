import streamlit as st
import requests
import json
import re
import os
from functools import lru_cache
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

OPENROUTER_API_KEY = "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"

# Extended LLM Mappings with more specific intents
LLM_MAPPING = {
    **{
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
        "pets": "meta-llama/llama-3.1-70b-instruct:free",
        "advanced_programming": "meta-llama/llama-3.1-405b-instruct:free",
        "technical_writing": "google/gemma-2-9b-it:free",
    },
}

SECONDARY_LLM_MAPPING = {
    **{
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
}

class ChatHistory:
    def __init__(self, max_history=50):
        """
        Initialize chat history management
        
        Args:
            max_history (int): Maximum number of chat entries to store
        """
        # Initialize session state for chat history if not exists
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        self.max_history = max_history
    
    def add_entry(self, query, intent, llm, response):
        """
        Add a new entry to chat history
        
        Args:
            query (str): User's input query
            intent (str): Detected intent
            llm (str): LLM used
            response (str): LLM's response
        """
        # Create entry dictionary
        entry = {
            'query': query,
            'intent': intent,
            'llm': llm,
            'timestamp': time.time()
        }
        
        # Add to the beginning of the list
        st.session_state.chat_history.insert(0, entry)
        
        # Trim history if exceeds max
        if len(st.session_state.chat_history) > self.max_history:
            st.session_state.chat_history = st.session_state.chat_history[:self.max_history]
    
    def display_history(self):
        """
        Display chat history in the sidebar
        """
        st.sidebar.header("📜 Chat History")
        
        if not st.session_state.chat_history:
            st.sidebar.write("No chat history yet.")
            return
        
        for idx, entry in enumerate(st.session_state.chat_history, 1):
            # Truncate long queries for display
            query_preview = (entry['query'][:30] + '...') if len(entry['query']) > 30 else entry['query']
            
            # Create expandable section for each chat entry
            with st.sidebar.expander(f"{idx}. {query_preview}", expanded=False):
                st.write(f"**Intent:** {entry['intent']}")
                st.write(f"**LLM:** {entry['llm']}")
                st.write(f"**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry['timestamp']))}")

def parse_intent(intent_text):
    """
    Parse the intent text with the new, more structured format
    """
    try:
        # Extract intent
        intent_match = re.search(r'intent:\s*(\w+)', intent_text, re.IGNORECASE)
        if intent_match:
            intent = intent_match.group(1).lower()
            
            # Validate intent exists in mapping
            if intent in LLM_MAPPING:
                return intent
        
        # Fallback if no valid intent found
        return "general knowledge"
    
    except Exception as e:
        return "general knowledge"

def create_retry_session():
    """Create a session with retry and backoff strategy."""
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
        backoff_factor=1
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    return session

def sanitize_query(query):
    """Sanitize and validate user input."""
    if not query or len(query.strip()) == 0:
        raise ValueError("Query cannot be empty")
    
    # Basic input sanitization
    query = query.strip()
    max_length = 1000
    if len(query) > max_length:
        query = query[:max_length]
    
    return query

def detect_intent(query):
    """Detect intent using OpenRouter API with enhanced error handling."""
    try:
        # Sanitize query
        query = sanitize_query(query)
        
        data = {
            "model": "google/gemma-2-9b-it:free",  
            "messages": [
                {
                    "role": "user",
                    "content": f"""Strictly classify the intent of the following query into ONE domain from this list: 
                    (healthcare, mathematics, programming, creative writing, science, reasoning, education, 
                    general knowledge, travel, entertainment, finance, technology, shopping, history, 
                    geography, art, music, sports, fitness, food, childcare, language, business, 
                    marketing, job, diy, dating, psychology, law, environment, astronomy, fashion, 
                    gaming, mythology, religion, pets). 

                    Format your response as: 
                    intent: [chosen intent from list]
                    accuracy: [confidence score between 0-1]

                    Query: '{query}'"""
                }
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        
        # Use retry session
        session = create_retry_session()
        
        # Add timeout
        response = session.post(
            "https://openrouter.ai/api/v1/chat/completions", 
            headers=headers, 
            json=data,
            timeout=10
        )
        
        response.raise_for_status()
        
        response_data = response.json()
        intent = response_data["choices"][0]["message"]["content"].strip().lower()

        # Additional intent parsing
        other_intents = intent.split()[1:]
        
        final_intent = parse_intent(intent)
        
        if final_intent not in LLM_MAPPING:
            final_intent = "general knowledge"
        
        return final_intent
    
    except Exception as e:
        return "general knowledge"
    
    except requests.exceptions.RequestException as e:
        st.error("Network error during intent detection. Please try again.")
        return "general knowledge"
    except Exception as e:
        return "general knowledge"

@lru_cache(maxsize=100)
def fetch_response(query, llm_name, secondary_llm=None):
    """Fetch response with caching, retry, and fallback mechanisms."""
    try:
        # Sanitize query
        query = sanitize_query(query)
        
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
        
        # Use retry session
        session = create_retry_session()
        
        # Add timeout
        response = session.post(
            "https://openrouter.ai/api/v1/chat/completions", 
            headers=headers, 
            json=data,
            timeout=10
        )
        
        response.raise_for_status()
        
        response_data = response.json()
        if "choices" in response_data and response_data["choices"]:
            return response_data["choices"][0]["message"]["content"].strip()
        else:
            st.error("Unexpected response format from Primary LLM.")
            if secondary_llm:
                return fetch_response(query, secondary_llm)
            return None
    
    except requests.exceptions.RequestException as e:
        if secondary_llm:
            st.warning(f"Trying Secondary LLM due to network error: {secondary_llm}")
            return fetch_response(query, secondary_llm)
        return None
    except Exception as e:
        if secondary_llm:
            st.warning(f"Trying Secondary LLM due to unexpected error: {secondary_llm}")
            return fetch_response(query, secondary_llm)
        return None

def track_performance(start_time, intent, llm):
    """Track and log query processing performance."""
    end_time = time.time()
    duration = end_time - start_time
    
    return duration

def main():
    st.title("AiGator - Smart Router")
    
    # Initialize chat history
    chat_history = ChatHistory()
    
    # Sidebar chat history
    chat_history.display_history()
    
    # User input
    user_query = st.text_input("Enter your query:")
    
    if st.button("Ask"):
        if user_query:
            start_time = time.time()
            
            with st.spinner("Detecting intent..."):
                try:
                    # Detect intent
                    intent = detect_intent(user_query)
                    st.success(f"Detected Intent: {intent.capitalize()}")
                    
                    # Select appropriate LLMs
                    primary_llm = LLM_MAPPING.get(intent, "meta-llama/llama-3.1-70b-instruct:free")
                    secondary_llm = SECONDARY_LLM_MAPPING.get(intent, "meta-llama/llama-3.1-70b-instruct:free")
                    
                    st.info(f"Using Primary LLM: {primary_llm}")
                    
                    # Fetch response
                    with st.spinner("Fetching response from LLM..."):
                        response = fetch_response(user_query, primary_llm, secondary_llm)
                        
                        st.write("### Response")
                        if response:
                            st.write(response)
                            
                            # Add to chat history
                            chat_history.add_entry(
                                query=user_query, 
                                intent=intent, 
                                llm=primary_llm, 
                                response=response
                            )
                        else:
                            st.error("Unable to fetch response. Try again later.")
                        
                        # Track performance
                        processing_time = track_performance(start_time, intent, primary_llm)
                        st.info(f"Query processed in {processing_time:.2f} seconds")
                    

                except Exception as e:
                    st.error(f"Unexpected Error: {e}")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()