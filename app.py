import streamlit as st
import requests
import re
import os
from functools import lru_cache
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

load_dotenv()

MONGO_CONNECTION_URL = os.getenv('MONGO_CONNECTION_URL', '')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', '')


class MongoDBHandler:
    def __init__(self, connection_string):
        """
        Initialize MongoDB connection

        Args:
            connection_string (str): MongoDB connection string
        """
        try:
            # Create a new client and connect to the server
            self.client = MongoClient(
                connection_string, server_api=ServerApi('1'))

            # Select the database
            self.db = self.client['aigator-dev-db']

            # Create collections
            self.chat_histories = self.db['chat_histories']
            self.model_scores = self.db['model_scores']

            # Verify the connection
            self.client.admin.command('ping')
            st.success("Successfully connected to MongoDB!")

        except Exception as e:
            st.error(f"Error connecting to MongoDB: {e}")
            self.client = None
            self.db = None

    def insert_chat_history(self, entry):
        """
        Insert a chat history entry into MongoDB

        Args:
            entry (dict): Chat history entry to insert
        """
        if self.chat_histories is not None:
            try:
                return self.chat_histories.insert_one(entry)
            except Exception as e:
                st.error(f"Error inserting chat history: {e}")
                return None

    def insert_model_score(self, score_entry):
        """
        Insert a model score entry into MongoDB

        Args:
            score_entry (dict): Model score entry to insert
        """
        if self.model_scores is not None:
            try:
                return self.model_scores.insert_one(score_entry)
            except Exception as e:
                st.error(f"Error inserting model score: {e}")
                return None

    def get_chat_histories(self, limit=50):
        """
        Retrieve recent chat histories

        Args:
            limit (int): Number of chat histories to retrieve

        Returns:
            list: Recent chat histories
        """
        if self.chat_histories is not None:
            try:
                return list(self.chat_histories.find().sort('timestamp', -1).limit(limit))
            except Exception as e:
                st.error(f"Error retrieving chat histories: {e}")
                return []
        return []

    def get_chat_history_with_pagination(self, page=1, items_per_page=10):
        """
        Retrieve paginated chat history from MongoDB

        Args:
            page (int): Page number (1-based)
            items_per_page (int): Number of items per page

        Returns:
            tuple: (list of chat entries, total count)
        """
        if self.chat_histories is not None:
            try:
                # Calculate skip value
                skip = (page - 1) * items_per_page

                # Get total count
                total_count = self.chat_histories.count_documents({})

                # Get paginated results
                cursor = self.chat_histories.find() \
                    .sort('timestamp', -1) \
                    .skip(skip) \
                    .limit(items_per_page)

                return list(cursor), total_count
            except Exception as e:
                st.error(f"Error retrieving chat histories: {e}")
                return [], 0
        return [], 0

    def insert_intent_scores(self, query, primary_intent, additional_intents):
        """
        Insert intent scores for a specific query

        Args:
            query (str): Original user query
            primary_intent: Primary intent data
            additional_intents: Additional intent data
        """
        result = {}
        i = 0
        while i < len(additional_intents):
            # Check if this is the start of an intent group
            if 'accuracy:' in additional_intents[i + 1]:
                intent = additional_intents[i]
                accuracy = float(additional_intents[i + 2])
                result[intent] = accuracy
                i += 4  # Move to the next intent group
            else:
                i += 1  # Skip unrelated entries

        if self.model_scores is not None:
            try:
                # Prepare intent scores entry
                intent_scores_entry = {
                    'query': query,
                    'primary_intent': primary_intent,
                    'additional_intents': result,
                    'timestamp': time.time()
                }

                return self.model_scores.insert_one(intent_scores_entry)
            except Exception as e:
                st.error(f"Error inserting intent scores: {e}")
                return None


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
    def __init__(self, max_history=50, mongodb_handler=None):
        """
        Initialize chat history management

        Args:
            max_history (int): Maximum number of chat entries to store
            mongodb_handler (MongoDBHandler): MongoDB connection handler
        """
        self.mongodb_handler = mongodb_handler

        # Try to load chat history from MongoDB if possible
        if self.mongodb_handler:
            loaded_histories = self.mongodb_handler.get_chat_histories(
                max_history)

            if loaded_histories:
                st.session_state.chat_history = loaded_histories
            else:
                st.session_state.chat_history = [
                    {
                        'query': '(Example) What are the top restaurants in New York?',
                        'intent': 'food',
                        'llm': 'mistralai/mistral-7b-instruct:free',
                        'timestamp': time.time() - 3600  # 1 hour ago
                    },
                    {
                        'query': '(Example) Explain quantum computing basics',
                        'intent': 'technology',
                        'llm': 'meta-llama/llama-3.1-405b-instruct:free',
                        'timestamp': time.time() - 7200  # 2 hours ago
                    },
                    {
                        'query': '(Example) Best practices for Python programming',
                        'intent': 'programming',
                        'llm': 'huggingfaceh4/zephyr-7b-beta:free',
                        'timestamp': time.time() - 14400  # 4 hours ago
                    }
                ]
        else:
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = [
                    {
                        'query': '(Example) What are the top restaurants in New York?',
                        'intent': 'food',
                        'llm': 'mistralai/mistral-7b-instruct:free',
                        'timestamp': time.time() - 3600  # 1 hour ago
                    },
                    {
                        'query': '(Example) Explain quantum computing basics',
                        'intent': 'technology',
                        'llm': 'meta-llama/llama-3.1-405b-instruct:free',
                        'timestamp': time.time() - 7200  # 2 hours ago
                    },
                    {
                        'query': '(Example) Best practices for Python programming',
                        'intent': 'programming',
                        'llm': 'huggingfaceh4/zephyr-7b-beta:free',
                        'timestamp': time.time() - 14400  # 4 hours ago
                    }
                ]

        self.max_history = max_history

    # def add_entry(self, query, intent, llm, response):
    #     """
    #     Add a new entry to chat history

    #     Args:
    #         query (str): User's input query
    #         intent (str): Detected intent
    #         llm (str): LLM used
    #         response (str): LLM's response
    #     """
    #     # Create entry dictionary
    #     entry = {
    #         'query': query,
    #         'intent': intent,
    #         'llm': llm,
    #         'timestamp': time.time()
    #     }

    #     # Add to the beginning of the list
    #     st.session_state.chat_history.insert(0, entry)

    #     # Trim history if exceeds max
    #     if len(st.session_state.chat_history) > self.max_history:
    #         st.session_state.chat_history = st.session_state.chat_history[:self.max_history]

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
            'response': response,
            'timestamp': time.time()
        }

        # Add to MongoDB if handler exists
        if self.mongodb_handler:
            self.mongodb_handler.insert_chat_history(entry)

        # Add to the beginning of the list
        st.session_state.chat_history.insert(0, entry)

        # Trim history if exceeds max
        if len(st.session_state.chat_history) > self.max_history:
            st.session_state.chat_history = st.session_state.chat_history[:self.max_history]

    def display_history(self):
        """
        Display chat history in the sidebar
        """
        unique_models = sorted(
            set(list(LLM_MAPPING.values()) + list(SECONDARY_LLM_MAPPING.values())))

        st.sidebar.selectbox(
            "Explore Assistants",
            # sorted(list(LLM_MAPPING.keys())),
            unique_models,
        )
        st.sidebar.header("📜 Chat History")

        if not st.session_state.chat_history:
            st.sidebar.write("No chat history yet.")

            for _ in range(25):
                st.sidebar.markdown("")

            st.sidebar.markdown("""
                <style>
                .button-border{
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    padding: 5px 10px;
                }
                .sidebar-buttons-wrapper{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    grid-template-rows: auto auto;
                    gap: 10px;
                }
                .top {
                    grid-column: span 2;
                    text-align: center;
                }

                .bottom-left, .bottom-right {
                    text-align: center;
                }
                </style>
            """, unsafe_allow_html=True)

            st.sidebar.markdown("""<div class='sidebar-buttons-wrapper'>
                    <button class='top button-border'>🚪 Sign In</button>
                    <button class='bottom-left button-border'>🔑 Add API</button>
                    <button class='bottom-right button-border'>⚙️ Settings</button>
                </div>""",
                                unsafe_allow_html=True
                                )

        for idx, entry in enumerate(st.session_state.chat_history, 1):
            # Truncate long queries for display
            query_preview = (
                entry['query'][:30] + '...') if len(entry['query']) > 30 else entry['query']

            # Create expandable section for each chat entry
            with st.sidebar.expander(f"{idx}. {query_preview}", expanded=False):
                st.write(f"**Intent:** {entry['intent']}")
                st.write(f"**LLM:** {entry['llm']}")
                st.write(
                    f"**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry['timestamp']))}")

        for _ in range(10):
            st.sidebar.markdown("")

        st.sidebar.markdown("""
            <style>
            .button-border{
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 5px 10px;
            }
            .sidebar-buttons-wrapper{
                display: grid;
                grid-template-columns: 1fr 1fr;
                grid-template-rows: auto auto;
                gap: 10px;
            }
            .top {
                grid-column: span 2;
                text-align: center;
            }

            .bottom-left, .bottom-right {
                text-align: center;
            }
            </style>
        """, unsafe_allow_html=True)

        st.sidebar.markdown("""<div class='sidebar-buttons-wrapper'>
                <button class='top button-border'>🚪 Sign In</button>
                <button class='bottom-left button-border'>🔑 Add API</button>
                <button class='bottom-right button-border'>⚙️ Settings</button>
            </div>""",
                            unsafe_allow_html=True
                            )

    def add_api(self):
        print("Adding API")

    def settings_page(self):
        print("Settings Page")

    def sign_in(self):
        print("Sign In")


def track_performance(start_time, intent, llm, mongodb_handler=None):
    """
    Track and log query processing performance and optionally save to MongoDB

    Args:
        start_time (float): Start time of query processing
        intent (str): Detected intent
        llm (str): LLM used
        mongodb_handler (MongoDBHandler, optional): MongoDB connection handler

    Returns:
        float: Processing duration
    """
    end_time = time.time()
    duration = end_time - start_time

    # Optional: Save performance metrics to MongoDB
    if mongodb_handler:
        score_entry = {
            'intent': intent,
            'llm': llm,
            'processing_time': duration,
            'timestamp': end_time
        }
        mongodb_handler.insert_model_score(score_entry)

    return duration


def parse_intent(intent_text):
    """
    Parse the intent text with the new, more structured format
    """
    try:
        # Extract intent
        intent_match = re.search(
            r'intent:\s*(\w+)', intent_text, re.IGNORECASE)
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

                    and two more intents that are relevant to the query.

                    Sort them in descending order of confidence, with the primary intent first, do not mention any reasoning or anything more, just format your response as:
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
        intent = response_data["choices"][0]["message"]["content"].strip(
        ).lower()

        # Additional intent parsing
        other_intents = intent.split()[1:]

        final_intent = parse_intent(intent)

        if final_intent not in LLM_MAPPING:
            final_intent = "general knowledge"

        return final_intent, other_intents

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
            st.warning(f"Trying Secondary LLM due to network error: {
                       secondary_llm}")
            return fetch_response(query, secondary_llm)
        return None
    except Exception as e:
        if secondary_llm:
            st.warning(f"Trying Secondary LLM due to unexpected error: {
                       secondary_llm}")
            return fetch_response(query, secondary_llm)
        return None


def initialize_session_state():
    """Initialize default session state variables"""
    if 'chat_initialized' not in st.session_state:
        st.session_state.chat_initialized = True
        st.session_state.user_query = ""
        st.session_state.submitted_query = None
        st.session_state.last_response = None
        st.session_state.detected_intent = None
        st.session_state.processing_time = None
        st.session_state.show_response = False


def clear_chat_callback():
    """Callback function to clear chat state without page rerun"""
    st.session_state.user_query = ""
    st.session_state.submitted_query = None
    st.session_state.last_response = None
    st.session_state.detected_intent = None
    st.session_state.processing_time = None
    st.session_state.show_response = False


def handle_user_input():
    """Callback function to handle user input"""
    if st.session_state.user_query:
        st.session_state.submitted_query = st.session_state.user_query
        st.session_state.show_response = True


def format_timestamp(timestamp):
    """Format timestamp into readable date/time"""
    from datetime import datetime
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def display_chat_history_page(mongodb_handler):
    """Display the chat history page with filters and pagination"""
    st.title("Chat History")

    # Add filters in the sidebar
    st.sidebar.header("Filters")

    # Get unique intents and LLMs for filtering
    if mongodb_handler is not None and mongodb_handler.chat_histories is not None:
        unique_intents = mongodb_handler.chat_histories.distinct('intent')
        unique_llms = mongodb_handler.chat_histories.distinct('llm')
    else:
        unique_intents = []
        unique_llms = []

    # Filter selections
    selected_intent = st.sidebar.multiselect(
        "Filter by Intent",
        options=['All'] + list(unique_intents),
        default='All'
    )

    selected_llm = st.sidebar.multiselect(
        "Filter by LLM",
        options=['All'] + list(unique_llms),
        default='All'
    )

    # Pagination controls
    items_per_page = st.sidebar.selectbox(
        "Items per page",
        options=[10, 20, 50, 100],
        index=0
    )

    if mongodb_handler:
        # Get current page from query params or default to 1
        current_page = st.query_params.get("page", 1)
        try:
            current_page = int(current_page)
        except ValueError:
            current_page = 1

        # Get paginated chat history
        chats, total_count = mongodb_handler.get_chat_history_with_pagination(
            page=current_page,
            items_per_page=items_per_page
        )

        # Calculate total pages
        total_pages = (total_count + items_per_page - 1) // items_per_page

        # Display chats
        if chats:
            st.write(f"Showing {len(chats)} of {total_count} total chats")

            for chat in chats:
                with st.expander(f"Query: {chat['query'][:50]}...", expanded=False):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.write("**Full Query:**")
                        st.write(chat['query'])

                        st.write("**Response:**")
                        st.write(chat['response'])

                    with col2:
                        st.write("**Details:**")
                        st.write(f"Intent: {chat['intent']}")
                        st.write(f"LLM: {chat.get('llm', 'N/A')}")
                        st.write(
                            f"Time: {format_timestamp(chat['timestamp'])}")

                        # If you have intent scores stored
                        if 'intents_data' in chat:
                            st.write("**Intent Scores:**")
                            st.write(f"Primary ({chat['intents_data']['primary']['name']}): "
                                     f"{chat['intents_data']['primary']['confidence']:.2f}")

                            if chat['intents_data']['additional']:
                                st.write("Additional Intents:")
                                for intent in chat['intents_data']['additional']:
                                    st.write(
                                        f"- {intent['name']}: {intent['confidence']:.2f}")

                st.markdown("---")

            # Pagination controls
            col1, col2, col3 = st.columns([1, 2, 1])

            with col1:
                if current_page > 1:
                    if st.button("Previous"):
                        st.query_params["page"] = current_page - 1

            with col2:
                st.write(f"Page {current_page} of {total_pages}")

            with col3:
                if current_page < total_pages:
                    if st.button("Next"):
                        st.query_params["page"] = current_page + 1
        else:
            st.info("No chat history found.")
    else:
        st.error("Could not connect to MongoDB. Chat history is unavailable.")


def main():
    page = st.sidebar.radio("Navigation", ["Chat", "History"])
    initialize_session_state()

    try:
        mongodb_handler = MongoDBHandler(MONGO_CONNECTION_URL)
    except Exception as e:
        st.error(f"MongoDB Connection Error: {e}")
        mongodb_handler = None

    if page == "Chat":
        st.markdown("""
            <style>
            .button-border{
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 5px 10px;
            }
            .header-buttons-wrapper{
                display: flex;
                justify-content: space-between;
            }
            .flex-with-gap{
                display: flex;
                gap: 10px;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("""<div class='header-buttons-wrapper'>
                <div class='flex-with-gap'>
                    <button class='button-border'>LLMChat</button>
                    <button class='button-border'>Plugins (0)</button>
                </div>
                <div class='flex-with-gap'>
                    <button class='button-border'>Star on Github</button>
                    <button class='button-border'>Feedback</button>
                </div>
            </div>""",
                    unsafe_allow_html=True
                    )

        st.title("AiGator - Smart Router")

        # Initialize chat history
        chat_history = ChatHistory(mongodb_handler=mongodb_handler)

        # Sidebar chat history
        chat_history.display_history()

        col1, col2 = st.columns([4, 1])
        with col2:
            st.button("➕ New Chat", on_click=clear_chat_callback)

        # Add some vertical space to push content down
        for _ in range(10):
            st.markdown("")

        # Prompt suggestions
        suggested_prompts = [
            "Top-rated restaurants in my city",
            "Recent news in my area",
            "Summarize this article for me",
            "How to improve my Python skills",
            "What are the latest trends in renewable energy?"
        ]

        st.subheader("Quick Suggestions")
        col_prompts = st.columns(len(suggested_prompts))
        for i, prompt in enumerate(suggested_prompts):
            with col_prompts[i]:
                if st.button(prompt, key=f"prompt_{i}"):
                    st.session_state.user_query = prompt

        if 'submitted_query' not in st.session_state:
            st.session_state.submitted_query = None

        # User input
        # st.markdown("---")
        user_query = st.text_input(
            "Enter your query:",
            key="user_query",
            # on_change=lambda: setattr(st.session_state, 'submitted_query', st.session_state.user_query)
            on_change=handle_user_input
        )

        if st.button("Ask") or st.session_state.submitted_query:
            query_to_process = st.session_state.submitted_query or user_query

            if query_to_process:
                # Reset submitted query to prevent duplicate processing
                st.session_state.submitted_query = None
                start_time = time.time()

                with st.spinner("Detecting intent..."):
                    try:
                        # Detect intent
                        intent, intents_data = detect_intent(query_to_process)

                        # Store intent scores in MongoDB if handler exists
                        if mongodb_handler:
                            mongodb_handler.insert_intent_scores(
                                query_to_process, intent, intents_data)

                        st.success(f"Detected Intent: {intent.capitalize()}")

                        # Select appropriate LLMs
                        primary_llm = LLM_MAPPING.get(
                            intent, "meta-llama/llama-3.1-70b-instruct:free")
                        secondary_llm = SECONDARY_LLM_MAPPING.get(
                            intent, "meta-llama/llama-3.1-70b-instruct:free")

                        st.info(f"Using Primary LLM: {primary_llm}")

                        # Fetch response
                        with st.spinner("Fetching response from LLM..."):
                            response = fetch_response(
                                query_to_process, primary_llm, secondary_llm)

                            st.write("### Response")
                            if response:
                                st.write(response)

                                # Add to chat history
                                chat_history.add_entry(
                                    query=query_to_process,
                                    intent=intent,
                                    llm=primary_llm,
                                    response=response
                                )
                            else:
                                st.error(
                                    "Unable to fetch response. Try again later.")

                            # Track performance
                            # processing_time = track_performance(start_time, intent, primary_llm, mongodb_handler)
                            # st.info(f"Query processed in {processing_time:.2f} seconds")

                    except Exception as e:
                        st.error(f"Unexpected Error: {e}")
            else:
                st.warning("Please enter a query.")
        pass
    else:
        display_chat_history_page(mongodb_handler)


if __name__ == "__main__":
    main()
