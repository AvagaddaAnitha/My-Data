import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.markdown("""
<style>
    .stApp {
        background: url("https://t4.ftcdn.net/jpg/04/25/71/47/360_F_425714791_144GwzyrqZ2qibkWAG5cXkk2XknX2UOt.jpg") no-repeat center center fixed;
        background-size: cover;
    }
    .chat-font {
        font-family: 'Cursive', sans-serif;
        color: #d63384;
    }
    .user-msg {
        background: #ffccff !important;
        border-radius: 18px !important;
        border: 2px solid #ff66b2 !important;
        padding: 10px;
    }
    .bot-msg {
        background: #ffe6f2 !important;
        border-radius: 18px !important;
        border: 2px solid #ff99cc !important;
        padding: 10px;
    }
    .stChatInput {
        background: #ffffff;
        border: 2px solid #ff66b2;
    }
</style>
""", unsafe_allow_html=True)

# Configure Gemini AI
genai.configure(api_key="AIzaSyChdnIsx6-c36f1tU2P2BYqkrqBccTyhBE")  
gemini = genai.GenerativeModel('gemini-1.5-flash')

embedder = SentenceTransformer('all-MiniLM-L6-v2') 

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('My_Data.csv')  # Replace with your dataset file name
        if 'question' not in df.columns or 'answer' not in df.columns:
            st.error("The CSV file must contain 'question' and 'answer' columns.")
            st.stop()
        df['context'] = df.apply(
            lambda row: f"Question: {row['question']}\nAnswer: {row['answer']}", 
            axis=1
        )
        embeddings = embedder.encode(df['context'].tolist())
        index = faiss.IndexFlatL2(embeddings.shape[1])  # FAISS index for similarity search
        index.add(np.array(embeddings).astype('float32'))
        return df, index
    except Exception as e:
        st.error(f"Failed to load data. Error: {e}")
        st.stop()

df, faiss_index = load_data()

st.markdown('<h1 class="chat-font">💖 Meet Anitha! 💖</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="chat-font">Ask me anything, and I\'ll respond with charm! </h3>', unsafe_allow_html=True)
st.markdown("---")

def find_closest_question(query, faiss_index, df):
    query_embedding = embedder.encode([query])
    _, I = faiss_index.search(query_embedding.astype('float32'), k=1)  # Top 1 match
    if I.size > 0:
        return df.iloc[I[0][0]]['answer']  # Return the closest answer
    return None

def generate_refined_answer(query, retrieved_answer):
    prompt = f"""You are Anitha, an AI Student. Respond to the following question in a friendly and conversational tone:
    Question: {query}
    Retrieved Answer: {retrieved_answer}
    - Provide a detailed and accurate response.
    - Ensure the response is grammatically correct and engaging.
    """
    response = gemini.generate_content(prompt)
    return response.text

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], 
                        avatar="👩" if message["role"] == "user" else "💖"):
        st.markdown(message["content"], unsafe_allow_html=True)

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking... 💭"):
        try:
            retrieved_answer = find_closest_question(prompt, faiss_index, df)
            if retrieved_answer:
                refined_answer = generate_refined_answer(prompt, retrieved_answer)
                response = f"💖 **Anitha**:\n{refined_answer}"
            else:
                response = "💖 **Anitha**:\nI'm sorry, I cannot answer that question."  
        except Exception as e:
            response = f"An error occurred: {e}"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
