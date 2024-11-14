import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
import faiss
import requests
import pickle
import configparser

# Load configuration
config = configparser.ConfigParser()
config.read('.config')
openai.api_key = config['openai']['api_key']

# Initialize FAISS index and document data
index = faiss.IndexFlatL2(1536)  # Assuming OpenAI embeddings are 1536-dimensional
document_data = {}  # Store content, URL, and summaries for each document

# Load previous FAISS index and document data if they exist
try:
    with open("faiss_store_openai.pkl", "rb") as f:
        index = pickle.load(f)
    with open("document_data.pkl", "rb") as f:
        document_data = pickle.load(f)
except FileNotFoundError:
    pass  # If files don't exist, proceed with an empty FAISS index and document data

def fetch_article_content(url):
    """Fetch content from a given URL."""
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        st.error(f"Failed to retrieve content from {url}")
        return None

def categorize_content(content):
    """Generate summaries for each required category: Benefits, Application Process, Eligibility, Documents."""
    categories = {
        "Benefits": f"Summarize the scheme benefits: {content}",
        "Application Process": f"Summarize the application process: {content}",
        "Eligibility": f"Summarize eligibility requirements: {content}",
        "Documents": f"Summarize required documents: {content}"
    }
    summaries = {}
    for category, prompt in categories.items():
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )
        summaries[category] = response.choices[0].text.strip()
    return summaries

def process_content(url, content):
    """Generate embeddings and store in FAISS index with URL and structured data."""
    embeddings = OpenAIEmbeddings()
    document_embeddings = embeddings.embed_text(content)
    index.add(document_embeddings)
    document_data[url] = {
        "content": content,
        "summaries": categorize_content(content),
        "embedding": document_embeddings
    }

def ask_question(query):
    """Query FAISS index and get the answer, including relevant URL and summaries."""
    embeddings = OpenAIEmbeddings()
    query_vector = embeddings.embed_text(query)
    distances, indices = index.search(query_vector, k=1)  # Top-1 result

    if len(indices) > 0:
        answer_url = list(document_data.keys())[indices[0][0]]
        summaries = document_data[answer_url]["summaries"]
        answer_content = document_data[answer_url]["content"]
        answer_summary = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Based on the content: {answer_content}\nAnswer the question: {query}",
            max_tokens=200
        ).choices[0].text.strip()
        return answer_summary, answer_url, summaries
    else:
        return "No relevant information found.", None, None

def save_faiss_index():
    """Save the FAISS index and document data for future use."""
    with open("faiss_store_openai.pkl", "wb") as f:
        pickle.dump(index, f)
    with open("document_data.pkl", "wb") as f:
        pickle.dump(document_data, f)

def main():
    st.title("Automated Scheme Research Tool")
    st.sidebar.title("Input Options")
    
    url_input = st.sidebar.text_input("Enter Scheme URL")
    file_upload = st.sidebar.file_uploader("Upload a file containing URLs", type=['txt'])

    if st.sidebar.button("Process URLs"):
        urls = [url_input] if url_input else []
        
        if file_upload:
            urls += [line.strip() for line in file_upload.readlines()]
        
        for url in urls:
            content = fetch_article_content(url)
            if content:
                process_content(url, content)
                st.success(f"Processed: {url}")
        
        save_faiss_index()
        st.success("All URLs processed and indexed!")

    query = st.text_input("Ask a question about the schemes:")
    if query:
        answer, source_url, summaries = ask_question(query)
        if source_url:
            st.write("Answer:", answer)
            st.write("Source URL:", source_url)
            st.write("Summaries:", summaries)
        else:
            st.write("No relevant information found.")

if __name__ == "__main__":
    main()
