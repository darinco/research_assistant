import streamlit as st
import requests
import io
import numpy as np
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Research Assistant", page_icon="🔬", layout="wide")

st.title("🔬 AI Research Assistant")
st.write("Finding recent articles, analyzing, and summarizing on the fly.")

# --- SIDEBAR & SECURITY ---
with st.sidebar:
    st.header("API Settings")
    
    # Logic: If key exists in secrets, use it and hide/disable input. 
    # Otherwise, show the input field.
    
    # 1. Exa Key
    if "EXA_API_KEY" in st.secrets:
        exa_key = st.secrets["EXA_API_KEY"]
        st.success("✅ Exa API Key loaded from secrets")
    else:
        exa_key = st.text_input("Exa API Key", type="password")

    # 2. Fireworks Key
    if "FIREWORKS_API_KEY" in st.secrets:
        fw_key = st.secrets["FIREWORKS_API_KEY"]
        st.success("✅ Fireworks API Key loaded from secrets")
    else:
        fw_key = st.text_input("Fireworks API Key", type="password")
    
    st.divider()
    num_results = st.slider("How many articles to search?", 3, 10, 5)

# --- HELPER FUNCTIONS ---

def get_embedding(text, api_key):
    """Fetches embedding for a single text."""
    url = "https://api.fireworks.ai/inference/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # Nomic model supports 8192 context length, so we can increase input size safely
    payload = {"input": text[:8000], "model": "nomic-ai/nomic-embed-text-v1.5"}
    
    try:
        resp = requests.post(url, json=payload, headers=headers)
        if resp.status_code == 200:
            return np.array(resp.json()["data"][0]["embedding"])
        else:
            return np.zeros(768)
    except:
        return np.zeros(768)

def process_pdf(paper):
    """Downloads and extracts text from a single PDF URL."""
    try:
        r = requests.get(paper["url"], timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            f = io.BytesIO(r.content)
            pdf = PdfReader(f)
            
            # Optimized extraction
            texts = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    texts.append(t)
            
            full_text = " ".join(texts)
            
            if len(full_text) > 500:
                return {
                    "title": paper.get("title", "No Title"), 
                    "url": paper["url"], 
                    "text": full_text[:20000] # Keep more text for the LLM
                }
    except Exception as e:
        # We fail silently here to not spam the UI, but return None
        return None
    return None

def generate_summary(doc, api_key):
    """Generates summary for a single document."""
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    prompt = f"""Summarize this research paper titled '{doc['title']}'. 
    Focus on: core idea, methods, results. Be concise (bullet points).
    
    Text snippet:
    {doc['text'][:6000]}
    """
    
    payload = {
        "model": "accounts/fireworks/models/deepseek-v3",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 700
    }
    
    try:
        resp = requests.post(url, json=payload, headers=headers)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {resp.text}"
    except Exception as e:
        return f"Error generating summary: {e}"

# --- MAIN LOGIC ---
query = st.text_input("Enter the research topic:", "new methods for federated learning accuracy vs latency 2025")

if st.button("🚀 Start Research"):
    if not exa_key or not fw_key:
        st.error("Missing API Keys! Please check your secrets or sidebar.")
        st.stop()

    headers_exa = {"Authorization": f"Bearer {exa_key}", "Content-Type": "application/json"}

    # 1. SEARCH
    with st.status("🔍 Step 1: Searching Exa...", expanded=True) as status:
        exa_url = "https://api.exa.ai/search"
        payload = {
            "query": query,
            "category": "papers",
            "type": "fast",
            "numResults": num_results,
            "filters": {"date": ">2024", "filetype": "pdf"}
        }
        
        try:
            resp = requests.post(exa_url, json=payload, headers=headers_exa)
            resp.raise_for_status()
            papers = resp.json().get("results", [])
            st.write(f"✅ Found {len(papers)} papers.")
        except Exception as e:
            st.error(f"Exa Search Error: {e}")
            st.stop()
        
        status.update(label="Search completed", state="complete", expanded=False)

    # 2. DOWNLOADING (PARALLEL)
    docs = []
    
    with st.spinner("📥 Step 2: Downloading and reading PDFs (Parallel)..."):
        # Using ThreadPoolExecutor to download multiple PDFs at once
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(process_pdf, papers))
        
        # Filter out None results (failed downloads)
        docs = [d for d in results if d is not None]

    if not docs:
        st.error("Could not read any PDFs. They might be behind paywalls or blocked.")
        st.stop()
        
    st.info(f"Successfully extracted text from {len(docs)} papers.")

    # 3. EMBEDDINGS & RANKING
    with st.spinner("🧠 Step 3: Analyzing semantic relevance..."):
        q_emb = get_embedding(query, fw_key)
        
        valid_docs = []
        # Calculate embeddings for docs
        for d in docs:
            d_emb = get_embedding(d["text"], fw_key)
            if not np.all(d_emb == 0):
                d["embedding"] = d_emb
                valid_docs.append(d)

        # Rank
        if valid_docs:
            sims = [cosine_similarity([q_emb], [d["embedding"]])[0][0] for d in valid_docs]
            # Zip, sort by score desc
            ranked = sorted(zip(sims, valid_docs), reverse=True, key=lambda x: x[0])
            top_docs = [d for _, d in ranked[:2]] # Top 2
        else:
            top_docs = docs[:2] # Fallback

    # 4. SUMMARIZATION (PARALLEL)
    st.subheader("📝 Research Brief")
    
    with st.spinner("Generating summaries..."):
        # Prepare arguments for parallel execution
        # We need to pass the doc and the key, so we use a lambda or list comprehension
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Create a list of futures
            future_to_doc = {executor.submit(generate_summary, doc, fw_key): doc for doc in top_docs}
            
            # As they complete, add summary to the doc object
            for future in future_to_doc:
                doc = future_to_doc[future]
                try:
                    doc["summary"] = future.result()
                except Exception as exc:
                    doc["summary"] = f"Generation failed: {exc}"

    # DISPLAY RESULTS
    for d in top_docs:
        with st.container(border=True):
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                st.markdown(f"### {d['title']}")
                st.caption(f"Source: {d['url']}")
            with col2:
                # Add a download button-like link or actual button if you saved the file
                st.link_button("Open PDF", d['url'])
            
            st.markdown("---")
            st.markdown(d["summary"])
