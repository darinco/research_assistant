import streamlit as st
import requests
import io
import numpy as np
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

# --- PAGE CONFIG ---
st.set_page_config(page_title="DeepResearch Agent", page_icon="🧠", layout="centered")

# Custom CSS to make it look cleaner
st.markdown("""
<style>
    .stChatMessage {
        background-color: transparent; 
    }
    .stChatInput {
        padding-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 DeepResearch Agent")
st.caption("Powered by Exa (Search) & Fireworks (Inference)")

# --- SIDEBAR & SETTINGS ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Keys Management
    if "EXA_API_KEY" in st.secrets:
        exa_key = st.secrets["EXA_API_KEY"]
    else:
        exa_key = st.text_input("Exa API Key", type="password")

    if "FIREWORKS_API_KEY" in st.secrets:
        fw_key = st.secrets["FIREWORKS_API_KEY"]
    else:
        fw_key = st.text_input("Fireworks API Key", type="password")
    
    st.divider()
    
    st.subheader("Search Parameters")
    num_results = st.slider("Articles to analyze", 3, 10, 4)
    year_range = st.slider("Publication Year", 2015, 2025, (2023, 2025))
    start_year, end_year = year_range

# --- SESSION STATE (Chat History) ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am ready to conduct scientific research. What topic are you interested in?"}
    ]

# --- HELPER FUNCTIONS (Logic) ---

def get_embedding(text, api_key):
    url = "https://api.fireworks.ai/inference/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"input": text[:8000], "model": "nomic-ai/nomic-embed-text-v1.5"}
    try:
        resp = requests.post(url, json=payload, headers=headers)
        return np.array(resp.json()["data"][0]["embedding"]) if resp.status_code == 200 else np.zeros(768)
    except:
        return np.zeros(768)

def process_pdf(paper):
    try:
        r = requests.get(paper["url"], timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            pdf = PdfReader(io.BytesIO(r.content))
            texts = [p.extract_text() for p in pdf.pages if p.extract_text()]
            full_text = " ".join(texts)
            if len(full_text) > 500:
                return {"title": paper.get("title"), "url": paper["url"], "text": full_text[:20000]}
    except:
        return None
    return None

def generate_summary(doc, api_key):
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    prompt = f"""Summarize this paper '{doc['title']}'. 
    Focus on: core idea, methods, results. Use Markdown.
    Text snippet: {doc['text'][:6000]}"""
    
    payload = {
        "model": "accounts/fireworks/models/deepseek-v3",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 800
    }
    try:
        resp = requests.post(url, json=payload, headers=headers)
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {e}"

# --- CHAT INTERFACE ---

# 1. Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 2. Handle new input
if query := st.chat_input("For example: LLM optimization methods 2024"):
    
    # Check keys
    if not exa_key or not fw_key:
        st.error("🔑 Please set API Keys in Sidebar!")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    # 3. Process logic (Assistant response)
    with st.chat_message("assistant"):
        
        # Container for the final answer
        response_container = st.empty()
        full_response = ""

        # Use st.status to hide the "ugly" logs
        with st.status("🔬 Exploring the topic...", expanded=True) as status:
            
            # --- STEP 1: Search ---
            status.write(f"🔍 Looking for articles ({start_year}-{end_year})...")
            exa_url = "https://api.exa.ai/search"
            payload = {
                "query": query, "category": "papers", "type": "fast", 
                "numResults": num_results,
                "startPublishedDate": f"{start_year}-01-01",
                "endPublishedDate": f"{end_year}-12-31",
                "filters": {"filetype": "pdf"}
            }
            try:
                resp = requests.post(exa_url, json=payload, headers={"Authorization": f"Bearer {exa_key}"})
                papers = resp.json().get("results", [])
                status.write(f"✅ Found {len(papers)} links.")
            except Exception as e:
                status.update(label="Search error", state="error")
                st.error(str(e))
                st.stop()

            # --- STEP 2: Download ---
            status.write("📥 Downloading and reading PDF...")
            docs = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                results = list(executor.map(process_pdf, papers))
                docs = [d for d in results if d is not None]
            
            if not docs:
                status.update(label="Failed to download PDF", state="error")
                st.error("Could not read PDF. They might be locked.")
                st.stop()

            # --- STEP 3: Rank ---
            status.write("🧠 Analyzing relevance...")
            q_emb = get_embedding(query, fw_key)
            valid_docs = []
            for d in docs:
                d_emb = get_embedding(d["text"], fw_key)
                if not np.all(d_emb == 0):
                    d["embedding"] = d_emb
                    valid_docs.append(d)
            
            if valid_docs:
                sims = [cosine_similarity([q_emb], [d["embedding"]])[0][0] for d in valid_docs]
                ranked = sorted(zip(sims, valid_docs), reverse=True, key=lambda x: x[0])
                top_docs = [d for _, d in ranked[:2]]
            else:
                top_docs = docs[:2]

            # --- STEP 4: Summarize ---
            status.write("📝 Summarizing...")
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_to_doc = {executor.submit(generate_summary, doc, fw_key): doc for doc in top_docs}
                for future in future_to_doc:
                    doc = future_to_doc[future]
                    doc["summary"] = future.result()

            status.update(label="Готово!", state="complete", expanded=False)

        # --- DISPLAY FINAL RESULT ---
        final_md = "### 🏁 Research results\n\n"
        
        for d in top_docs:
            final_md += f"#### 📄 [{d['title']}]({d['url']})\n"
            final_md += f"{d['summary']}\n\n---\n"
        
        st.markdown(final_md)
        
        # Save to history
        st.session_state.messages.append({"role": "assistant", "content": final_md})
