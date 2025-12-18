import streamlit as st
import requests
import io
import numpy as np
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Research Assistant", page_icon="🔬")

st.title("🔬 Real-Time Research Assistant")
st.write("Finding recent articles, analyzing, and summarizing on the fly.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("API settings")
    exa_key = st.text_input("Exa API Key", type="password", value=st.secrets.get("EXA_API_KEY", ""))
    fw_key = st.text_input("Fireworks API Key", type="password", value=st.secrets.get("FIREWORKS_API_KEY", ""))
    
    st.divider()
    num_results = st.slider("How many articles to search?", 3, 10, 5)

# --- MAIN INTERFACE ---
query = st.text_input("Enter the research topic:", "new methods for federated learning accuracy vs latency 2025")

if st.button("Start research"):
    if not exa_key or not fw_key:
        st.error("Please, specify the API keys in the sidebar!")
        st.stop()

    headers_exa = {"Authorization": f"Bearer {exa_key}", "Content-Type": "application/json"}
    headers_fw = {"Authorization": f"Bearer {fw_key}", "Content-Type": "application/json"}

    # 1. SEARCH
    with st.status("🔍 Step 1: Searching for articles in Exa...", expanded=True) as status:
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
            papers = resp.json().get("results", [])
            st.write(f"Found links: {len(papers)}")
        except Exception as e:
            st.error(f"Search error: {e}")
            st.stop()
        
        status.update(label="Search completed", state="complete", expanded=False)

    # 2. DOWNLOADING
    docs = []
    progress_bar = st.progress(0)
    
    with st.spinner("📥 Step 2: Downloading and reading PDF..."):
        for i, p in enumerate(papers):
            try:
                r = requests.get(p["url"], timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code == 200:
                    pdf = PdfReader(io.BytesIO(r.content))
                    text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                    if len(text) > 500:
                        docs.append({"title": p.get("title"), "url": p["url"], "text": text[:15000]})
            except:
                pass
            progress_bar.progress((i + 1) / len(papers))

    if not docs:
        st.error("Could not read any PDFs.")
        st.stop()

    # 3. Embeddings and Ranking
    with st.spinner("🧠 Step 3: Analysis and selection of the best articles..."):
        fw_embed_url = "https://api.fireworks.ai/inference/v1/embeddings"
        
        def get_embedding(txt):
            try:
                pl = {"input": txt[:2000], "model": "nomic-ai/nomic-embed-text-v1.5"}
                r = requests.post(fw_embed_url, json=pl, headers=headers_fw)
                return np.array(r.json()["data"][0]["embedding"])
            except:
                return np.zeros(768)

        q_emb = get_embedding(query)
        valid_docs = []
        
        for d in docs:
            d_emb = get_embedding(d["text"])
            if not np.all(d_emb == 0):
                d["embedding"] = d_emb
                valid_docs.append(d)

        if valid_docs:
            sims = [cosine_similarity([q_emb], [d["embedding"]])[0][0] for d in valid_docs]
            ranked = sorted(zip(sims, valid_docs), reverse=True, key=lambda x: x[0])
            top_docs = [d for _, d in ranked[:2]] # Берем топ-2
        else:
            top_docs = docs[:2] 

    # 4. Summarization
    st.subheader("📝 Research Results")
    
    fw_chat_url = "https://api.fireworks.ai/inference/v1/chat/completions"

    for d in top_docs:
        with st.container(border=True):
            st.markdown(f"### [{d['title']}]({d['url']})")
            
            with st.spinner(f"Generating summary for: {d['title']}..."):
                prompt = f"""Summarize this paper titled '{d['title']}'. 
                Focus on: core idea, methods, results. Be concise (bullet points).
                Text: {d['text'][:6000]}"""
                
                payload_llm = {
                    "model": "accounts/fireworks/models/deepseek-v3",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 700
                }
                
                try:
                    res = requests.post(fw_chat_url, json=payload_llm, headers=headers_fw)
                    summary = res.json()["choices"][0]["message"]["content"]
                    st.markdown(summary)
                except Exception as e:
                    st.error(f"LLM error: {e}")
