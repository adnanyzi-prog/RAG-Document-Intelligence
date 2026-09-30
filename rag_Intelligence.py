import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

load_dotenv()
st.set_page_config(page_title="RAG Document Intelligence Lab", page_icon="📚", layout="wide")

st.markdown("""
<style>
.stApp{background:#f6f8fc}.stApp h1,.stApp h2,.stApp h3{color:#172554}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #e5e7eb}
.hero{background:linear-gradient(135deg,#fff,#eef4ff);border:1px solid #dbe4f0;border-radius:20px;padding:28px 30px;margin-bottom:22px;box-shadow:0 8px 28px rgba(15,23,42,.06)}
.hero h1{margin:0;color:#172554}.hero p{color:#475569;font-size:1.02rem}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:15px;min-height:105px;box-shadow:0 4px 16px rgba(15,23,42,.04)}
.num{font-size:.75rem;font-weight:700;color:#4f46e5;text-transform:uppercase}.title{font-weight:700;color:#0f172a;margin-top:5px}.desc{color:#64748b;font-size:.82rem;margin-top:4px}
.metric{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:15px;text-align:center}.value{font-size:1.4rem;font-weight:800;color:#172554}.label{color:#64748b;font-size:.8rem}
.source{background:#f8fafc;border-left:4px solid #6366f1;border-radius:8px;padding:12px;margin:8px 0}.answer{background:#fff;border:1px solid #dbe4f0;border-radius:16px;padding:20px;box-shadow:0 5px 18px rgba(15,23,42,.05)}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero"><h1>📚 RAG Document Intelligence Lab</h1>
<p><b>This is Intelligent Chatbot</b><br>
Upload documents, split them into chunks, create embeddings, store them in Chroma,
retrieve relevant evidence, and generate answers grounded in the uploaded documents.</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    key_input = st.text_input("Groq API Key", type="password", placeholder="Enter your Groq API key")
    api_key = key_input or os.getenv("GROQ_API_KEY")
    st.divider(); st.subheader("RAG Configuration")
    chunk_size = st.slider("Chunk Size", 400, 1400, 900, 100)
    overlap = st.slider("Chunk Overlap", 50, 300, 120, 10)
    top_k = st.slider("Retrieved Chunks", 2, 6, 4, 1)
    
if not api_key:
    st.info("Enter your Groq API key in the sidebar or set GROQ_API_KEY in .env.")
    st.stop()
os.environ["GROQ_API_KEY"] = api_key

st.markdown("#### Generation")
model_name = st.selectbox(
        "Groq model",
        ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
        index=0,
    )

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", encode_kwargs={"normalize_embeddings": True})

@st.cache_resource
def get_llm():
    return ChatGroq(model="openai/gpt-oss-20b", temperature=0)

embeddings = get_embeddings(); llm = get_llm()

for k, v in {"vectorstore":None,"chunks":[],"files":[],"info":{}}.items():
    if k not in st.session_state: st.session_state[k] = v

st.subheader("🔄 RAG Pipeline")
steps=[("01","Load Document","PyPDFLoader reads PDFs."),("02","Split Chunks","RecursiveCharacterTextSplitter creates chunks."),("03","Embeddings","HuggingFace converts text to vectors."),("04","Vector DB","Chroma stores vectors."),("05","Retrieve","MMR finds relevant evidence."),("06","Generate","Groq answers from evidence.")]
cols=st.columns(6)
for c,(n,t,d) in zip(cols,steps):
    with c: st.markdown(f'<div class="card"><div class="num">Step {n}</div><div class="title">{t}</div><div class="desc">{d}</div></div>',unsafe_allow_html=True)

st.write(""); st.subheader("📄 1. Upload Documents")
files=st.file_uploader("Upload one or more PDF files",type="pdf",accept_multiple_files=True)
if files:
    st.success(f"{len(files)} PDF file(s) selected")
    for f in files: st.write(f"• **{f.name}** — {f.size/1024:.1f} KB")

if st.button("🚀 Build Knowledge Base",type="primary",use_container_width=True,disabled=not files):
    docs=[]; names=[]; bar=st.progress(0); status=st.empty()
    try:
        for i,f in enumerate(files):
            status.info(f"Loading {f.name}..."); tmp=None
            try:
                with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as out:
                    out.write(f.getbuffer()); tmp=out.name
                loaded=PyPDFLoader(tmp).load()
                for d in loaded: d.metadata["source_file"]=f.name
                docs.extend(loaded); names.append(f.name)
            finally:
                if tmp and os.path.exists(tmp): os.remove(tmp)
            bar.progress((i+1)/len(files))
        if not docs: st.error("No readable text was found."); st.stop()
        status.info("Splitting documents...")
        splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=overlap,separators=["\n\n","\n",". "," ",""])
        chunks=splitter.split_documents(docs)
        status.info("Creating embeddings and Chroma vector database...")
        st.session_state.vectorstore=Chroma.from_documents(documents=chunks,embedding=embeddings)
        st.session_state.chunks=chunks; st.session_state.files=names
        st.session_state.info={"files":len(names),"pages":len(docs),"chunks":len(chunks),"size":chunk_size}
        status.success("Knowledge base built successfully."); st.rerun()
    except Exception as e: st.error(f"Could not build the knowledge base: {e}")

if st.session_state.vectorstore is not None:
    st.subheader("📊 Knowledge Base Status")
    info=st.session_state.info; cols=st.columns(4)
    for c,v,l in zip(cols,[info["files"],info["pages"],info["chunks"],info["size"]],["PDF Files","Pages","Text Chunks","Chunk Size"]):
        with c: st.markdown(f'<div class="metric"><div class="value">{v:,}</div><div class="label">{l}</div></div>',unsafe_allow_html=True)
    with st.expander("📁 Documents in Knowledge Base"):
        for f in st.session_state.files: st.write("• "+f)

st.divider(); st.subheader("💬 2. Ask Questions")
if st.session_state.vectorstore is None:
    st.warning("Build the knowledge base first, then ask a question.")
else:
    question=st.text_area("Enter your question",placeholder="Example: What is the main purpose of this document?",height=100)
    if st.button("🔎 Retrieve Evidence & Generate Answer",type="primary",use_container_width=True):
        if not question.strip(): st.warning("Please enter a question."); st.stop()
        try:
            with st.spinner("Retrieving relevant evidence..."):
                retriever=st.session_state.vectorstore.as_retriever(search_type="mmr",search_kwargs={"k":top_k,"fetch_k":max(10,top_k*4)})
                retrieved=retriever.invoke(question)
            if not retrieved: st.warning("No relevant evidence was found."); st.stop()
            parts=[]
            for i,d in enumerate(retrieved,1):
                src=d.metadata.get("source_file",d.metadata.get("source","Unknown document")); page=d.metadata.get("page"); page=int(page)+1 if page is not None else "Unknown"
                parts.append(f"SOURCE {i}\nFile: {src}\nPage: {page}\nContent:\n{d.page_content}")
            prompt=f'''You are a document question-answering assistant. Answer ONLY from the document evidence below. Do not use outside knowledge or invent facts. Use simple language. Mention source numbers such as [Source 1] when supported. If the answer is not present, say exactly: "I could not find that answer in the uploaded documents."\n\nDOCUMENT EVIDENCE:\n{chr(10).join(parts)}\n\nQUESTION:\n{question}'''
            with st.spinner("Generating grounded answer..."): response=llm.invoke(prompt)
            st.markdown('<div class="answer"><h3>🤖 AI Answer</h3>',unsafe_allow_html=True); st.markdown(response.content); st.markdown('</div>',unsafe_allow_html=True)
            st.write(""); st.subheader("📚 Retrieved Evidence")
            for i,d in enumerate(retrieved,1):
                src=d.metadata.get("source_file",d.metadata.get("source","Unknown document")); page=d.metadata.get("page"); page=int(page)+1 if page is not None else "Unknown"
                with st.expander(f"Source {i} | {src} | Page {page}"): st.markdown(f'<div class="source">{d.page_content}</div>',unsafe_allow_html=True)
        except Exception as e: st.error(f"Question answering failed: {e}")

st.divider(); st.subheader("🔍 3. Search the Knowledge Base")
query=st.text_input("Search query",placeholder="Example: objectives, methodology, policy...")
if st.button("🔍 Search Knowledge Base",use_container_width=True):
    if st.session_state.vectorstore is None: st.warning("Build the knowledge base first.")
    elif not query.strip(): st.warning("Enter a search query.")
    else:
        try:
            results=st.session_state.vectorstore.similarity_search(query,k=top_k)
            st.success(f"{len(results)} relevant chunk(s) found.")
            for i,d in enumerate(results,1):
                src=d.metadata.get("source_file",d.metadata.get("source","Unknown document")); page=d.metadata.get("page"); page=int(page)+1 if page is not None else "Unknown"
                with st.expander(f"Result {i} | {src} | Page {page}"): st.write(d.page_content)
        except Exception as e: st.error(f"Semantic search failed: {e}")


st.divider()
with st.sidebar:

    st.subheader("🛡️ Hallucination Control")

    strict_grounding = st.toggle(
    "Strict document-only mode",
    value=True,
    help=(
        "When enabled, the AI must answer only from retrieved document "
        "evidence. If the answer is not found, it will say so."
    ),
)

if strict_grounding:
    st.success("🛡️ Hallucination protection: ON")
else:
    st.warning("⚠️ Flexible mode: AI may use general knowledge")

st.markdown('<div style="text-align:center;color:#64748b;font-size:.85rem;padding:25px">Designed By Muhammad Adnan • Data Science and AI • Batch-06<br>RAG Document Intelligence Application</div>',unsafe_allow_html=True)
