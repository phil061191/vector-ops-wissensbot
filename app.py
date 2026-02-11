import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

load_dotenv()

# Streamlit Cloud Secrets oder .env
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --- Seiten-Konfiguration ---
st.set_page_config(
    page_title="Vector Ops - Firmen-Wissensbot",
    page_icon="🔍",
    layout="wide"
)

# --- Custom CSS (Darkmode) ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4DA8FF, #7B61FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #999;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-ready {
        color: #4CAF50;
        font-weight: bold;
    }
    .status-waiting {
        color: #FFC107;
        font-weight: bold;
    }
    .demo-badge {
        background: linear-gradient(90deg, #4DA8FF, #7B61FF);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .feature-card {
        background-color: #1A1F2B;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #2A2F3B;
        margin-bottom: 0.8rem;
    }
    .feature-card h4 {
        color: #4DA8FF;
        margin-bottom: 0.5rem;
    }
    .feature-card p {
        color: #BBB;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

DEMO_DOCS_DIR = os.path.join(os.path.dirname(__file__), "demo_docs")


def extract_text_from_pdfs(pdf_files):
    """Text aus hochgeladenen PDFs extrahieren."""
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def load_demo_docs():
    """Demo-Dokumente automatisch laden."""
    text = ""
    if os.path.exists(DEMO_DOCS_DIR):
        for filename in os.listdir(DEMO_DOCS_DIR):
            if filename.endswith(".pdf"):
                filepath = os.path.join(DEMO_DOCS_DIR, filename)
                reader = PdfReader(filepath)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
    return text


def create_vector_store(text):
    """Text in Chunks aufteilen und Vektor-Datenbank erstellen."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store


def create_chain(vector_store):
    """RAG-Chain mit Gemini erstellen."""
    from langchain.prompts import PromptTemplate

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Du bist ein hilfreicher Firmen-Assistent. Beantworte die Frage ausfuehrlich und strukturiert auf Deutsch, basierend auf den folgenden Dokumenten.

Nutze Aufzaehlungen und Absaetze fuer eine gute Lesbarkeit. Wenn du konkrete Zahlen, Preise oder Details findest, nenne sie alle. Fasse am Ende kurz zusammen, wenn es sinnvoll ist.

Wenn du die Antwort nicht in den Dokumenten findest, sage das ehrlich.

Dokumente:
{context}

Frage: {question}

Ausfuehrliche Antwort:"""
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )
    return chain


def main():
    # Header
    st.markdown('<div class="main-header">Vector Ops - Firmen-Wissensbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Laden Sie Ihre Firmendokumente hoch und stellen Sie Fragen - der KI-Bot findet die Antworten.</div>', unsafe_allow_html=True)

    # Session State initialisieren
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### 🔍 Vector Ops")
        st.divider()

        api_key = os.getenv("GOOGLE_API_KEY", "")

        # Demo laden Button
        if not st.session_state.documents_processed:
            st.markdown('<span class="demo-badge">DEMO</span>', unsafe_allow_html=True)
            if st.button("Demo starten", type="primary", use_container_width=True):
                if not api_key:
                    st.error("API Key nicht konfiguriert.")
                else:
                    with st.spinner("Demo-Dokumente werden geladen..."):
                        text = load_demo_docs()
                        if text.strip():
                            vector_store = create_vector_store(text)
                            st.session_state.chain = create_chain(vector_store)
                            st.session_state.documents_processed = True
                            st.session_state.chat_history = []
                            st.success("Demo geladen! Stellen Sie jetzt Fragen.")
                        else:
                            st.error("Keine Demo-Dokumente gefunden.")

            st.divider()
            st.markdown("**Oder eigene Dokumente:**")

        uploaded_files = st.file_uploader(
            "PDF-Dateien hochladen",
            type=["pdf"],
            accept_multiple_files=True,
            help="Laden Sie eine oder mehrere PDF-Dateien hoch"
        )

        if uploaded_files and st.button("Dokumente verarbeiten", use_container_width=True):
            if not api_key:
                st.error("API Key nicht konfiguriert.")
            else:
                with st.spinner("Dokumente werden verarbeitet..."):
                    text = extract_text_from_pdfs(uploaded_files)
                    if text.strip():
                        vector_store = create_vector_store(text)
                        st.session_state.chain = create_chain(vector_store)
                        st.session_state.documents_processed = True
                        st.session_state.chat_history = []
                        st.success(f"{len(uploaded_files)} Dokument(e) erfolgreich verarbeitet!")
                    else:
                        st.error("Keine Texte in den PDFs gefunden.")

        st.divider()

        if st.session_state.documents_processed:
            st.markdown('<span class="status-ready">Status: Bereit</span>', unsafe_allow_html=True)
            if st.button("Neuen Chat starten", use_container_width=True):
                st.session_state.chain = None
                st.session_state.chat_history = []
                st.session_state.documents_processed = False
                st.rerun()
        else:
            st.markdown('<span class="status-waiting">Status: Warte auf Dokumente</span>', unsafe_allow_html=True)

        st.divider()
        st.caption("Powered by Vector Ops")

    # --- Chat-Bereich ---
    if not st.session_state.documents_processed:
        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>Dokumente hochladen</h4>
                <p>Laden Sie PDFs hoch - Handbuecher, Kataloge, FAQs oder interne Dokumente.</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>Fragen stellen</h4>
                <p>Stellen Sie Fragen in natuerlicher Sprache - der Bot durchsucht Ihre Dokumente.</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="feature-card">
                <h4>Sofort Antworten</h4>
                <p>Erhalten Sie praezise Antworten basierend auf Ihren eigenen Firmendokumenten.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("**Probieren Sie es aus:** Klicken Sie links auf *Demo starten* oder laden Sie eigene PDFs hoch.")

    else:
        # Chat-Verlauf anzeigen
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat-Eingabe
        if prompt := st.chat_input("Stellen Sie eine Frage zu Ihren Dokumenten..."):
            # Nutzer-Nachricht anzeigen
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # Bot-Antwort generieren
            with st.chat_message("assistant"):
                with st.spinner("Suche in Dokumenten..."):
                    response = st.session_state.chain.invoke({"question": prompt})
                    answer = response["answer"]
                    st.markdown(answer)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
