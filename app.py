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

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .status-ready {
        color: #28a745;
        font-weight: bold;
    }
    .status-waiting {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


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

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True
    )
    return chain


def main():
    # Header
    st.markdown('<div class="main-header">Firmen-Wissensbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Laden Sie Ihre Firmendokumente hoch und stellen Sie Fragen - der Bot findet die Antworten.</div>', unsafe_allow_html=True)

    # Session State initialisieren
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### Vector Ops")
        st.header("Dokumente hochladen")

        api_key = os.getenv("GOOGLE_API_KEY", "")

        uploaded_files = st.file_uploader(
            "PDF-Dateien hochladen",
            type=["pdf"],
            accept_multiple_files=True,
            help="Laden Sie eine oder mehrere PDF-Dateien hoch"
        )

        if uploaded_files and st.button("Dokumente verarbeiten", type="primary", use_container_width=True):
            if not api_key:
                st.error("Bitte geben Sie einen Google API Key ein.")
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
        else:
            st.markdown('<span class="status-waiting">Status: Warte auf Dokumente</span>', unsafe_allow_html=True)

        st.divider()
        st.caption("Powered by Vector Ops")

    # --- Chat-Bereich ---
    if not st.session_state.documents_processed:
        st.info("Laden Sie Dokumente in der Seitenleiste hoch, um zu starten.")

        # Demo-Beispiele anzeigen
        st.markdown("### Beispiel-Fragen nach dem Upload:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("- *Welche Produkte bietet die Firma an?*")
            st.markdown("- *Was sind die Wartungsintervalle?*")
        with col2:
            st.markdown("- *Wie ist die Urlaubsregelung?*")
            st.markdown("- *Welche Sicherheitsvorschriften gelten?*")
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
