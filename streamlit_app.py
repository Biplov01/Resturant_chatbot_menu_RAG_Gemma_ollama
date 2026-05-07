# -*- coding: utf-8 -*-
"""
Himalayan Spice Restaurant — RAG Chatbot
Stack: SentenceTransformers + FAISS + Ollama (gemma:2b) + Streamlit
Run:
  1. ollama serve          (separate terminal)
  2. streamlit run app.py
  (menu.json must be in the same folder)
"""

import json
import os
import shutil

import streamlit as st
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

# ── Config ─────────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = "faiss_restaurant_index"
MENU_FILE        = r"menu.json"
OLLAMA_MODEL     = "gemma:2b"   # change to mistral-small once downloaded

st.set_page_config(
    page_title="Himalayan Spice Chatbot",
    page_icon="🍛",
    layout="wide",
)

# ── Load menu.json ─────────────────────────────────────────────────────────────
def load_menu() -> dict:
    if not os.path.exists(MENU_FILE):
        st.error(f" '{MENU_FILE}' not found. Place it in the same folder as app.py.")
        st.stop()
    with open(MENU_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# ── Convert menu → LangChain Documents ────────────────────────────────────────
def menu_to_documents(menu_data: dict) -> list[Document]:
    docs = []
    info = menu_data["restaurant"]

    # Restaurant info as one document
    info_text = (
        f"Restaurant Name: {info['name']}\n"
        f"Tagline: {info['tagline']}\n"
        f"Address: {info['address']}\n"
        f"Phone: {info['phone']}\n"
        f"Opening Hours: {info['hours']}\n"
        f"Reservations: {info['policies']['reservations']}\n"
        f"Dietary Options: {info['policies']['dietary']}\n"
        f"Payment Methods: {info['policies']['payment']}"
    )
    docs.append(Document(page_content=info_text, metadata={"section": "Restaurant Info"}))

    # One document per menu section
    for section, items in menu_data["menu"].items():
        lines = [f"Section: {section}\n"]
        for item in items:
            veg   = "Vegetarian" if item.get("vegetarian") else "Non-Vegetarian"
            spicy = " | Spicy"   if item.get("spicy")      else ""
            lines.append(
                f"- {item['name']} | Price: {item['price']} NPR | {veg}{spicy}\n"
                f"  Description: {item['description']}\n"
            )
        docs.append(Document(
            page_content="\n".join(lines),
            metadata={"section": section}
        ))

    return docs

# ── Build vectorstore (cached — only runs once per session) ───────────────────
@st.cache_resource(show_spinner=" Building knowledge base from menu...")
def build_vectorstore():
    menu_data = load_menu()
    documents = menu_to_documents(menu_data)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)
    vectorstore = FAISS.from_documents(documents, embedding=embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore

# ── Build RAG chain (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤖 Loading AI model (Ollama)...")
def build_chain(_vectorstore):
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.0,
        base_url="http://localhost:11434",
    )

    menu_data = load_menu()
    restaurant_name = menu_data["restaurant"]["name"]

    prompt = ChatPromptTemplate.from_template(
        f"You are a friendly restaurant assistant for {restaurant_name}.\n"
        "Help customers with menu questions, recommendations, dietary needs, pricing, and reservations.\n"
        "Answer using ONLY the context below. If something is not in the context, say so politely.\n"
        "Always mention prices in NPR. Be warm, concise, and helpful.\n\n"
        "Context:\n{context}\n\n"
        "Customer: {question}\n\n"
        "Assistant:"
    )

    retriever = _vectorstore.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# ── Initialise ─────────────────────────────────────────────────────────────────
vectorstore          = build_vectorstore()
rag_chain, retriever = build_chain(vectorstore)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    menu_data = load_menu()
    info      = menu_data["restaurant"]

    st.title("🍛 " + info["name"])
    st.caption(info["tagline"])
    st.divider()

    st.markdown("### 📍 Info")
    st.markdown(f"**Address:** {info['address']}")
    st.markdown(f"**Phone:** {info['phone']}")
    st.markdown(f"**Hours:** {info['hours']}")
    st.divider()

    st.markdown("### 📋 Menu Sections")
    for section in menu_data["menu"]:
        count = len(menu_data["menu"][section])
        st.markdown(f"- {section} ({count} items)")
    st.divider()

    st.markdown("### 💡 Try asking...")
    suggestions = [
        "What vegetarian dishes do you have?",
        "What is the most expensive dish?",
        "Recommend a starter and main course combo",
        "What are your spicy options?",
        "What desserts are available?",
        "Do you have gluten-free options?",
        "What drinks do you serve?",
        "What are your opening hours?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=s):
            st.session_state["suggestion"] = s

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main chat UI ───────────────────────────────────────────────────────────────
st.title("🍛 Himalayan Spice — Restaurant Assistant")
st.caption("Ask me anything about our menu, prices, dietary options, or reservations!")

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                f"Namaste! 🙏 Welcome to **{info['name']}**!\n\n"
                "I can help you with:\n"
                "- 🍽️ Menu items and descriptions\n"
                "- 💰 Prices\n"
                "- 🥦 Vegetarian / dietary options\n"
                "- 🌶️ Spice levels\n"
                "- 📅 Reservations & opening hours\n\n"
                "What can I get for you today?"
            ),
        }
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle sidebar suggestion button
if "suggestion" in st.session_state:
    user_input = st.session_state.pop("suggestion")
else:
    user_input = st.chat_input("Ask about our menu, prices, dietary options...")

# Process input
if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke(user_input)

                # Show retrieved sources in expander
                retrieved = retriever.invoke(user_input)
                st.markdown(response)

                with st.expander("🔍 Sources used", expanded=False):
                    for doc in retrieved:
                        st.caption(f"📂 **{doc.metadata.get('section', '—')}**")
                        st.text(doc.page_content[:300] + "...")

            except Exception as e:
                if "not found" in str(e).lower() or "404" in str(e):
                    response = (
                        f"❌ Could not connect to Ollama model `{OLLAMA_MODEL}`.\n\n"
                        f"Please make sure:\n"
                        f"1. Ollama is running: `ollama serve`\n"
                        f"2. Model is pulled: `ollama pull {OLLAMA_MODEL}`"
                    )
                else:
                    response = f" An error occurred: {str(e)}"
                st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
