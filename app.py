import os
import sys
import asyncio
import streamlit as st
import spacy
import base64
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from config import GEMINI_API_KEY
from langchain_community.vectorstores import Chroma
from neo4j import GraphDatabase

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

try:
    asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

nlp = spacy.load("en_core_web_sm")

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  
    location="us-central1"
)
embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001")

persist_directory = "./chroma_db"
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "test1234"
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

logo_path = "logo.png"

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def show_logo():
    logo_base64 = get_base64_of_bin_file(logo_path)
    logo_html = f'''
        <div style="position:fixed; bottom:10px; right:20px; opacity:0.85; z-index:100;">
            <img src="data:image/png;base64,{logo_base64}" width="120" />
            <div style="text-align:center; font-size:12px; color:#444;">Circonomit AI Prototype</div>
        </div>
    '''
    st.markdown(logo_html, unsafe_allow_html=True)

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def add_to_neo4j(entities):
    with driver.session() as session:
        for entity, label in entities:
            session.run(
                "MERGE (e:Entity {name: $name, type: $type})",
                name=entity,
                type=label,
            )

def add_to_vectorstore(text):
    vectorstore.add_texts([text])

st.title("Circonomit AI Prototype")

show_logo()

if st.button("Run demo pipeline"):
    with open("sample.txt", "r") as f:
        text = f.read()

    with st.expander("📄 Original Text", expanded=True):
        st.write(text)

    entities = extract_entities(text)
    with st.expander(f"🧠 Extracted Entities ({len(entities)})", expanded=True):
        if entities:
            st.table([{"Entity": e[0], "Type": e[1]} for e in entities])
        else:
            st.info("No entities found.")

    # Store text embedding
    add_to_vectorstore(text)
    st.success("✅ Text embedded & stored locally!")

    # Store entities in Neo4j
    add_to_neo4j(entities)
    st.success("✅ Entities stored in Neo4j graph!")

    # Ask Gemini for summary
    prompt = PromptTemplate.from_template("Summarise the business process: {text}")
    prompt_text = prompt.format(text=text)
    response = llm([HumanMessage(content=prompt_text)])

    with st.expander("🤖 Gemini Summary", expanded=True):
        st.write(response.content)

st.info("This is a prototype combining: local vector db + Neo4j + Gemini + spaCy")
