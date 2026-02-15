import os
import json
import streamlit as st
import streamlit.components.v1 as components
import streamlit.components.v1 as components
import tempfile
from pyvis.network import Network
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from pypdf import PdfReader

# ─────────────────────────────────────────────
# LOAD ENV + CONNECT
# ─────────────────────────────────────────────
load_dotenv()

ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────


def extract_text(uploaded_file):
    """Extract text from TXT or PDF."""
    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join([p.extract_text() for p in reader.pages])
    else:
        return uploaded_file.read().decode("utf-8")


def chunk_text(text):
    """Split document into chunks."""
    return [c.strip() for c in text.split("\n\n") if c.strip()]


def extract_entities_relations(chunk):
    """LLM extraction into JSON."""
    prompt = f"""
Extract entities and relations from this text.

Return ONLY valid JSON:

{{
 "entities":[{{"name":"AI","type":"CONCEPT"}}],
 "relations":[{{"source":"ML","relation":"SUBSET_OF","target":"AI"}}]
}}

Text:
{chunk}
"""

    res = ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    output = res.choices[0].message.content.strip()

    if "```" in output:
        output = output.split("```")[1].replace("json", "").strip()

    try:
        return json.loads(output)
    except:
        return {"entities": [], "relations": []}


def store_graph(entities, relations):
    """Store entities + relations in Neo4j Aura."""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

        for e in entities:
            session.run(
                "MERGE (n:Entity {name:$name}) SET n.type=$type",
                name=e["name"],
                type=e["type"]
            )

        for r in relations:
            session.run(
                """
                MATCH (a:Entity {name:$src}),
                      (b:Entity {name:$tgt})
                MERGE (a)-[:RELATES {type:$rel}]->(b)
                """,
                src=r["source"],
                tgt=r["target"],
                rel=r["relation"]
            )

def visualize_graph():
    query = """
    MATCH (a:Entity)-[r:RELATES]->(b:Entity)
    RETURN a.name AS source, r.type AS rel, b.name AS target
    LIMIT 50
    """

    nodes = set()
    edges = []

    with driver.session() as session:
        results = session.run(query)

        for record in results:
            src = record["source"]
            tgt = record["target"]
            rel = record["rel"]

            nodes.add(src)
            nodes.add(tgt)
            edges.append((src, tgt, rel))

    net = Network(height="500px", width="100%", bgcolor="white")

    for node in nodes:
        net.add_node(node, label=node)

    for src, tgt, rel in edges:
        net.add_edge(src, tgt, label=rel)

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp_file.name)

    with open(tmp_file.name, "r", encoding="utf-8") as f:
        html = f.read()

    components.html(html, height=550, scrolling=True)

def retrieve_context(question):
    """Retrieve graph context safely for question."""

    prompt = f"""
Extract only the keywords from this question.

Return ONLY a valid JSON array like:

["AI", "Machine Learning"]

Question: {question}
"""

    res = ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = res.choices[0].message.content.strip()

    # Remove markdown fences if present
    if "```" in raw:
        raw = raw.split("```")[1].replace("json", "").strip()

    # Safe JSON parsing
    try:
        keywords = json.loads(raw)
    except:
        keywords = [question]  # fallback: use full question

    context = []

    with driver.session() as session:
        for kw in keywords:
            records = session.run(
                """
                MATCH (n:Entity)
                WHERE toLower(n.name) CONTAINS toLower($kw)
                OPTIONAL MATCH (n)-[r]->(m)
                RETURN n.name AS entity,
                       collect(DISTINCT m.name) AS links
                """,
                kw=kw
            )

            for rec in records:
                context.append(f"{rec['entity']} → {rec['links']}")

    return "\n".join(context) if context else "No relevant graph context found."



def answer_question(question):
    """Answer grounded only in KG."""
    context = retrieve_context(question)

    res = ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "Answer ONLY using the graph context provided."},
            {"role": "user",
             "content": f"Graph Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.3
    )

    return res.choices[0].message.content


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────

st.set_page_config(page_title="KG Chat App", layout="wide")

st.title("🧠 Knowledge Graph Chat App (Neo4j Aura)")
st.write("Upload a document → Build KG → Ask questions grounded in graph context.")

# Upload file
uploaded = st.file_uploader("📄 Upload TXT or PDF", type=["txt", "pdf"])

if uploaded:
    text = extract_text(uploaded)
    chunks = chunk_text(text)

    st.success(f"Loaded {len(chunks)} chunks")

    if st.button("🚀 Build Knowledge Graph"):
        all_entities, all_relations = [], []

        with st.spinner("Extracting entities + relations..."):
            for chunk in chunks:
                data = extract_entities_relations(chunk)
                all_entities.extend(data["entities"])
                all_relations.extend(data["relations"])

        with st.spinner("Storing into Neo4j Aura..."):
            store_graph(all_entities, all_relations)

        st.success("✅ Knowledge Graph Created Successfully!")

        # ✅ Graph Visualization Button
        st.subheader("🌐 Graph Visualization")

        if st.button("Show Graph"):
            visualize_graph()


# Chat Interface
st.subheader("💬 Chat with your Knowledge Graph")

question = st.text_input("Ask a question:")

if question:
    with st.spinner("Thinking..."):
        response = answer_question(question)

    st.markdown("### 💡 Answer")
    st.write(response)

st.markdown("---")
st.caption("Powered by Neo4j Aura + OpenAI + Streamlit 🚀")
