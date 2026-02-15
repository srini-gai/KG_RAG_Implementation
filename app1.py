import os
import json
import streamlit as st
import streamlit.components.v1 as components
import tempfile

from pyvis.network import Network
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from pypdf import PdfReader

# ─────────────────────────────────────────────
# STREAMLIT CONFIG (ONLY ONCE)
# ─────────────────────────────────────────────
st.set_page_config(page_title="KG Chat App", layout="wide")

# ─────────────────────────────────────────────
# LOAD ENV + CONNECT TO NEO4J AURA
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
    """Store extracted knowledge into Neo4j Aura."""
    with driver.session() as session:

        # ⚠️ Clears old data (demo purpose)
        session.run("MATCH (n) DETACH DELETE n")

        # Store Entities
        for e in entities:
            session.run(
                "MERGE (n:Entity {name:$name}) SET n.type=$type",
                name=e["name"],
                type=e["type"]
            )

        # Store Relations
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


# ─────────────────────────────────────────────
# GRAPH VISUALIZATION (QUESTION SUBGRAPH)
# ─────────────────────────────────────────────

def visualize_subgraph(question):
    """Show only graph related to user question (with fallback)."""

    # --- Extract keywords safely ---
    prompt = f"""
Extract 1–3 important keywords from this question.
Return ONLY JSON array.

Example: ["AI","Machine Learning"]

Question: {question}
"""

    res = ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = res.choices[0].message.content.strip()

    if "```" in raw:
        raw = raw.split("```")[1].replace("json", "").strip()

    try:
        keywords = json.loads(raw)
    except:
        keywords = []

    st.info(f"🔍 Keywords: {keywords}")

    # --- Subgraph Query (both directions) ---
    query = """
    MATCH (a:Entity)-[r:RELATES]-(b:Entity)
    WHERE any(k IN $keywords WHERE toLower(a.name) CONTAINS toLower(k))
       OR any(k IN $keywords WHERE toLower(b.name) CONTAINS toLower(k))

    RETURN a.name AS source, a.type AS source_type,
           r.type AS rel,
           b.name AS target, b.type AS target_type
    LIMIT 40
    """

    nodes = {}
    edges = []

    with driver.session() as session:
        results = session.run(query, keywords=keywords)

        for record in results:
            src = record["source"]
            tgt = record["target"]
            rel = record["rel"]

            nodes[src] = record["source_type"]
            nodes[tgt] = record["target_type"]

            edges.append((src, tgt, rel))

    # --- Fallback if no match ---
    if len(edges) == 0:
        st.warning("⚠️ No matching subgraph found. Showing default graph.")

        fallback = """
        MATCH (a:Entity)-[r:RELATES]->(b:Entity)
        RETURN a.name AS source, a.type AS source_type,
               r.type AS rel,
               b.name AS target, b.type AS target_type
        LIMIT 15
        """

        with driver.session() as session:
            results = session.run(fallback)

            for record in results:
                src = record["source"]
                tgt = record["target"]
                rel = record["rel"]

                nodes[src] = record["source_type"]
                nodes[tgt] = record["target_type"]

                edges.append((src, tgt, rel))

    # --- Node Color Map ---
    color_map = {
        "CONCEPT": "#FFA500",
        "TOOL": "#4DA6FF",
        "TECH": "#66CC66",
        "PERSON": "#FF99CC",
        "ORG": "#B266FF"
    }

    # --- Build PyVis Graph ---
    net = Network(
        height="750px",
        width="100%",
        bgcolor="white",
        directed=True
    )

    net.barnes_hut()

    # Add nodes
    for node, ntype in nodes.items():
        net.add_node(
            node,
            label=node,
            color=color_map.get(ntype, "gray"),
            size=28,
            font={"size": 20},
            title=f"Type: {ntype}"
        )

    # Add edges
    for src, tgt, rel in edges:
        net.add_edge(src, tgt, label=rel, arrows="to")

    # Save graph
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp_file.name)

    with open(tmp_file.name, "r", encoding="utf-8") as f:
        html = f.read()

    components.html(html, height=800, scrolling=True)


# ─────────────────────────────────────────────
# RETRIEVAL + ANSWER
# ─────────────────────────────────────────────

def retrieve_context(question):
    """Retrieve graph context safely."""

    prompt = f"""
Extract keywords from question as JSON array.

Question: {question}
"""

    res = ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = res.choices[0].message.content.strip()

    if "```" in raw:
        raw = raw.split("```")[1].replace("json", "").strip()

    try:
        keywords = json.loads(raw)
    except:
        keywords = [question]

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

    return "\n".join(context) if context else "No relevant context found."


def answer_question(question):
    """Answer grounded only in KG context."""
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

st.title("🧠 Knowledge Graph Chat App (Neo4j Aura)")
st.write("Upload a document → Build KG → Ask questions grounded in graph context.")

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

        st.subheader("🌐 Graph Preview")
        visualize_subgraph("AI")


# Chat Interface
st.subheader("💬 Chat with your Knowledge Graph")

question = st.text_input("Ask a question:")

if question:
    with st.spinner("Thinking..."):
        response = answer_question(question)

    st.markdown("### 💡 Answer")
    st.write(response)

    st.subheader("🌐 Graph Related to Your Question")
    visualize_subgraph(question)

st.markdown("---")
st.caption("Powered by Neo4j Aura + OpenAI + Streamlit 🚀")
