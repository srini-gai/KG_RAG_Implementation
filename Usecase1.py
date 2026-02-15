import os
import json
from openai import OpenAI
from neo4j import GraphDatabase
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# LOAD ENV VARIABLES
# ─────────────────────────────────────────────
load_dotenv()

ai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─────────────────────────────────────────────
# CONNECT TO NEO4J AURA (SSL handled automatically)
# ─────────────────────────────────────────────
db = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# ─────────────────────────────────────────────
# STEP 1: Load Document
# ─────────────────────────────────────────────
def load_faq(path=r"C:\KG_RAG_Usecase\AI.txt"):
    """Loads FAQ document and splits into chunks."""
    with open(path, encoding="utf-8") as f:
        text = f.read()

    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    print(f"📄 Loaded {len(chunks)} chunks")
    return chunks


# ─────────────────────────────────────────────
# STEP 2: Extract Entities & Relations (LLM)
# ─────────────────────────────────────────────
def extract(chunk):
    """Uses OpenAI to extract entities + relations in JSON format."""

    prompt = f"""
Extract entities and relations from this text.

Return ONLY valid JSON like:

{{
  "entities": [
    {{"name": "AI", "type": "CONCEPT"}}
  ],
  "relations": [
    {{"source": "ML", "relation": "SUBSET_OF", "target": "AI"}}
  ]
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

    # Remove markdown fences if present
    if "```" in output:
        output = output.split("```")[1].replace("json", "").strip()

    try:
        return json.loads(output)
    except Exception:
        return {"entities": [], "relations": []}


# ─────────────────────────────────────────────
# STEP 3: Store in Neo4j
# ─────────────────────────────────────────────
def store(entities, relations):
    """Stores extracted knowledge into Neo4j graph."""

    with db.session() as s:

        # Clear old data
        s.run("MATCH (n) DETACH DELETE n")

        # Insert entities
        for e in entities:
            s.run(
                """
                MERGE (n:Entity {name: $name})
                SET n.type = $type
                """,
                name=e["name"],
                type=e["type"]
            )

        # Insert relations
        for r in relations:
            s.run(
                """
                MATCH (a:Entity {name: $src}),
                      (b:Entity {name: $tgt})
                MERGE (a)-[:RELATES {type: $rel}]->(b)
                """,
                src=r["source"],
                rel=r["relation"],
                tgt=r["target"]
            )

    print(f"💾 Stored {len(entities)} entities, {len(relations)} relations")


# ─────────────────────────────────────────────
# STEP 4: Graph Search (RAG Retrieval)
# ─────────────────────────────────────────────
def search_graph(question):
    """Search graph for relevant entities + relationships."""

    # Extract keywords using OpenAI
    res = ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f'Extract keywords from this question as a JSON array: "{question}"'
        }],
        temperature=0
    )

    output = res.choices[0].message.content.strip()

    if "```" in output:
        output = output.split("```")[1].replace("json", "").strip()

    keywords = json.loads(output)

    results = []

    with db.session() as s:
        for kw in keywords:
            records = s.run(
                """
                MATCH (n:Entity)
                WHERE toLower(n.name) CONTAINS toLower($kw)

                OPTIONAL MATCH (n)-[r:RELATES]->(m)
                OPTIONAL MATCH (p)-[r2:RELATES]->(n)

                RETURN n.name AS entity,
                       n.type AS type,
                       collect(DISTINCT {rel: r.type, target: m.name}) AS out,
                       collect(DISTINCT {rel: r2.type, source: p.name}) AS inc
                """,
                kw=kw
            )

            for rec in records:
                d = rec.data()

                info = f"{d['entity']} ({d['type']})"

                for o in d["out"]:
                    if o["target"]:
                        info += f"\n  → {d['entity']} --{o['rel']}--> {o['target']}"

                for i in d["inc"]:
                    if i["source"]:
                        info += f"\n  ← {i['source']} --{i['rel']}--> {d['entity']}"

                results.append(info)

    return "\n\n".join(results) if results else "No info found."


# ─────────────────────────────────────────────
# STEP 5: Ask Question Using KG Context
# ─────────────────────────────────────────────
def ask(question):
    """Answer user question using KG context only."""

    context = search_graph(question)

    res = ai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer based ONLY on this Knowledge Graph context. Be concise."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.3
    )

    return res.choices[0].message.content


# ─────────────────────────────────────────────
# RUN THE PIPELINE
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("\n🚀 Building Knowledge Graph...\n")

    chunks = load_faq()

    all_entities, all_relations = [], []
    seen_e, seen_r = set(), set()

    # Extract entities + relations chunk by chunk
    for i, chunk in enumerate(chunks):
        print(f"  🧠 Extracting chunk {i+1}/{len(chunks)}...")

        data = extract(chunk)

        for e in data["entities"]:
            if e["name"] not in seen_e:
                seen_e.add(e["name"])
                all_entities.append(e)

        for r in data["relations"]:
            key = (r["source"], r["relation"], r["target"])
            if key not in seen_r:
                seen_r.add(key)
                all_relations.append(r)

    # Store into Neo4j
    store(all_entities, all_relations)

    print("\n✅ Knowledge Graph Ready!\n")

    # Query Loop
    print("💬 Ask anything! (type 'quit' to exit)\n")

    while True:
        q = input("❓ Question: ").strip()

        if q.lower() in ["quit", "exit", "q"]:
            break

        if q:
            print("\n💡 Answer:\n", ask(q), "\n")

    db.close()
