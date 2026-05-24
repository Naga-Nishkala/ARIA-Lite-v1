# ============================================================
# SECTION 2 — Imports
# ============================================================

import os
import json
import pickle
import tempfile

import faiss
import networkx as nx
import numpy as np
import streamlit as st

from pyvis.network import Network

from sentence_transformers import SentenceTransformer

from transformers import pipeline

from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# SECTION 3 — Mount Google Drive
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

# ============================================================
# SECTION 4 — Paths
# ============================================================

PROJECT_ROOT = "/content/drive/MyDrive/Colab_Notebooks/LLMs/ARIA_Lite"

GRAPH_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "aria_lite_graph_v2.pkl"
)

PAPERS_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "clean_papers.json"
)

FAISS_INDEX_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "indices",
    "faiss_index.bin"
)

PMIDS_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "indices",
    "pmids.pkl"
)

TEXTS_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "indices",
    "texts.pkl"
)

# ============================================================
# SECTION 5 — Load Data
# ============================================================

# Load graph
with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)

print("Graph loaded")

# Load papers
with open(PAPERS_PATH, "r") as f:
    papers = json.load(f)

print("Papers loaded:", len(papers))

# Load FAISS index
index = faiss.read_index(FAISS_INDEX_PATH)

print("FAISS index loaded")

# Load PMIDs
with open(PMIDS_PATH, "rb") as f:
    pmids = pickle.load(f)

# Load texts
with open(TEXTS_PATH, "rb") as f:
    texts = pickle.load(f)

print("Metadata loaded")

# ============================================================
# SECTION 6 — Paper Lookup Dictionary
# ============================================================

paper_lookup = {}

for p in papers:
    paper_lookup[p["pmid"]] = p

print("Paper lookup built")

# ============================================================
# SECTION 7 — Load Embedding Model
# ============================================================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded")

# ============================================================
# SECTION 8 — Load LLM
# ============================================================

generator = pipeline(
    "text-generation",
    model="microsoft/Phi-3-mini-4k-instruct",
    device_map="auto"
)

print("LLM loaded")

# ============================================================
# SECTION 9 — Normalize Function
# ============================================================

def normalize(text):

    return text.lower().strip()

# ============================================================
# SECTION 10 — Simple Entity Extraction
# ============================================================

def simple_entity_extract(query):

    tokens = query.lower().split()

    entities_found = []

    for node, data in G.nodes(data=True):

        if data.get("type") != "entity":
            continue

        node_lower = node.lower()

        for token in tokens:

            if token in node_lower:

                entities_found.append(node)

                break

    return list(set(entities_found))

# ============================================================
# SECTION 11 — Graph Retrieval
# ============================================================

def graph_retrieval(query_entities):

    paper_scores = {}
    related_entities = set()

    for ent in query_entities:

        if ent not in G:
            continue

        neighbors = list(G.neighbors(ent))

        for n in neighbors:

            node_type = G.nodes[n].get("type")

            # Paper node
            if node_type == "paper":

                degree = G.degree(n)

                score = 1 / (degree + 1)

                paper_scores[n] = paper_scores.get(n, 0) + score

            # Entity node
            elif node_type == "entity":

                related_entities.add(n)

    ranked_papers = sorted(
        paper_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_papers, list(related_entities)


# ============================================================
# SECTION 12 — Vector Retrieval
# ============================================================

def vector_search(query, top_k=10):

    query_embedding = embedding_model.encode([query])

    D, I = index.search(
        np.array(query_embedding).astype("float32"),
        top_k
    )

    results = []

    for idx, score in zip(I[0], D[0]):

        pmid = pmids[idx]

        results.append((pmid, float(score)))

    return results

# ============================================================
# SECTION 13 — Hybrid Retrieval
# ============================================================

def hybrid_retrieval(query):

    query_entities = simple_entity_extract(query)

    graph_papers, related_entities = graph_retrieval(
        query_entities
    )

    vector_results = vector_search(query)

    hybrid_scores = {}

    # Graph scores
    for pmid, score in graph_papers:

        hybrid_scores[pmid] = hybrid_scores.get(
            pmid,
            0
        ) + score * 0.4

    # Vector scores
    for pmid, score in vector_results:

        hybrid_scores[pmid] = hybrid_scores.get(
            pmid,
            0
        ) + score * 0.6

    ranked = sorted(
        hybrid_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "query": query,
        "entities": query_entities,
        "related_entities": related_entities,
        "papers": ranked
    }

# ============================================================
# SECTION 14 — Build Context
# ============================================================

def build_context(retrieved_papers, top_k=3):

    context = ""

    for pmid, score in retrieved_papers[:top_k]:

        if pmid not in paper_lookup:
            continue

        p = paper_lookup[pmid]

        text = p.get("text", "")

        text = text[:1500]

        context += f"""
PMID: {pmid}

TEXT:
{text}

==================================================
"""

    return context

# ============================================================
# SECTION 15 — Generate Answer
# ============================================================

def generate_answer(query, context):

    prompt = f"""
You are a biomedical AI assistant.

Answer the question using ONLY the provided papers.

Question:
{query}

Context:
{context}

Answer:
"""

    response = generator(
        prompt,
        max_new_tokens=180,
        do_sample=False,
        return_full_text=False
    )

    answer = response[0]["generated_text"]

    return answer

# ============================================================
# SECTION 16 — Build Local Subgraph
# ============================================================

def build_local_subgraph(query_entities, retrieved_papers):

    selected_nodes = set()

    # Add entities
    for ent in query_entities:

        if ent in G:
            selected_nodes.add(ent)

            for neighbor in G.neighbors(ent):

                selected_nodes.add(neighbor)

    # Add top papers
    for pmid, score in retrieved_papers[:10]:

        if pmid in G:

            selected_nodes.add(pmid)

            for neighbor in G.neighbors(pmid):

                selected_nodes.add(neighbor)

    subgraph = G.subgraph(selected_nodes)

    return subgraph

# ============================================================
# SECTION 17 — Convert Graph to PyVis
# ============================================================

def visualize_graph(subgraph, query_entities):

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#111111",
        font_color="white"
    )

    # Add nodes
    for node, data in subgraph.nodes(data=True):

        node_type = data.get("type")

        if node in query_entities:

            color = "red"
            size = 30

        elif node_type == "paper":

            color = "green"
            size = 18

        else:

            color = "skyblue"
            size = 20

        net.add_node(
            node,
            label=str(node)[:50],
            color=color,
            size=size
        )

    # Add edges
    for u, v, data in subgraph.edges(data=True):

        edge_type = data.get("type", "")

        if edge_type == "MENTIONS":
            color = "gray"

        elif edge_type == "CO_OCCURS":
            color = "orange"

        elif edge_type == "SIMILAR_PAPER":
            color = "purple"

        else:
            color = "white"

        net.add_edge(
            u,
            v,
            color=color
        )

    return net

# ============================================================
# SECTION 18 — Streamlit App
# ============================================================

st.set_page_config(
    page_title="ARIA-Lite",
    layout="wide"
)

st.title("ARIA-Lite")
st.subheader("Biomedical GraphRAG Assistant")

query = st.text_input(
    "Enter biomedical query:"
)

if st.button("Search"):

    # Retrieval
    results = hybrid_retrieval(query)

    query_entities = results["entities"]

    retrieved_papers = results["papers"]

    # Context
    context = build_context(
        retrieved_papers,
        top_k=3
    )

    # LLM answer
    answer = generate_answer(
        query,
        context
    )

    # Graph
    subgraph = build_local_subgraph(
        query_entities,
        retrieved_papers
    )

    net = visualize_graph(
        subgraph,
        query_entities
    )

    # ========================================================
    # LEFT COLUMN
    # ========================================================

    col1, col2 = st.columns([1, 1])

    with col1:

        st.markdown("## Generated Answer")

        st.write(answer)

        st.markdown("## Extracted Entities")

        st.write(query_entities)

    # ========================================================
    # RIGHT COLUMN
    # ========================================================

    with col2:

        st.markdown("## Top Retrieved Papers")

        for pmid, score in retrieved_papers[:5]:

            if pmid not in paper_lookup:
                continue

            paper = paper_lookup[pmid]

            st.markdown(f"""
### PMID: {pmid}

Score: {score:.4f}

Title:
{paper.get("title", "")}
""")

    # ========================================================
    # GRAPH VISUALIZATION
    # ========================================================

    st.markdown("## Knowledge Graph Visualization")

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".html"
    ) as tmp_file:

        net.save_graph(tmp_file.name)

        html_content = open(
            tmp_file.name,
            "r",
            encoding="utf-8"
        ).read()

    st.components.v1.html(
        html_content,
        height=800,
        scrolling=True
    )

