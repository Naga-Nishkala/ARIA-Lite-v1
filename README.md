# 🧠 ARIA-Lite v1 — Hybrid Graph + Vector Retrieval for Biomedical Papers

## Overview

ARIA-Lite v1 is a lightweight **Graph-enhanced Retrieval-Augmented Retrieval (RAG) prototype** designed for biomedical literature search.

It combines:

* **Dense semantic retrieval (FAISS embeddings)**
* **Biomedical knowledge graph connectivity**
* **Score-level fusion for final ranking**

The system improves retrieval quality by combining **semantic similarity** with **domain-specific entity structure**.

---

## 🔍 Retrieval Strategy

Given a query, the system retrieves relevant papers using two complementary signals:

### 1. Semantic Retrieval (Vector Space)

* Each paper abstract is embedded into a dense vector space
* Query is embedded using the same model
* Retrieval is performed using FAISS with **inner-product similarity over L2-normalized embeddings**
* This is equivalent to **cosine similarity search**

This captures:

* semantic similarity
* paraphrased biomedical concepts
* contextual relevance beyond exact entities

---

### 2. Graph-Based Retrieval (Biomedical Knowledge Graph)

A lightweight biomedical graph is constructed with:

* **ENTITY nodes** (e.g., genes, diseases, treatments)
* **PAPER nodes**
* Edges representing entity occurrence in papers

#### Query-driven graph traversal:

Given query entities (e.g.):

```
["breast", "cancer", "her2", "immunotherapy"]
```

The graph retrieval process:

1. Identify matching ENTITY nodes
2. Retrieve connected PAPER nodes (1-hop neighbors)
3. Extract ENTITY nodes connected to those PAPER nodes
4. Compute paper-level graph relevance scores

---

## 📊 Graph Scoring Mechanism

Each paper is scored based on the importance of its associated entities.

For a given paper:

* Let it contain multiple connected entities
* Each entity is assigned a weight based on:

  * **inverse frequency (rarity)**
  * **graph degree (common entities are down-weighted)**

Then:

> **Graph Score of a paper = sum of entity weights**

This behaves similarly to a **TF-IDF-inspired weighting scheme over a knowledge graph**, emphasizing rare and informative biomedical entities.

---

## ⚙️ Vector Retrieval (FAISS)

* Embeddings are L2-normalized
* FAISS index uses `IndexFlatIP`
* Inner product similarity is used, which is equivalent to cosine similarity under normalization

This produces a ranked list of papers based on semantic similarity to the query.

---

## 🔗 Hybrid Fusion

Final ranking is computed by combining both signals:

[
\text{Final Score} = a \cdot \text{Graph Score} + b \cdot \text{Cosine Similarity Score}
]

Where:

* `a` = weight for graph-based relevance
* `b` = weight for semantic similarity

Both scores are normalized before fusion.

---

## 🚀 Intuition

ARIA-Lite v1 combines:

* **FAISS retrieval → semantic understanding**
* **Graph retrieval → biomedical structure & specificity**

This enables the system to:

* retrieve semantically relevant papers
* prioritize biologically meaningful relationships
* reduce generic or overly broad matches

---

## 🧪 Summary

ARIA-Lite v1 is a:

> **Hybrid biomedical retrieval system combining dense vector search with entity-aware graph scoring using weighted score fusion.**
