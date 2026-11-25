"""
Retrieval-Augmented Generation (RAG) Engine for Risk Policies & Mandates:
- Document Ingestion, Chunking & Metadata Tagging
- Vector Embeddings with Cosine Similarity Retrieval (pgvector compatible)
- Pre-loaded Institutional Risk Policies, Investment Mandates, and Regulatory Limits
- Semantic Search with Document Citations
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import re
from datetime import datetime


# ─── Default Institutional Risk Documents ─────────────────────────────────────

DEFAULT_RISK_DOCUMENTS = [
    {
        "doc_id": "pol_mandate_01",
        "title": "Global Portfolio Risk Policy & Investment Mandate",
        "doc_type": "Investment Mandate",
        "chunks": [
            {
                "section": "1.1 Asset Concentration Limits",
                "text": "No single equity asset holding shall exceed 25.0% of the total portfolio market value under active management. Any position exceeding 25.0% triggers an immediate Concentration Warning and requires a risk reduction rebalance within 5 business days. Positions exceeding 30.0% represent a Critical Mandate Breach.",
            },
            {
                "section": "1.2 Portfolio Volatility and Value at Risk (VaR) Constraints",
                "text": "The maximum permissible 1-day 95% Historical Value at Risk (VaR) is capped at 5.0% of total portfolio value. Maximum annualized portfolio volatility is set at 25.0%. Portfolios exceeding these thresholds must hedge equity beta or reduce high-beta component risk.",
            },
            {
                "section": "1.3 Drawdown & Stop-Loss Provisions",
                "text": "A trailing peak-to-trough portfolio drawdown exceeding 15.0% requires an emergency risk committee review. A cumulative drawdown exceeding 20.0% mandates an immediate 30% reduction in gross risk-weighted asset exposure.",
            },
        ],
    },
    {
        "doc_id": "pol_basel_02",
        "title": "Basel Committee Supervisory Framework for Market Risk Backtesting",
        "doc_type": "Regulatory Framework",
        "chunks": [
            {
                "section": "2.1 Basel Traffic Light Framework",
                "text": "Internal Value at Risk models are backtested over a 250-day rolling window at the 99% confidence level. 0 to 4 exceptions falls into the Green Zone (model accepted, 3.0x capital multiplier). 5 to 9 exceptions falls into the Yellow Zone (supervisory scrutiny, progressive capital multiplier add-on from +0.40 to +0.85). 10 or more exceptions falls into the Red Zone (model rejection, 4.0x capital multiplier).",
            },
            {
                "section": "2.2 Kupiec and Christoffersen Statistical Tests",
                "text": "Unconditional coverage is validated using the Kupiec Proportion of Failures likelihood ratio test. Independence of exceptions is validated using the Christoffersen test to ensure VaR exceedances do not exhibit volatility clustering.",
            },
        ],
    },
    {
        "doc_id": "pol_stress_03",
        "title": "Institutional Stress Testing & Scenario Analysis Guidelines",
        "doc_type": "Risk Policy",
        "chunks": [
            {
                "section": "3.1 Historical Scenario Replay Requirements",
                "text": "Portfolios must be stress-tested quarterly against the 2008 Lehman Brothers Global Financial Crisis, the 2020 COVID-19 Liquidity Shock, the 2022 Inflation/Rate Shock, and the 2023 Regional Banking Crisis. Stress results must report dollar loss, percentage drawdown, and recovery return needed.",
            },
            {
                "section": "3.2 Reverse Stress Testing Methodology",
                "text": "Reverse stress testing identifies the minimal plausible multi-factor shock vector required to breach a predefined portfolio capital loss threshold of 25%, 30%, or 40%, identifying latent single-asset and sector vulnerabilities.",
            },
        ],
    },
]


# ─── Simple Deterministic Vector Store ────────────────────────────────────────

class DocumentChunk:
    def __init__(self, doc_id: str, title: str, doc_type: str, section: str, text: str, embedding: np.ndarray):
        self.doc_id = doc_id
        self.title = title
        self.doc_type = doc_type
        self.section = section
        self.text = text
        self.embedding = embedding

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "doc_type": self.doc_type,
            "section": self.section,
            "text": self.text,
        }


def _simple_text_embedding(text: str, dim: int = 128) -> np.ndarray:
    """
    Deterministic pseudo-embedding for fast local search without external API dependencies.
    Tokenizes words, maps to hashed buckets, and normalizes to unit norm.
    """
    vec = np.zeros(dim, dtype=float)
    words = re.findall(r'\w+', text.lower())
    for w in words:
        h = hash(w) % dim
        vec[h] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


class RiskPolicyVectorStore:
    """
    In-memory vector store for institutional risk documentation with cosine similarity retrieval.
    Can be backed by PostgreSQL pgvector in production environments.
    """
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.chunks: List[DocumentChunk] = []
        self._load_default_documents()

    def _load_default_documents(self):
        for doc in DEFAULT_RISK_DOCUMENTS:
            for ch in doc["chunks"]:
                combined_text = f"{doc['title']} {ch['section']} {ch['text']}"
                emb = _simple_text_embedding(combined_text, self.dim)
                chunk_obj = DocumentChunk(
                    doc_id=doc["doc_id"],
                    title=doc["title"],
                    doc_type=doc["doc_type"],
                    section=ch["section"],
                    text=ch["text"],
                    embedding=emb,
                )
                self.chunks.append(chunk_obj)

    def add_document(self, doc_id: str, title: str, doc_type: str, section: str, text: str):
        combined_text = f"{title} {section} {text}"
        emb = _simple_text_embedding(combined_text, self.dim)
        chunk_obj = DocumentChunk(
            doc_id=doc_id, title=title, doc_type=doc_type, section=section, text=text, embedding=emb
        )
        self.chunks.append(chunk_obj)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top_k document chunks most relevant to query using cosine similarity.
        """
        if not self.chunks:
            return []

        q_emb = _simple_text_embedding(query, self.dim)
        results = []

        for ch in self.chunks:
            # Cosine similarity (vectors are unit normalized)
            sim = float(np.dot(q_emb, ch.embedding))
            results.append({
                "similarity_score": round(sim, 4),
                "doc_id": ch.doc_id,
                "title": ch.title,
                "doc_type": ch.doc_type,
                "section": ch.section,
                "text": ch.text,
            })

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]


# Singleton instance
risk_rag_store = RiskPolicyVectorStore()
