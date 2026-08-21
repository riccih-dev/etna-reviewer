"""Stage 3 — retrieve candidate evidence sentences for a claim.

Hybrid retrieval: TF-IDF cosine similarity (catches exact/near-exact wording)
blended with sentence-embedding cosine similarity (catches paraphrases and
synonyms that share no literal words at all — the gap pure TF-IDF can't
close). The embedding model is loaded lazily, once per process; if
sentence-transformers isn't installed, retrieval falls back to TF-IDF alone
rather than failing.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..models import Claim, Evidence, Sentence

_embedder = None


def _get_embedder():
    """Load the sentence embedding model once per process — reloading it per call is
    what would make naive embedding retrieval slow."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _tfidf_scores(claim_text: str, candidate_texts: list[str]) -> list[float]:
    corpus = candidate_texts + [claim_text]
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return [0.0] * len(candidate_texts)  # e.g. corpus is all stopwords on a tiny input
    claim_vec = matrix[-1]
    candidate_matrix = matrix[:-1]
    return cosine_similarity(claim_vec, candidate_matrix).flatten().tolist()


def _embedding_scores(claim_text: str, candidate_texts: list[str]) -> list[float] | None:
    """Semantic similarity, independent of shared vocabulary. Returns None (rather than
    zeros) when the model can't be loaded, so the caller can fall back to pure TF-IDF
    instead of silently down-weighting every score."""
    try:
        model = _get_embedder()
    except Exception:
        return None
    embeddings = model.encode([claim_text] + candidate_texts, normalize_embeddings=True)
    claim_emb, candidate_embs = embeddings[0], embeddings[1:]
    return (candidate_embs @ claim_emb).tolist()


def retrieve_evidence(claim: Claim, sentences: list[Sentence], top_k: int = 1) -> list[Evidence]:
    candidates = [s for s in sentences if s.id != claim.sentence_id]
    if not candidates:
        return []

    candidate_texts = [c.text for c in candidates]
    tfidf = _tfidf_scores(claim.text, candidate_texts)
    semantic = _embedding_scores(claim.text, candidate_texts)

    if semantic is None:
        combined = tfidf
    else:
        # Semantic similarity carries more weight — it's the whole point of adding it:
        # TF-IDF alone misses evidence that supports the claim in different words.
        combined = [0.35 * t + 0.65 * s for t, s in zip(tfidf, semantic)]

    ranked = sorted(zip(candidates, combined), key=lambda pair: pair[1], reverse=True)
    top = [pair for pair in ranked if pair[1] > 0][:top_k]

    evidence = []
    for sentence, score in top:
        evidence.append(Evidence(
            id=f"E{sentence.id[1:]}",  # keyed by sentence, not claim — same sentence, same node everywhere
            text=sentence.text,
            sentence_id=sentence.id,
            section=sentence.section,
            page=sentence.page,
            retrieval_score=round(float(score), 3),
        ))
    return evidence
