from __future__ import annotations

import dspy

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

_MODEL_NAME = "all-MiniLM-L6-v2"
_MODEL = None  # lazy-loaded on first call


def _get_model():
    global _MODEL
    if not _ST_AVAILABLE:
        return None
    if _MODEL is None:
        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


def _cosine_sim(a, b) -> float:
    """Cosine similarity between two 1-D arrays (already unit-normalised)."""
    import numpy as np
    return float(np.dot(a, b))


def fitness_metric(example: dspy.Example,
                   prediction: dspy.Prediction,
                   trace=None,
                   pred_name=None,
                   pred_trace=None):
    """Semantic similarity via sentence-transformer embeddings.

    Encodes both ``example.expected_behavior`` (gold rubric) and
    ``prediction.output`` (agent response) using the
    ``all-MiniLM-L6-v2`` model (384-dim, ~22 MB) and returns their
    cosine similarity as a score in [0, 1].

    Why this differs from lexical metrics
    --------------------------------------
    ``f1`` and ``bag_of_words`` are zero for paraphrases that share no
    surface tokens ("residence" vs "home").  This metric captures
    semantic equivalence — a rubric about "error recovery" scores high
    against a response about "handling failures" even without exact
    word matches.

    Cosine mapping
    --------------
    Normalised embeddings → dot product ∈ [-1, 1].
    Mapped to [0, 1] via ``(score + 1) / 2``.

    Graceful degradation
    --------------------
    If ``sentence_transformers`` is not installed the metric falls back
    to plain word-overlap recall so the pipeline never crashes.

    GEPA feedback
    -------------
    When called from GEPA (``pred_name`` is not None), returns a
    ``dspy.Prediction(score, feedback)`` with the raw similarity value
    so the reflection LM can see how semantically distant the response
    is from the rubric.  When called from MIPROv2 or the evaluation
    harness (``pred_name`` is None), returns a plain float.
    """
    if not getattr(prediction, "output", "").strip():
        if pred_name is not None:
            return dspy.Prediction(score=0.0, feedback="score=0.00; response was empty")
        return 0.0

    model = _get_model()

    if model is None:
        # Graceful fallback: word-overlap recall
        expected = set(example.expected_behavior.lower().split())
        output = set(prediction.output.lower().split())
        score = len(expected & output) / len(expected) if expected else 0.5
        score = min(1.0, max(0.0, score))
        if pred_name is not None:
            return dspy.Prediction(
                score=score,
                feedback=(f"score={score:.2f}; sentence_transformers not installed "
                          f"— fell back to word-overlap recall"),
            )
        return score

    emb_exp = model.encode(example.expected_behavior, normalize_embeddings=True)
    emb_out = model.encode(prediction.output, normalize_embeddings=True)
    raw = _cosine_sim(emb_exp, emb_out)           # [-1, 1]
    score = min(1.0, max(0.0, (raw + 1.0) / 2.0))  # → [0, 1]

    if pred_name is not None:
        return dspy.Prediction(
            score=score,
            feedback=f"score={score:.2f}; semantic cosine similarity={raw:.3f}",
        )
    return score
