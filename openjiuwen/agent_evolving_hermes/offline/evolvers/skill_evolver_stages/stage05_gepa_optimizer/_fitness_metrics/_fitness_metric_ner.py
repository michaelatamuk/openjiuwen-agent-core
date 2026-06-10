from __future__ import annotations

from typing import Set

import dspy

try:
    import spacy
    try:
        _NLP = spacy.load("en_core_web_sm")
        _SPACY_AVAILABLE = True
    except OSError:
        # Model not downloaded — pip install spacy && python -m spacy download en_core_web_sm
        _NLP = None
        _SPACY_AVAILABLE = False
except ImportError:
    _NLP = None
    _SPACY_AVAILABLE = False


def _extract_entities(text: str) -> Set[str]:
    """Extract named entities from *text* using spaCy (lowercased, stripped)."""
    if not _SPACY_AVAILABLE or _NLP is None:
        return set()
    doc = _NLP(text)
    return {ent.text.lower().strip() for ent in doc.ents if ent.text.strip()}


def _fallback_keywords(text: str) -> Set[str]:
    """Capitalised-word heuristic — proxy when spaCy is unavailable.

    Captures proper nouns and title-case terms without an NLP model.
    Strips punctuation, requires length > 2 to avoid single-letter artefacts.
    """
    words: Set[str] = set()
    for w in text.split():
        w_clean = w.strip(".,!?;:\"'()[]{}\\/")
        if w_clean and w_clean[0].isupper() and len(w_clean) > 2:
            words.add(w_clean.lower())
    return words


def fitness_metric(example: dspy.Example,
                   prediction: dspy.Prediction,
                   trace=None,
                   pred_name=None,
                   pred_trace=None):
    """Named-entity coverage metric.

    Extracts named entities (persons, organisations, locations, dates,
    technical terms, etc.) from ``example.expected_behavior`` using
    spaCy ``en_core_web_sm``, then measures what fraction appear in
    ``prediction.output`` using a recall-biased F1:

    ::

        score = 0.7 × recall + 0.3 × precision

    Recall is weighted higher because the rubric defines required entities
    the agent *must* mention.  Precision guards against completely irrelevant
    entity-stuffing.

    Why entities, not all words
    ---------------------------
    ``f1`` treats all content words equally.  Entities (named things) are
    typically the highest-information terms in a rubric — mentioning the
    correct name, organisation, or technology matters more than matching
    common domain vocabulary.  This metric isolates that signal.

    Graceful degradation
    --------------------
    If spaCy is not installed, falls back to capitalised-word extraction.
    If neither the rubric nor the output has any entities, returns 0.5.

    GEPA feedback
    -------------
    When called from GEPA (``pred_name`` is not None), returns a
    ``dspy.Prediction(score, feedback)`` listing missing entity names so
    the reflection LM can request they be explicitly mentioned.  When
    called from MIPROv2 or the evaluation harness (``pred_name`` is None),
    returns a plain float.
    """
    if not getattr(prediction, "output", "").strip():
        if pred_name is not None:
            return dspy.Prediction(score=0.0, feedback="score=0.00; response was empty")
        return 0.0

    if _SPACY_AVAILABLE:
        expected_entities = _extract_entities(example.expected_behavior)
        output_entities = _extract_entities(prediction.output)
        method = "spaCy NER"
    else:
        expected_entities = _fallback_keywords(example.expected_behavior)
        output_entities = _fallback_keywords(prediction.output)
        method = "capitalised-word fallback (spaCy not installed)"

    if not expected_entities:
        score = 0.5
        if pred_name is not None:
            return dspy.Prediction(
                score=score,
                feedback=f"score={score:.2f}; no named entities found in rubric ({method})",
            )
        return score

    covered = expected_entities & output_entities
    recall = len(covered) / len(expected_entities)
    precision = len(covered) / len(output_entities) if output_entities else 0.0
    score = min(1.0, max(0.0, 0.7 * recall + 0.3 * precision))

    if pred_name is not None:
        missing = sorted(expected_entities - output_entities)[:6]
        parts = [f"score={score:.2f}"]
        if missing:
            parts.append(f"missing entities: {', '.join(missing)}")
        else:
            parts.append("all entities covered")
        return dspy.Prediction(score=score, feedback="; ".join(parts))

    return score
