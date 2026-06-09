# nodes/grade.py

import os
from typing import Any

from src.recruiter_copilot.state import RecruiterCopilotState


def grade_retrieved_docs_node(state: RecruiterCopilotState) -> dict:
    retrieved_docs = state.get("retrieved_docs", [])
    if not retrieved_docs:
        return {
            "scored_candidates": [],
            "selected_candidates": [],
            "selected_count": 0,
            "retrieval_count": 0,
        }

    requested_k = max(int(state.get("requested_k") or 10), 1)
    multiplier = max(int(os.getenv("GRADE_MULTIPLIER", "3")), 1)
    filters = state.get("extracted_filters", {}) or {}
    target_candidate_ids = {str(x).lower() for x in state.get("target_candidate_ids", [])}
    target_candidate_files = {str(x).lower() for x in state.get("target_candidate_files", [])}

    def normalize_text(doc: dict[str, Any]) -> str:
        metadata = doc.get("metadata", {}) or {}
        content = doc.get("content", "") or ""
        meta_text = " ".join(str(v) for v in metadata.values())
        return f"{content} {meta_text}".lower()

    def get_doc_distance(doc: dict[str, Any]) -> float:
        score = doc.get("score", 1.0)
        try:
            return float(score)
        except (TypeError, ValueError):
            return 1.0

    def get_candidate_id(doc: dict[str, Any]) -> str:
        metadata = doc.get("metadata", {}) or {}
        return str(
            metadata.get("candidate_id")
            or metadata.get("id")
            or doc.get("candidate_id")
            or ""
        ).strip()

    def get_resume_file(doc: dict[str, Any]) -> str:
        metadata = doc.get("metadata", {}) or {}
        return str(
            metadata.get("resume_file")
            or metadata.get("file_name")
            or metadata.get("source")
            or doc.get("resume_file")
            or ""
        ).strip()

    def add_filter_bonus(doc: dict[str, Any]) -> tuple[float, list[str]]:
        text = normalize_text(doc)
        metadata = doc.get("metadata", {}) or {}
        bonus = 0.0
        reasons: list[str] = []

        candidate_id = get_candidate_id(doc).lower()
        resume_file = get_resume_file(doc).lower()

        if candidate_id and candidate_id in target_candidate_ids:
            bonus += 0.30
            reasons.append("exact candidate_id match")

        if resume_file and resume_file in target_candidate_files:
            bonus += 0.30
            reasons.append("exact resume file match")

        loc = str(filters.get("location") or "").lower().strip()
        if loc and loc in text:
            bonus += 0.05
            reasons.append(f"location match: {loc}")

        for skill in filters.get("skills", []):
            skill_l = str(skill).lower().strip()
            if skill_l and skill_l in text:
                bonus += 0.03
                reasons.append(f"skill match: {skill_l}")

        for lang in filters.get("languages", []):
            lang_l = str(lang).lower().strip()
            if lang_l and lang_l in text:
                bonus += 0.02
                reasons.append(f"language match: {lang_l}")

        years_min = filters.get("years_min")
        if years_min:
            years_text = str(years_min).strip()
            if years_text and years_text in text:
                bonus += 0.03
                reasons.append(f"years signal: {years_text}")

        role = str(filters.get("role") or "").lower().strip()
        if role and role in text:
            bonus += 0.04
            reasons.append(f"role match: {role}")

        return bonus, reasons

    scored_candidates = []
    for doc in retrieved_docs:
        distance = get_doc_distance(doc)
        base_similarity = 1.0 - distance
        bonus, bonus_reasons = add_filter_bonus(doc)
        final_score = base_similarity + bonus

        enriched_doc = dict(doc)
        enriched_doc["candidate_id"] = get_candidate_id(doc)
        enriched_doc["resume_file"] = get_resume_file(doc)
        enriched_doc["_distance"] = distance
        enriched_doc["_base_similarity"] = base_similarity
        enriched_doc["_bonus"] = bonus
        enriched_doc["_score"] = final_score
        enriched_doc["_score_reasons"] = bonus_reasons

        scored_candidates.append(enriched_doc)

    scored_candidates = sorted(
        scored_candidates,
        key=lambda d: (
            d.get("_score", 0.0),
            d.get("_base_similarity", 0.0),
        ),
        reverse=True,
    )

    broad_limit = max(requested_k * multiplier, requested_k)
    scored_candidates = scored_candidates[:broad_limit]
    selected_candidates = scored_candidates[:requested_k]

    return {
        "scored_candidates": scored_candidates,
        "selected_candidates": selected_candidates,
        "selected_count": len(selected_candidates),
        "retrieval_count": len(scored_candidates),
    }