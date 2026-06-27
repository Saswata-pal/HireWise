import json
import os
import re
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from config import UNIVERSAL_TECH_TAXONOMY, ARTIFACT_DIR, ASSETS_DIR
from trust_engine import evaluate_candidate_trust

CURRENT_DATE = datetime(2026, 6, 17)

def run_pipeline(candidates_path):
    print("[*] Loading Model & Precomputing Taxonomy Vectors...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    tax_embeddings = model.encode(UNIVERSAL_TECH_TAXONOMY, normalize_embeddings=True)

    candidates = []
    with open(candidates_path, "r") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))

    metadata = []
    raw_vectors = []
    tax_score_matrix = []

    print(f"[*] Processing {len(candidates)} candidates using GPU Batching...")
    BATCH_SIZE = 1000

    for i in tqdm(range(0, len(candidates), BATCH_SIZE), desc="Processing Batches"):
        batch = candidates[i : i + BATCH_SIZE]

        all_sentences = []
        candidate_boundaries = []
        current_idx = 0

        for cand in batch:
            profile = cand.get("profile", {})
            history = cand.get("career_history", [])

            headline = profile.get("headline", "")
            summary = profile.get("summary", "")
            history_desc = " ".join([h.get("description", "") for h in history])
            full_text = f"{headline}. {summary}. {history_desc}"

            sentences = re.split(r'(?<=[.!?]) +', full_text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            if not sentences:
                sentences = ["No detailed text provided."]

            all_sentences.extend(sentences)
            candidate_boundaries.append((current_idx, current_idx + len(sentences)))
            current_idx += len(sentences)

        batch_sent_vecs = model.encode(all_sentences, batch_size=256, normalize_embeddings=True)

        for c_idx, cand in enumerate(batch):
            start, end = candidate_boundaries[c_idx]
            sent_vecs = batch_sent_vecs[start:end]

            sim_matrix = np.dot(sent_vecs, tax_embeddings.T)

            max_sim_scores = np.max(sim_matrix, axis=0)
            best_sentence_indices = np.argmax(sim_matrix, axis=0)

            supporting_counts = np.maximum(0, np.sum(sim_matrix > 0.65, axis=0) - 1)

            bonus_multiplier = 1.0 + np.clip(supporting_counts * 0.02, 0.0, 0.10)

            adjusted_tax_scores = np.clip(max_sim_scores * bonus_multiplier, 0.0, 1.0)

            cand_text_lower = " ".join(all_sentences[start:end]).lower()
            impact_matches = len(re.findall(r'(\d+%|\d+\s*(million|m|k|users|requests|tps)|reduced|increased|improved|saved)', cand_text_lower))
            impact_score = min(impact_matches / 10.0, 1.0)

            evidence_dict = {}
            for tax_idx in range(len(UNIVERSAL_TECH_TAXONOMY)):
                tag = UNIVERSAL_TECH_TAXONOMY[tax_idx]
                best_idx = best_sentence_indices[tax_idx]
                highest_score = max_sim_scores[tax_idx]

                if highest_score >= 0.65:
                    evidence_dict[tag] = all_sentences[start + best_idx]
                else:
                    evidence_dict[tag] = "NO_EVIDENCE"

            raw_vec = np.mean(sent_vecs, axis=0)
            raw_vec = raw_vec / np.linalg.norm(raw_vec)

            profile = cand.get("profile", {})
            redrob_signals = cand.get("redrob_signals", {})
            yoe = profile.get("years_of_experience", 0)

            last_active_str = redrob_signals.get("last_active_date")
            if last_active_str:
                try:
                    days_since_active = (CURRENT_DATE - datetime.strptime(last_active_str, "%Y-%m-%d")).days
                except ValueError:
                    days_since_active = 999
            else:
                days_since_active = 999

            trust_data = evaluate_candidate_trust(cand)

            metadata.append({
                "candidate_id": cand.get("candidate_id"),
                "is_fatal": trust_data["fatal"],
                "trust_multiplier": trust_data["trust_multiplier"],
                "risk_flags": json.dumps(trust_data["risk_flags"]),
                "years_of_experience": yoe,
                "days_since_active": days_since_active,
                "applications_submitted": redrob_signals.get("applications_submitted_30d", 0),
                "response_rate": redrob_signals.get("recruiter_response_rate", 0.0),
                "completeness_score": redrob_signals.get("profile_completeness_score", 50.0) / 100.0,
                "saved_by_recruiters": redrob_signals.get("saved_by_recruiters_30d", 0),
                "interview_completion": redrob_signals.get("interview_completion_rate", 1.0),
                "notice_period": redrob_signals.get("notice_period_days", 30),
                "github_score": redrob_signals.get("github_activity_score", -1),
                "impact_score": impact_score,
                "evidence_dict": json.dumps(evidence_dict)
            })

            raw_vectors.append(raw_vec)
            tax_score_matrix.append(adjusted_tax_scores)

    raw_matrix = np.vstack(raw_vectors).astype(np.float32)
    tax_matrix = np.vstack(tax_score_matrix).astype(np.float32)

    print("\n[*] Normalizing taxonomy scores...")
    tax_matrix_normalized = np.clip(tax_matrix, 0.0, 1.0)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    print("[*] Saving optimized artifacts to disk...")
    pd.DataFrame(metadata).to_parquet(os.path.join(ARTIFACT_DIR, "candidate_metadata.parquet"), index=False)
    np.save(os.path.join(ARTIFACT_DIR, "candidate_raw_matrix.npy"), raw_matrix)
    np.save(os.path.join(ARTIFACT_DIR, "candidate_tax_matrix_norm.npy"), tax_matrix_normalized)

    print("[+] Offline Processing Complete. Feature store updated.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', type=str, default=os.path.join(ASSETS_DIR, 'candidates.jsonl'))
    args = parser.parse_args()
    run_pipeline(args.candidates)
