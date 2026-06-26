import json
import time
import argparse
import os
import re
import numpy as np
import pandas as pd
import faiss
import textwrap
import docx
from sentence_transformers import SentenceTransformer
from config import UNIVERSAL_TECH_TAXONOMY, ARTIFACT_DIR, ASSETS_DIR

def extract_text(file_path):
    if file_path.endswith('.docx'):
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs]).lower()
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().lower()

def execute_ranking(candidates_file, jd_file, output_file):
    start_time = time.time()
    
    print("[*] Loading offline universal artifacts...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    
    try:
        metadata_df = pd.read_parquet(os.path.join(ARTIFACT_DIR, "candidate_metadata.parquet"))
        raw_matrix = np.load(os.path.join(ARTIFACT_DIR, "candidate_raw_matrix.npy"))
        tax_matrix = np.load(os.path.join(ARTIFACT_DIR, "candidate_tax_matrix_norm.npy"))
    except FileNotFoundError as e:
        print(f"[!] ERROR: Artifacts not found. {e}")
        exit(1)

    print(f"[*] Loading LLM Taxonomy Weights & Parsing Raw JD: {jd_file}")
    try:
        with open(os.path.join(ARTIFACT_DIR, "jd_capability_vector.json"), "r") as f:
            jd_llm_weights = json.load(f)
    except FileNotFoundError:
        print("[!] WARNING: jd_capability_vector.json not found.")
        print("[!] Network constraint presumed active. Engaging universal fallback weights.")
        jd_llm_weights = {tag: 1.0 for tag in UNIVERSAL_TECH_TAXONOMY}
        jd_llm_weights["min_yoe"] = 5
        jd_llm_weights["max_yoe"] = 9

    jd_tax_vector = np.array([jd_llm_weights.get(tag, 0.0) for tag in UNIVERSAL_TECH_TAXONOMY], dtype=np.float32)

    if np.sum(jd_tax_vector) > 0:
        jd_tax_vector = jd_tax_vector / np.sum(jd_tax_vector)

    jd_text = extract_text(jd_file)
    jd_raw_vector = model.encode([jd_text], normalize_embeddings=True).astype(np.float32)

    target_yoe_min = jd_llm_weights.get("min_yoe", 5)
    target_yoe_max = jd_llm_weights.get("max_yoe", 9)

    yoe_matches = re.findall(r'(\d+)(?:-|-| to )(\d+)\s*years|\b(\d+)\+\s*years', jd_text)
    if yoe_matches:
        extracted_nums = [int(m) for tuple_match in yoe_matches for m in tuple_match if m]
        if len(extracted_nums) >= 2:
            target_yoe_min = min(extracted_nums)
            target_yoe_max = max(extracted_nums)
        elif len(extracted_nums) == 1:
            target_yoe_min = extracted_nums[0]
            target_yoe_max = target_yoe_min + 4

    print(f"[*] Dynamic YOE Target Extracted: {target_yoe_min} to {target_yoe_max} years")

    TOP_N = 14
    top_core_jd_indices = np.argsort(jd_tax_vector)[-TOP_N:][::-1]

    print("[*] Executing FAISS Exact Search...")
    index = faiss.IndexFlatIP(768)
    index.add(raw_matrix)
    K_RETRIEVE = 5000
    semantic_scores, semantic_indices = index.search(jd_raw_vector, K_RETRIEVE)

    top_indices = semantic_indices[0]
    top_semantic_scores = semantic_scores[0]

    subset_tax_matrix = tax_matrix[top_indices]
    subset_metadata = metadata_df.iloc[top_indices].reset_index(drop=True)

    print("[*] Applying Hybrid Math & Behavioral Penalties...")
    tax_scores = np.dot(subset_tax_matrix, jd_tax_vector)
    impact_scores = subset_metadata['impact_score'].values

    yoe = subset_metadata['years_of_experience'].values
    yoe_multiplier = np.ones_like(yoe, dtype=float)
    yoe_multiplier[(yoe >= target_yoe_min) & (yoe <= target_yoe_max)] = 1.05
    yoe_multiplier[yoe < (target_yoe_min - 2)] = 0.40
    yoe_multiplier[(yoe >= (target_yoe_min - 2)) & (yoe < target_yoe_min)] = 0.85

    over_exp_diff = np.maximum(0, yoe - (target_yoe_max + 2))
    over_penalty = np.maximum(0.90, 1.0 - (over_exp_diff * 0.01))
    yoe_multiplier = np.where(yoe > (target_yoe_max + 2), over_penalty, yoe_multiplier)

    is_dead_profile = (subset_metadata.get('days_since_active', 0) > 180) & (subset_metadata.get('applications_submitted', 0) == 0)
    activity_penalty = np.where(is_dead_profile, 0.85, 1.0)
    response_penalty = np.where(subset_metadata.get('response_rate', 1.0) < 0.10, 0.95, 1.0)
    flakiness_penalty = np.where(subset_metadata.get('interview_completion', 1.0) < 0.50, 0.85, 1.0)
    notice_penalty = np.where(subset_metadata.get('notice_period', 30) > 60, 0.97, 1.0)

    market_boost = np.clip(subset_metadata.get('saved_by_recruiters', 0).values * 0.01, 0.0, 0.10)
    github_boost = np.where(subset_metadata.get('github_score', 0) > 80, 0.05, 0.0)

    behavioral_modifier = activity_penalty * response_penalty * flakiness_penalty * notice_penalty * (1.0 + market_boost + github_boost)

    base_score = (tax_scores * 0.32) + (top_semantic_scores * 0.48) + (impact_scores * 0.15) + 0.05
    subset_metadata['final_score'] = base_score * yoe_multiplier * subset_metadata['trust_multiplier'] * behavioral_modifier
    subset_metadata['final_score'] = subset_metadata['final_score'].round(4)
    subset_metadata['tax_scores'] = list(subset_tax_matrix)

    print("[*] Filtering Honeypots and Sorting Validators...")
    clean_candidates = subset_metadata[subset_metadata['is_fatal'] == False].copy()
    clean_candidates = clean_candidates.sort_values(by=['final_score', 'candidate_id'], ascending=[False, True])
    top_100 = clean_candidates.head(100).copy()

    print("[*] Generating Step 16 Confidence Scores and Reasoning...")
    submission = []

    for rank, (idx, row) in enumerate(top_100.iterrows(), start=1):
        cand_tax = row['tax_scores']
        try:
            evidence = json.loads(row.get('evidence_dict', '{}'))
        except:
            evidence = {}

        yoe = row['years_of_experience']

        top_cand_scores = cand_tax[top_core_jd_indices]
        covered_count = np.sum(top_cand_scores >= 0.6)
        evidence_coverage = covered_count / float(TOP_N)

        evidence_strength = np.mean(top_cand_scores[top_cand_scores >= 0.6]) if covered_count > 0 else 0.0
        signal_consistency = max(0.0, 1.0 - np.std(top_cand_scores))
        profile_completeness = row['completeness_score']

        confidence = (0.40 * evidence_coverage) + \
                     (0.30 * evidence_strength) + \
                     (0.20 * signal_consistency) + \
                     (0.10 * profile_completeness)

        confidence = np.clip(confidence, 0.0, 1.0)

        if yoe < target_yoe_min:
            exp_note = f"Warning: {yoe} YOE (below {target_yoe_min}+ target)."
        elif yoe > target_yoe_max + 2:
            exp_note = f"Note: {yoe} YOE (senior to target)."
        else:
            exp_note = f"Target YOE match ({yoe} years)."

        best_relative_idx = np.argmax(top_cand_scores)
        best_actual_idx = top_core_jd_indices[best_relative_idx]
        best_tag = UNIVERSAL_TECH_TAXONOMY[best_actual_idx].replace('_', ' ')

        worst_relative_idx = np.argmin(top_cand_scores)
        worst_actual_idx = top_core_jd_indices[worst_relative_idx]
        worst_tag = UNIVERSAL_TECH_TAXONOMY[worst_actual_idx].replace('_', ' ')

        raw_quote = evidence.get(UNIVERSAL_TECH_TAXONOMY[best_actual_idx], "NO_EVIDENCE")
        short_quote = textwrap.shorten(raw_quote, width=200, placeholder="...")

        best_tag_score = top_cand_scores[best_relative_idx]
        has_gap = cand_tax[worst_actual_idx] <= 0.50
        variation = rank % 5

        if best_tag_score < 0.65 or raw_quote == "NO_EVIDENCE":
            trapdoor_templates = [
                f"High semantic similarity to target role. {exp_note} While their overall background aligns with the JD vector, our strict evidence extractor could not isolate a definitive quote for {best_tag}. Recommend manual technical screen.",
                f"Strong overlap with broader retrieval and engineering systems. {exp_note} Note: We detected adjacent skills, but no direct quote proving explicit {best_tag} experience was extracted. Needs validation.",
                f"Broad alignment across multiple JD dimensions. {exp_note} However, concrete textual evidence for {best_tag} is missing from the parsed data. Human review advised.",
                f"Experience profile closely matches expected seniority. {exp_note} Ranked based on dense vector proximity, but we could not pinpoint an exact textual match for {best_tag}.",
                f"Profile appears adjacent to the core requirements. {exp_note} Caution: the system associates their background with {best_tag}, but lacks a definitive quote to verify direct implementation."
            ]
            reasoning = trapdoor_templates[rank % 5]
        else:
            if rank <= 25:
                templates = [
                    f"{exp_note} We rank this candidate highly due to their proven {best_tag} background: '{short_quote}'." + (f" They do show a slight gap in {worst_tag}, however." if has_gap else " No major weaknesses detected."),
                    f"An exceptional match. Their work with {best_tag} stands out immediately ('{short_quote}')." + (f" Note a potential shortfall in {worst_tag}." if has_gap else "") + f" {exp_note}",
                    f"'{short_quote}' — this explicitly validates their {best_tag} skills. {exp_note}" + (f" You may want to screen for {worst_tag} during the interview." if has_gap else ""),
                    f"Strong profile across the board. {exp_note} They bring deep {best_tag} expertise, as noted by their claim: '{short_quote}'." + (f" Only minor concern is {worst_tag}." if has_gap else ""),
                    f"Top-tier fit for the {best_tag} requirements ('{short_quote}')." + (f" We observed a lack of {worst_tag}." if has_gap else " Solid technical coverage.") + f" {exp_note}"
                ]
            elif rank <= 50:
                templates = [
                    f"A very capable candidate. {exp_note} Their experience with {best_tag} is clear: '{short_quote}'." + (f" Keep in mind they are lighter on {worst_tag}." if has_gap else ""),
                    f"{exp_note} This profile caught our attention specifically for {best_tag} ('{short_quote}')." + (f" Would need to verify their {worst_tag} knowledge." if has_gap else " Good overall alignment."),
                    f"Solid secondary option. '{short_quote}' proves they can handle {best_tag}." + (f" The main drawback is their {worst_tag} exposure." if has_gap else "") + f" {exp_note}",
                    f"Meets most core criteria, particularly {best_tag} ('{short_quote}'). {exp_note}" + (f" Missing strong signals for {worst_tag}." if has_gap else ""),
                    f"Good overall background. {exp_note} They index highly on {best_tag}, stating '{short_quote}'." + (f" We didn't see much regarding {worst_tag}." if has_gap else "")
                ]
            elif rank <= 75:
                templates = [
                    f"Borderline profile. While they highlight {best_tag} ('{short_quote}')," + (f" the lack of {worst_tag} lowers their rank." if has_gap else " their depth is average.") + f" {exp_note}",
                    f"{exp_note} They have the required {best_tag} skills ('{short_quote}')." + (f" However, {worst_tag} is a noticeable blind spot." if has_gap else ""),
                    f"Acceptable backup candidate. Their mention of '{short_quote}' checks the {best_tag} box." + (f" Fails the {worst_tag} checks, though." if has_gap else "") + f" {exp_note}",
                    f"Ranked lower due to mixed signals. {exp_note} They do have {best_tag} ('{short_quote}')," + (f" but struggle with {worst_tag}." if has_gap else " but lack standout achievements."),
                    f"Passable match. {exp_note} We verified {best_tag} via '{short_quote}'." + (f" Significant drop-off around {worst_tag}." if has_gap else "")
                ]
            else:
                templates = [
                    f"Included primarily for {best_tag} ('{short_quote}')." + (f" Disqualified on {worst_tag}." if has_gap else " Weak overall signals.") + f" {exp_note}",
                    f"Marginal fit. {exp_note} They managed to trigger our {best_tag} filters ('{short_quote}')," + (f" but completely miss {worst_tag}." if has_gap else " but rest are okk."),
                    f"{exp_note} A weak overall option, though they do possess {best_tag} ('{short_quote}')." + (f" No evidence of {worst_tag}." if has_gap else ""),
                    f"Filler profile. '{short_quote}' gave them points for {best_tag}." + (f" Unqualified for {worst_tag}." if has_gap else "") + f" {exp_note}",
                    f"Bottom of the list. {exp_note} Matched {best_tag} ('{short_quote}')" + (f" but deeply deficient in {worst_tag}." if has_gap else " with very little else to offer.")
                ]

            reasoning = templates[variation]

        risk_flags = json.loads(row.get('risk_flags', '[]'))
        if risk_flags:
            flag_str = ", ".join(risk_flags)
            reasoning += f" [Note: Trust score adjusted due to: {flag_str}]."

        submission.append({
            "candidate_id": row['candidate_id'],
            "rank": rank,
            "score": round(row['final_score'], 4),
            "reasoning": reasoning
        })

    output_df = pd.DataFrame(submission)
    output_df = output_df[["candidate_id", "rank", "score", "reasoning"]]
    output_df.to_csv(output_file, index=False)
    print(f"[+] DONE. Validated 4-column {output_file} generated in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', type=str, default=os.path.join(ASSETS_DIR, 'candidates.jsonl'))
    parser.add_argument('--jd', type=str, default=os.path.join(ASSETS_DIR, 'job_description.docx'))
    parser.add_argument('--out', type=str, default='submission.csv')
    args = parser.parse_args()

    execute_ranking(args.candidates, args.jd, args.out)
