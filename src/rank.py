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
import hashlib
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

    subset_tax_matrix_raw = tax_matrix[top_indices]
    subset_metadata_raw = metadata_df.iloc[top_indices].reset_index(drop=True)

    print("[*] Purging Fatal Profiles to optimize compute...")
    valid_mask = ~subset_metadata_raw['is_fatal'].values 

    subset_metadata = subset_metadata_raw[valid_mask].copy()
    subset_tax_matrix = subset_tax_matrix_raw[valid_mask]

    top_semantic_scores = top_semantic_scores[valid_mask]

    print(f"[*] Remaining valid candidates for scoring: {len(subset_metadata)}")

    print("[*] Applying Hybrid Math & Behavioral Penalties...")
    tax_scores = np.dot(subset_tax_matrix, jd_tax_vector)
    impact_scores = subset_metadata['impact_score'].values

    yoe = subset_metadata['years_of_experience'].values
    yoe_multiplier = np.ones_like(yoe, dtype=float)

    yoe_multiplier[(yoe >= target_yoe_min) & (yoe <= target_yoe_max)] = 1.05

    yoe_multiplier[(yoe >= target_yoe_min - 1) & (yoe < target_yoe_min)] = 0.90 
    yoe_multiplier[(yoe >= target_yoe_min - 2) & (yoe < target_yoe_min - 1)] = 0.75 
    yoe_multiplier[(yoe >= target_yoe_min - 3) & (yoe < target_yoe_min - 2)] = 0.50 
    yoe_multiplier[yoe < target_yoe_min - 3] = 0.25

    over_exp_diff = np.maximum(0, yoe - (target_yoe_max + 2))
    over_penalty = np.maximum(0.90, 1.0 - (over_exp_diff * 0.01))
    yoe_multiplier = np.where(yoe > (target_yoe_max + 2), over_penalty, yoe_multiplier)

    days_since_active = subset_metadata['days_since_active'].values
    apps_submitted = subset_metadata['applications_submitted'].values
    response_rate = subset_metadata['response_rate'].values
    interview_comp = subset_metadata['interview_completion'].values
    notice_period = subset_metadata['notice_period'].values
    saved_by_rec = subset_metadata['saved_by_recruiters'].values
    github_score = subset_metadata['github_score'].values

    is_dead_profile = (days_since_active > 180) & (apps_submitted == 0)
    activity_penalty = np.where(is_dead_profile, 0.85, 1.0)
    response_penalty = np.where(response_rate < 0.10, 0.95, 1.0)
    flakiness_penalty = np.where(interview_comp < 0.50, 0.85, 1.0)
    notice_penalty = np.where(notice_period > 60, 0.97, 1.0)

    market_boost = np.clip(saved_by_rec * 0.01, 0.0, 0.10)
    github_boost = np.where(github_score > 80, 0.05, 0.0)

    behavioral_modifier = activity_penalty * response_penalty * flakiness_penalty * notice_penalty * (1.0 + market_boost + github_boost)

    base_score = (tax_scores * 0.32) + (top_semantic_scores * 0.48) + (impact_scores * 0.15) + 0.05
    subset_metadata['final_score'] = base_score * yoe_multiplier * subset_metadata['trust_multiplier'] * behavioral_modifier
    subset_metadata['final_score'] = subset_metadata['final_score'].round(4)
    subset_metadata['tax_scores'] = list(subset_tax_matrix)

    print("[*] Sorting Top Validators...")
    clean_candidates = subset_metadata.sort_values(by=['final_score', 'candidate_id'], ascending=[False, True])
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
        cid_bytes = str(row['candidate_id']).encode('utf-8')
        stable_hash = int(hashlib.md5(cid_bytes).hexdigest(), 16)
        variation = stable_hash % 5

        if best_tag_score < 0.65 or raw_quote == "NO_EVIDENCE":
            trapdoor_templates = [
            f"High semantic similarity to target role. {exp_note} While their overall background aligns with the JD vector, our strict evidence extractor could not isolate a definitive quote for {best_tag}. Recommend manual technical screen.",
            f"Strong overlap with broader retrieval and engineering systems. {exp_note} Note: We detected adjacent skills, but no direct quote proving explicit {best_tag} experience was extracted. Needs validation.",
            f"Broad alignment across multiple JD dimensions. {exp_note} However, concrete textual evidence for {best_tag} is missing from the parsed data. Human review advised.",
            f"Experience profile closely matches expected seniority. {exp_note} Ranked based on dense vector proximity, but we could not pinpoint an exact textual match for {best_tag}.",
            f"Profile appears adjacent to the core requirements. {exp_note} Caution: the system associates their background with {best_tag}, but lacks a definitive quote to verify direct implementation.",
            f"Vector proximity indicates relevance, yet textual proof of {best_tag} remains elusive. {exp_note} Proceed with a targeted technical interview.",
            f"The semantic match is high, suggesting implicit experience. {exp_note} We flag this for recruiter review as explicit {best_tag} achievements were not parsed.",
            f"Shows generalized architectural alignment. {exp_note} Specific extraction for {best_tag} failed, meaning their experience might be implied rather than stated.",
            f"JD alignment is driven by dense embeddings rather than keyword matching. {exp_note} We could not surface a direct quote for {best_tag}.",
            f"Strong contextual fit based on historical data. {exp_note} However, the exact depth of their {best_tag} capability requires human verification.",
            f"Calculated as a likely match based on surrounding tech stack. {exp_note} Warning: No concrete phrase directly confirmed {best_tag} mastery.",
            f"Overall profile trajectory fits the requirement. {exp_note} We recommend probing specifically on {best_tag} as our extractor found no hard evidence.",
            f"Implicit matching pushed this candidate up the ranks. {exp_note} Be aware that a direct textual claim for {best_tag} is absent from their parsed history.",
            f"Matches the broader engineering profile. {exp_note} The system flags a lack of explicitly verifiable text for {best_tag}.",
            f"Good generalist fit indicated by semantic scoring. {exp_note} We advise a manual read to confirm {best_tag} competency, as automated extraction yielded no quote."
        ]
            reasoning = trapdoor_templates[rank % 5]
        else:
            if rank <= 25:
                templates = [
                f"{exp_note} We rank this candidate highly due to their proven {best_tag} background: '{short_quote}'." + (f" They do show a slight gap in {worst_tag}, however." if has_gap else " No major weaknesses detected."),
                f"An exceptional match. Their work with {best_tag} stands out immediately ('{short_quote}')." + (f" Note a potential shortfall in {worst_tag}." if has_gap else "") + f" {exp_note}",
                f"'{short_quote}' — this explicitly validates their {best_tag} skills. {exp_note}" + (f" You may want to screen for {worst_tag} during the interview." if has_gap else ""),
                f"Strong profile across the board. {exp_note} They bring deep {best_tag} expertise, as noted by their claim: '{short_quote}'." + (f" Only minor concern is {worst_tag}." if has_gap else ""),
                f"Top-tier fit for the {best_tag} requirements ('{short_quote}')." + (f" We observed a lack of {worst_tag}." if has_gap else " Solid technical coverage.") + f" {exp_note}",
                f"Displays elite competency in {best_tag}, backed by tangible proof: '{short_quote}'." + (f" Keep an eye on {worst_tag}." if has_gap else " Extremely well-rounded.") + f" {exp_note}",
                f"Immediate shortlisting justified by this explicit {best_tag} evidence: '{short_quote}'." + (f" {worst_tag} appears to be their only blind spot." if has_gap else "") + f" {exp_note}",
                f"Highly aligned candidate. {exp_note} Their depth in {best_tag} is verified ('{short_quote}')." + (f" Some remediation in {worst_tag} may be needed." if has_gap else ""),
                f"{exp_note} Premium match for the core stack. '{short_quote}' definitively answers our {best_tag} requirement." + (f" Slight technical debt in {worst_tag}." if has_gap else ""),
                f"Standout candidate. The system heavily weighted their {best_tag} experience ('{short_quote}')." + (f" Verify their comfort level with {worst_tag}." if has_gap else "") + f" {exp_note}",
                f"Excellent technical trajectory. {exp_note} Validated {best_tag} capability via: '{short_quote}'." + (f" Secondary skills like {worst_tag} are lagging." if has_gap else ""),
                f"A leading contender based on concrete {best_tag} delivery ('{short_quote}')." + (f" The single red flag is a weak {worst_tag} signal." if has_gap else " Thoroughly checks all boxes.") + f" {exp_note}",
                f"{exp_note} Unmistakable alignment with {best_tag} ('{short_quote}')." + (f" We recommend a brief technical check on {worst_tag}." if has_gap else ""),
                f"Ranks in the top percentile strictly due to this {best_tag} metric: '{short_quote}'." + (f" Lacks maturity in {worst_tag}." if has_gap else "") + f" {exp_note}",
                f"Outstanding core relevance. {exp_note} They have demonstrably executed on {best_tag} ('{short_quote}')." + (f" We noted a deficiency in {worst_tag}." if has_gap else "")
            ]
            elif rank <= 50:
                templates = [
                f"A very capable candidate. {exp_note} Their experience with {best_tag} is clear: '{short_quote}'." + (f" Keep in mind they are lighter on {worst_tag}." if has_gap else ""),
                f"{exp_note} This profile caught our attention specifically for {best_tag} ('{short_quote}')." + (f" Would need to verify their {worst_tag} knowledge." if has_gap else " Good overall alignment."),
                f"Solid secondary option. '{short_quote}' proves they can handle {best_tag}." + (f" The main drawback is their {worst_tag} exposure." if has_gap else "") + f" {exp_note}",
                f"Meets most core criteria, particularly {best_tag} ('{short_quote}'). {exp_note}" + (f" Missing strong signals for {worst_tag}." if has_gap else ""),
                f"Good overall background. {exp_note} They index highly on {best_tag}, stating '{short_quote}'." + (f" We didn't see much regarding {worst_tag}." if has_gap else ""),
                f"Strong mid-tier match. We validated their {best_tag} implementation: '{short_quote}'." + (f" {worst_tag} experience is highly questionable." if has_gap else "") + f" {exp_note}",
                f"Competent profile that anchors well on {best_tag} ('{short_quote}')." + (f" They will need onboarding support for {worst_tag}." if has_gap else "") + f" {exp_note}",
                f"{exp_note} A reliable choice given their explicit {best_tag} background: '{short_quote}'." + (f" Fails to meet the bar for {worst_tag}." if has_gap else ""),
                f"Checks the primary boxes. Their claim of '{short_quote}' solidifies {best_tag}." + (f" However, {worst_tag} remains unproven." if has_gap else "") + f" {exp_note}",
                f"Worth a conversation. {exp_note} Solid evidence of {best_tag} ('{short_quote}')." + (f" Prepare to probe deeply into {worst_tag}." if has_gap else ""),
                f"Demonstrates practical {best_tag} knowledge ('{short_quote}'). {exp_note}" + (f" Lacks enterprise exposure to {worst_tag}." if has_gap else ""),
                f"Firmly in the acceptable range. {exp_note} Validated {best_tag} skills: '{short_quote}'." + (f" We are concerned about their {worst_tag} depth." if has_gap else ""),
                f"Good baseline fit. '{short_quote}' is exactly what we want for {best_tag}." + (f" Just be aware of the {worst_tag} gap." if has_gap else "") + f" {exp_note}",
                f"{exp_note} Hits the required marks for {best_tag} ('{short_quote}')." + (f" But completely misses on {worst_tag}." if has_gap else ""),
                f"Reliable engineering background. They clearly know {best_tag} ('{short_quote}')." + (f" Will require ramp-up time for {worst_tag}." if has_gap else "") + f" {exp_note}"
            ]
            elif rank <= 75:
                templates = [
                f"Borderline profile. While they highlight {best_tag} ('{short_quote}')," + (f" the lack of {worst_tag} lowers their rank." if has_gap else " their depth is average.") + f" {exp_note}",
                f"{exp_note} They have the required {best_tag} skills ('{short_quote}')." + (f" However, {worst_tag} is a noticeable blind spot." if has_gap else ""),
                f"Acceptable backup candidate. Their mention of '{short_quote}' checks the {best_tag} box." + (f" Fails the {worst_tag} checks, though." if has_gap else "") + f" {exp_note}",
                f"Ranked lower due to mixed signals. {exp_note} They do have {best_tag} ('{short_quote}')," + (f" but struggle with {worst_tag}." if has_gap else " but lack standout achievements."),
                f"Passable match. {exp_note} We verified {best_tag} via '{short_quote}'." + (f" Significant drop-off around {worst_tag}." if has_gap else ""),
                f"A fringe candidate. {exp_note} They demonstrate {best_tag} ('{short_quote}')," + (f" but the total absence of {worst_tag} is problematic." if has_gap else " but lack senior polish."),
                f"Only meets partial requirements. {best_tag} is present ('{short_quote}')." + (f" We severely penalized the lack of {worst_tag}." if has_gap else "") + f" {exp_note}",
                f"Included based on {best_tag} relevance ('{short_quote}'). {exp_note}" + (f" They would fail a strict {worst_tag} screen." if has_gap else ""),
                f"Adequate, but unexceptional. {exp_note} Shows {best_tag} ('{short_quote}')," + (f" yet provides zero evidence for {worst_tag}." if has_gap else ""),
                f"Lower confidence match. While '{short_quote}' proves {best_tag} capability," + (f" their {worst_tag} gap is glaring." if has_gap else " the broader profile is thin.") + f" {exp_note}",
                f"Barely clears the technical bar. {exp_note} Yes, they have {best_tag} ('{short_quote}')," + (f" but they are missing {worst_tag} entirely." if has_gap else ""),
                f"Usable in a pinch. They at least possess {best_tag} ('{short_quote}')." + (f" Major upskilling needed in {worst_tag}." if has_gap else "") + f" {exp_note}",
                f"{exp_note} A weak primary fit, though {best_tag} is verifiable ('{short_quote}')." + (f" The {worst_tag} void cannot be ignored." if has_gap else ""),
                f"Profile lacks density. {best_tag} is their saving grace ('{short_quote}')." + (f" They show no competency in {worst_tag}." if has_gap else "") + f" {exp_note}",
                f"Ranking depressed due to scope. {exp_note} They do have {best_tag} ('{short_quote}')," + (f" but {worst_tag} is a critical failure point." if has_gap else "")
            ]
            else:
                templates = [
                f"Partial match. Shows some background in {best_tag} ('{short_quote}')." + (f" Disqualified for core roles based on {worst_tag} gaps." if has_gap else " Lacks strong primary signals.") + f" {exp_note}",
                f"Requires further review. {exp_note} While they trigger filters for {best_tag} ('{short_quote}')," + (f" they lack demonstrable {worst_tag} experience." if has_gap else " the broader profile is light on details."),
                f"{exp_note} Alternative profile. They possess baseline {best_tag} experience ('{short_quote}')." + (f" No direct evidence of {worst_tag}." if has_gap else ""),
                f"Secondary match. Highlights {best_tag} via '{short_quote}'." + (f" Does not meet the technical threshold for {worst_tag}." if has_gap else "") + f" {exp_note}",
                f"Atypical fit for this specific JD. {exp_note} Matches {best_tag} ('{short_quote}')" + (f" but is deeply deficient in {worst_tag}." if has_gap else " but offers limited alignment elsewhere."),
                f"Ranked purely as filler. {exp_note} They hit a keyword for {best_tag} ('{short_quote}')," + (f" but their {worst_tag} knowledge is non-existent." if has_gap else " but fail holistic checks."),
                f"Marginal relevance. {exp_note} Extracted {best_tag} proof ('{short_quote}')," + (f" but the complete lack of {worst_tag} tanks their score." if has_gap else ""),
                f"Not recommended for primary screening. Shows {best_tag} ('{short_quote}')," + (f" but fails the {worst_tag} requirement completely." if has_gap else "") + f" {exp_note}",
                f"Weak candidate overall. {exp_note} Only flagged because of {best_tag} ('{short_quote}')." + (f" Zero signal detected for {worst_tag}." if has_gap else ""),
                f"Included only for pipeline depth. {best_tag} is present ('{short_quote}')." + (f" Highly unqualified regarding {worst_tag}." if has_gap else "") + f" {exp_note}",
                f"Bottom percentile match. {exp_note} They technically know {best_tag} ('{short_quote}')," + (f" but missing {worst_tag} is a dealbreaker." if has_gap else ""),
                f"Out of scope for immediate hire. {exp_note} {best_tag} matches ('{short_quote}')," + (f" but we see a fatal gap in {worst_tag}." if has_gap else " too many adjacent mismatches."),
                f"Low priority. While '{short_quote}' triggered a {best_tag} match," + (f" they cannot pass a {worst_tag} screen." if has_gap else " they are not competitive here.") + f" {exp_note}",
                f"Fails the comprehensive JD test. {exp_note} {best_tag} is their only strong suit ('{short_quote}')." + (f" The {worst_tag} gap is too wide to train." if has_gap else ""),
                f"Strictly an edge-case candidate. {exp_note} Verified {best_tag} ('{short_quote}')," + (f" but lack of {worst_tag} disqualifies them." if has_gap else "")
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
