from datetime import datetime, timedelta

def evaluate_candidate_trust(candidate):
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    current_title = profile.get("current_title", "").lower()
    claimed_yoe = profile.get("years_of_experience", 0)

    result = {
        "fatal": False,
        "trust_multiplier": 1.0,
        "risk_flags": []
    }

    penalty = 0.0

    expert_zero_count = sum(1 for s in skills if s.get("proficiency", "").lower() == "expert" and s.get("duration_months", 0) <= 1)
    if expert_zero_count >= 3:
        result["fatal"] = True
        return result

    for s in skills:
        if s.get("duration_months", 0) > (claimed_yoe * 12) + 60:
            penalty += 0.10
            result["risk_flags"].append("Exaggerated Skill Duration")
            break

    for job in history:
        duration = job.get("duration_months")
        if duration is not None:
            if duration < 0:
                result["fatal"] = True
                return result
            if duration > (claimed_yoe * 12) + 36:
                result["fatal"] = True
                return result

    events = []
    for job in history:
        start_str = job.get("start_date")
        end_str = job.get("end_date")
        duration_m = job.get("duration_months", 0)
        if not start_str: continue
        try:
            start_dt = datetime.strptime(start_str, "%Y-%m-%d")
            if end_str:
                end_dt = datetime.strptime(end_str, "%Y-%m-%d")
            else:
                end_dt = start_dt + timedelta(days=int(duration_m * 30.4375))

            events.append((start_dt, 1))
            events.append((end_dt, -1))
        except ValueError: 
            continue

    events.sort(key=lambda x: (x[0], x[1]))
    current_concurrent, max_concurrent = 0, 0
    for date, delta in events:
        current_concurrent += delta
        max_concurrent = max(max_concurrent, current_concurrent)

    if max_concurrent >= 4:
        result["fatal"] = True
        return result

    if max_concurrent == 3:
        penalty += 0.15
        result["risk_flags"].append("High Concurrent Employment")

    valid_start_dates, valid_end_dates = [], []
    for job in history:
        start_str = job.get("start_date")
        end_str = job.get("end_date")
        duration_m = job.get("duration_months", 0)
        try:
            if start_str:
                start_dt = datetime.strptime(start_str, "%Y-%m-%d")
                valid_start_dates.append(start_dt)

                if end_str:
                    valid_end_dates.append(datetime.strptime(end_str, "%Y-%m-%d"))
                else:
                    valid_end_dates.append(start_dt + timedelta(days=int(duration_m * 30.4375)))
        except ValueError: 
            continue

    if valid_start_dates and valid_end_dates:
        timeline_years = (max(valid_end_dates) - min(valid_start_dates)).days / 365.25
        if timeline_years > claimed_yoe + 4:
            penalty += 0.10
            result["risk_flags"].append("Significant Career Gap / YOE Mismatch")

    for s in skills:
        if s.get("proficiency", "").lower() == "expert" and s.get("duration_months", 0) < 12:
            penalty += 0.10
            result["risk_flags"].append("Premature Expert Claims")
            break

    advanced_count = sum(1 for s in skills if s.get("proficiency", "").lower() in ["advanced", "expert"])
    if claimed_yoe < 3 and advanced_count >= 15:
        penalty += 0.05
        result["risk_flags"].append("Unlikely Skill Inflation")

    non_tech_titles = ["accountant", "hr", "marketing", "sales", "recruiter", "finance", "admin"]
    if any(t in current_title for t in non_tech_titles) and not any(any(kw in str(j.get("title", "")).lower() for kw in ["engineer", "developer", "data"]) for j in history):
        if sum(1 for s in skills if s.get("name", "").lower() in ["rag", "kubernetes", "llm", "pinecone"]) >= 3:
            penalty += 0.20
            result["risk_flags"].append("Keyword Stuffing (Non-Tech History)")

    consulting_firms = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"]
    if any(any(f in str(j.get("company", "")).lower() for f in consulting_firms) for j in history) and not any(not any(f in str(j.get("company", "")).lower() for f in consulting_firms) for j in history):
        penalty += 0.10
        result["risk_flags"].append("Pure Consulting Background")

    if len(history) >= 3 and sum(1 for j in history if j.get("duration_months", 999) <= 18) >= 3 and sum(1 for j in history if any(t in str(j.get("title", "")).lower() for t in ["senior", "staff", "principal"])) >= 2:
        penalty += 0.10
        result["risk_flags"].append("High Velocity Title Chasing")

    title_ranks = {"intern": 1, "junior": 2, "engineer": 3, "senior": 4, "lead": 5, "staff": 6, "principal": 7}
    if len(history) >= 2:
        sorted_history = sorted([j for j in history if j.get("start_date")], key=lambda x: x.get("start_date", ""))
        for i in range(len(sorted_history) - 1):
            old_r = next((r for kw, r in title_ranks.items() if kw in str(sorted_history[i].get("title", "")).lower()), None)
            new_r = next((r for kw, r in title_ranks.items() if kw in str(sorted_history[i+1].get("title", "")).lower()), None)
            if old_r and new_r and (new_r - old_r >= 5) and claimed_yoe < 5:
                penalty += 0.10
                result["risk_flags"].append("Unlikely Title Jump")
                break

    result["trust_multiplier"] = max(0.30, 1.0 - penalty)
    return result
