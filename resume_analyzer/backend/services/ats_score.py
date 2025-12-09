import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_STOPWORDS = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "be", "from", "that", "this"}

def clean_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _extract_sections(text: str):
    """
    Split resume into simple sections by common headings. Returns dict of section_name -> text.
    Simple heuristic: lines starting with known headings.
    """
    headings = ["skills", "technical skills", "experience", "work experience", "education", "projects", "summary"]
    lines = text.splitlines()
    current = "body"
    sections = {current: []}
    for ln in lines:
        l = ln.strip().lower()
        if any(l.startswith(h) for h in headings):
            current = l.split()[0]  # e.g., "skills"
            sections[current] = []
        else:
            sections.setdefault(current, []).append(ln)
    # join
    return {k: " ".join(v) for k, v in sections.items()}

def _get_ngrams(token_list, n):
    return [" ".join(token_list[i:i+n]) for i in range(len(token_list)-n+1)]

def calculate_ats_score(
    resume_text: str,
    job_description: str,
    synonyms: dict = None,
    fuzzy_cutoff: float = 0.8,
    weights: dict = None
) -> dict:
    """
    Improved ATS scoring.
    Args:
      resume_text, job_description: raw strings
      synonyms: optional mapping like {"js": ["javascript"], "ml": ["machine learning"]}
      fuzzy_cutoff: similarity threshold (0-1) for fuzzy matching
      weights: section weights, e.g. {"skills": 1.5, "experience": 1.2, "body": 1.0}
    Returns:
      dict with detailed report and final score (0-100)
    """
    if not resume_text or not job_description:
        return {"final_ats_score": 0.0, "match_percentage": 0.0, "similarity_score": 0.0, "detail": []}

    synonyms = synonyms or {}
    weights = weights or {"skills": 1.5, "experience": 1.2, "body": 1.0}

    resume_raw = resume_text
    jd_raw = job_description

    resume_clean = clean_text(resume_raw)
    jd_clean = clean_text(jd_raw)

    # Build JD keyword candidates using unigrams + bigrams
    jd_tokens = jd_clean.split()
    jd_unigrams = [t for t in jd_tokens if t not in DEFAULT_STOPWORDS and len(t) > 2]
    jd_bigrams = _get_ngrams(jd_tokens, 2)
    # filter bigrams with stopwords on both ends removed
    jd_bigrams = [bg for bg in jd_bigrams if all(tok not in DEFAULT_STOPWORDS and len(tok) > 2 for tok in bg.split())]
    keywords = sorted(set(jd_unigrams + jd_bigrams))

    if not keywords:
        return {"final_ats_score": 0.0, "match_percentage": 0.0, "similarity_score": 0.0, "detail": []}

    resume_tokens = resume_clean.split()
    # also precompute resume ngrams to match multiword skills
    resume_bigrams = _get_ngrams(resume_tokens, 2)
    resume_ngrams_set = set(resume_tokens + resume_bigrams)

    # Section detection to weight matches found inside 'skills' or 'experience'
    sections = _extract_sections(resume_raw)
    section_texts = {k: clean_text(v) for k, v in sections.items()}

    detail = []
    total_weight = 0.0
    matched_weight = 0.0

    for kw in keywords:
        found = False
        found_in = None
        match_score = 0.0  # 0..1

        # 1) exact ngram match
        if kw in resume_ngrams_set:
            found = True
            match_score = 1.0
            # find which section contains it (prefer 'skills')
            for sname, stext in section_texts.items():
                if kw in stext.split():
                    found_in = sname
                    break
            if not found_in:
                found_in = "body"

        # 2) synonyms mapping
        if (not found) and synonyms:
            # synonyms can map short -> list of equivalent tokens
            for key, syn_list in synonyms.items():
                if kw == key or kw in syn_list:
                    # check if any synonym appears in resume text
                    for syn in ([key] + syn_list):
                        if syn in resume_ngrams_set:
                            found = True
                            match_score = 1.0
                            found_in = "body"
                            break
                    if found:
                        break

        # 3) fuzzy matching: allow small typos / morphological variants
        if (not found):
            # compare against resume unigrams and bigrams
            candidates = list(resume_ngrams_set)
            close = difflib.get_close_matches(kw, candidates, n=1, cutoff=fuzzy_cutoff)
            if close:
                found = True
                found_in = "body"
                # measure normalized closeness
                # difflib ratio between strings
                match_score = difflib.SequenceMatcher(None, kw, close[0]).ratio()

        # Determine section weight
        sec_w = 1.0
        if found_in:
            sec_w = weights.get(found_in, 1.0)
        else:
            # attempt quick presence in skills section
            if "skills" in section_texts and kw in section_texts["skills"]:
                sec_w = weights.get("skills", 1.5)
                found = True
                match_score = max(match_score, 1.0)
                found_in = "skills"

        # accumulate
        kw_weight = sec_w  # each keyword weighted by section importance
        total_weight += kw_weight
        if found:
            matched_weight += kw_weight * match_score

        detail.append({
            "keyword": kw,
            "found": found,
            "found_in": found_in or "",
            "match_score": round(match_score, 3),
            "weight": kw_weight
        })

    # Compute match percentage as weighted ratio
    match_percentage = (matched_weight / total_weight) * 100 if total_weight > 0 else 0.0

    # Cosine similarity (TF-IDF) with ngrams to capture phrases
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english').fit_transform([resume_clean, jd_clean])
        vectors = vectorizer.toarray()
        cosine_sim = cosine_similarity(vectors)[0][1]
        similarity_score = cosine_sim * 100
    except Exception as e:
        logger.error(f"Cosine similarity failed: {e}")
        similarity_score = 0.0

    # Final ATS score: configurable blend (here 60% keyword match, 40% semantic)
    final_ats_score = (match_percentage * 0.6) + (similarity_score * 0.4)

    # Sort missing keywords (highest weight & not found)
    missing = [d for d in detail if not d["found"]]
    missing_sorted = sorted(missing, key=lambda x: x["weight"], reverse=True)

    return {
        "match_percentage": round(match_percentage, 2),
        "missing_keywords": [m["keyword"] for m in missing_sorted][:15],
        "similarity_score": round(similarity_score, 2),
        "final_ats_score": round(final_ats_score, 2),
        "detail": detail
    }
