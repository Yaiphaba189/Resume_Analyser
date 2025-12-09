
import re
import logging
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Spacy model
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Spacy model 'en_core_web_sm' loaded.")
except OSError:
    logger.warning("Spacy model 'en_core_web_sm' not found. Downloading...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_email(text):
    """Refined email extraction regex."""
    # Matches typical emails.
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    match = re.search(email_pattern, text)
    return match.group(0) if match else None

def extract_phone(text):
    """Refined phone extraction."""
    # Matches various formats, prevents dates/ISO strings from matching
    # (123) 456-7890, 123-456-7890, +1 123 456 7890, 123 456 7890
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    match = re.search(phone_pattern, text)
    return match.group(0) if match else None

def extract_name(text):
    """Extract name using Spacy NER with filtering and robust fallbacks."""
    # Limit text processing
    doc = nlp(text[:2000]) 
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            # Basic validation
            # 1. 2-3 words (First Last, First Middle Last)
            # 2. No numbers (allow special chars like - or .)
            # 3. Not a common tech word
            clean_name = name.replace(" ", "").replace("-", "").replace(".", "")
            if 1 < len(name.split()) <= 4 and clean_name.isalpha():
                lower_name = name.lower()
                # Expanded exclude list
                exclude_list = [
                    "java", "python", "resume", "curriculum", "vitae", "summary", 
                    "profile", "contact", "experience", "software engineer", "developer",
                    "email", "phone", "address", "education", "skills", "projects"
                ]
                if lower_name not in exclude_list:
                    return name
    
    # Fallback: Look at the first few non-empty lines
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for i in range(min(5, len(lines))): # Check first 5 lines
        line = lines[i]
        # Heuristic: 2-4 words, mostly letters
        # Allow some extra flexibility but ensure it's not a common header
        clean_line = line.replace(" ", "").replace("-", "").replace(".", "")
        if 1 < len(line.split()) <= 4 and clean_line.isalpha():
            lower_line = line.lower()
            exclude_list = [
                "java", "python", "resume", "curriculum", "vitae", "summary", 
                "profile", "contact", "experience", "software engineer", "developer",
                "email", "phone", "address", "education", "skills", "projects"
            ]
            if lower_line not in exclude_list:
                return line
             
    return None

def extract_skills(text):
    """
    Extract skills based on a predefined list.
    """
    skills_list = [
        "python", "java", "c++", "c#", "javascript", "typescript", "html", "css",
        "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
        "sql", "mysql", "postgresql", "mongodb", "aws", "azure", "gcp", "docker",
        "kubernetes", "git", "machine learning", "deep learning", "tensorflow", "pytorch",
        "nlp", "pandas", "numpy", "scikit-learn", "data engineering", "devops",
        "tableau", "power bi", "excel", "spark", "hadoop", "linux"
    ]
    
    found_skills = set()
    text_lower = text.lower()
    for skill in skills_list:
        # Use regex boundary to match whole words only (e.g. avoid matching "java" in "javascript" if list order mattered, but 'java' is its own word)
        # escape skill for regex safety
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.add(skill)
            
    return list(found_skills)

def extract_sections(text):
    """
    Robust section extraction using Regex for headers.
    Returns a dict {section_name: content}.
    """
    # Map normalized section names to regex patterns
    # Patterns look for lines that are primarily the header text, possibly with : or - chars
    section_patterns = {
        "education": [r"^education", r"^academic background", r"^academic history"],
        "experience": [r"^experience", r"^work experience", r"^employment history", r"^work history", r"^professional experience", r"^prior experience"],
        "skills": [r"^skills", r"^technical skills", r"^technologies", r"^competencies", r"^core competencies"],
        "summary": [r"^summary", r"^professional summary", r"^profile", r"^objective", r"^about me", r"^interests"],
        "projects": [r"^projects", r"^personal projects"],
        "certifications": [r"^certifications", r"^licenses", r"^training", r"^courses"]
    }
    
    text_lower = text.lower()
    lines = text.split('\n')
    
    # Identify headers in the text lines
    # We store (line_index, section_name)
    header_indices = []
    
    for idx, line in enumerate(lines):
        clean_line = line.strip().lower()
        # Header heuristic:
        # 1. Short (< 50 chars)
        # 2. Matches a pattern
        # 3. Mostly letters (ignore "Page 1", dates)
        if len(clean_line) > 50 or len(clean_line) < 3:
            continue
            
        for sec_name, patterns in section_patterns.items():
            matched = False
            for pat in patterns:
                # Allow optionally trailing colon or spacing
                # The pattern matches the START of the line. 
                # e.g. "Experience :" or "Work Experience"
                full_pat = pat + r"[\s:]*$"
                if re.match(full_pat, clean_line):
                    header_indices.append((idx, sec_name))
                    matched = True
                    break
            if matched:
                break
                
    # Sort by line index
    header_indices.sort(key=lambda x: x[0])
    
    extracted = {}
    
    for i, (start_line_idx, section_name) in enumerate(header_indices):
        # End is the start of the next header, or end of file
        if i + 1 < len(header_indices):
            end_line_idx = header_indices[i+1][0]
        else:
            end_line_idx = len(lines)
            
        # Get content lines (excluding the header line itself)
        content_lines = lines[start_line_idx+1 : end_line_idx]
        
        # Join and strip
        content = "\n".join(content_lines).strip()
        
        # Store. If duplicates (e.g. multiple 'Experience' sections?), append.
        if section_name in extracted:
             extracted[section_name] += "\n" + content
        else:
             extracted[section_name] = content
             
    return extracted


def clean_date_lines(text):
    """
    Remove lines that are purely dates or date ranges.
    e.g. "2018", "Oct 2020 - Present", "2019-2021"
    """
    if not text:
        return text
        
    lines = text.split('\n')
    cleaned_lines = []
    
    # Regex patterns for standalone dates
    patterns = [
        r"^\s*\d{4}\s*$", # "2018"
        r"^\s*\d{4}\s*[-–]\s*(?:\d{4}|Present|present|Current|current)\s*$", # "2018 - 2020" or "2018 - Present" (matches en-dash or hyphen)
        r"^\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*.*\d{4}.*$" # "Oct 2020..."
    ]
    
    for line in lines:
        is_date = False
        for pat in patterns:
            if re.match(pat, line, re.IGNORECASE):
                is_date = True
                break
        
        if not is_date:
            cleaned_lines.append(line)
            
    return "\n".join(cleaned_lines).strip()

def parse_resume(text):
    """
    Main parser function.
    """
    sections = extract_sections(text)
    
    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text), # Keywords found in WHOLE text (or just skills section if we preferred)
        "summary": sections.get("summary"),
        "education": clean_date_lines(sections.get("education")),
        "experience": clean_date_lines(sections.get("experience")),
    }
