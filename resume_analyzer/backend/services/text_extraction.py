
import io
import os
import pypdf
import logging
import cv2
import numpy as np
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize EasyOCR
try:
    import easyocr
    # Initialize reader for English. 'gpu=True' by default.
    ocr_reader = easyocr.Reader(['en'], gpu=True) 
    logger.info("EasyOCR initialized successfully.")
except Exception as e:
    logger.warning(f"Failed to initialize EasyOCR: {e}. OCR features might not work.")
    ocr_reader = None

def extract_text_pypdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using pypdf (faster, no OCR)."""
    text = ""
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        logger.error(f"pypdf extraction failed: {e}")
    return text.strip()

def extract_text_easyocr(file_bytes: bytes) -> str:
    """Extract text using EasyOCR (handles images)."""
    if not ocr_reader:
        return ""
    
    try:
        # EasyOCR expects image file path, or bytes, or numpy array.
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("Failed to decode image bytes for OCR.")
            return ""

        # Run OCR
        result = ocr_reader.readtext(img, detail=0)
        
        return "\n".join(result).strip()
        
    except Exception as e:
        logger.error(f"EasyOCR extraction failed: {e}")
        return ""

def extract_text_from_file(file_path: str = None, file_bytes: bytes = None, ext: str = ".pdf") -> str:
    """
    Main entry point. Tries pypdf first for PDFs. 
    If text is empty/sparse, falls back to OCR.
    """
    if file_path:
        ext = os.path.splitext(file_path)[1].lower()
        with open(file_path, "rb") as f:
            file_bytes = f.read()
    
    if not file_bytes:
        return ""

    text = ""
    
    # 1. Try standard extraction for PDFs
    if ext == ".pdf":
        text = extract_text_pypdf(file_bytes)
    
    # 2. Fallback to OCR if applicable (images or scanned pdfs)
    if ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
        text = extract_text_easyocr(file_bytes)
    elif ext == ".txt":
        try:
            text = file_bytes.decode("utf-8")
        except:
            text = file_bytes.decode("latin-1")
    elif len(text) < 50 and ext == ".pdf":
         # Optional: Try OCR concept if pypdf result is empty but user wants content.
         # For now, we return what we found.
         pass
             
    return text

# --- Structured Parsing Logic (Regex/Heuristic) ---

def parse_resume_fields(text: str) -> dict:
    """
    Extracts structured fields using basic patterns.
    """
    if not text:
        return {}

    data = {
        "name": None,
        "email": None,
        "phone": None,
        "summary": None,
        "education": None,
        "experience": None,
        "skills": None
    }

    # 1. Email
    email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    if email_match:
        data["email"] = email_match.group(0)

    # 2. Phone
    # Matches (123) 456-7890, 123-456-7890, etc.
    phone_match = re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    if phone_match:
        data["phone"] = phone_match.group(0)

    # 3. Name (Heuristic)
    # Assume name is one of the first few lines, capitalized, not a label
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for i in range(min(5, len(lines))):
        line = lines[i]
        # Basic check: shorter than 30 chars, mostly letters, doesn't contain '@' or 'phone'
        if 3 < len(line) < 30 and not re.search(r'[@\d]', line):
             # Also exclude common headers if they appear at top
             if line.lower() not in ["resume", "cv", "curriculum vitae", "summary"]:
                data["name"] = line
                break

    # 4. Sections (Heuristic)
    # Define common headers
    headers = {
        "education": ["education", "academic history", "academic background"],
        "experience": ["experience", "work experience", "employment history", "work history"],
        "skills": ["skills", "technical skills", "competencies", "technologies"],
        "summary": ["summary", "profile", "professional summary", "objective"]
    }
    
    lower_text = text.lower()
    
    # Find all header positions
    found_headers = []
    for section_key, aliases in headers.items():
        for alias in aliases:
            # We look for the alias as a distinct line or paragraph start
            # For simplicity, we search for the substring
            idx = lower_text.find(alias)
            if idx != -1:
                found_headers.append((idx, section_key))
                # Break to avoid double counting same section with different alias match
                break
    
    found_headers.sort(key=lambda x: x[0])
    
    # Extract text between headers
    for i, (start_idx, section_key) in enumerate(found_headers):
        # Determine end index
        if i + 1 < len(found_headers):
            end_idx = found_headers[i+1][0]
        else:
            end_idx = len(text)
            
        # Extract substring
        # Add some offset to skip the header itself (approx length of key)
        # Ideally we find the exact end of the header line.
        header_len = len(section_key) # approximation, or search newline
        
        # We take the text slice
        # Using original text ensures case is preserved
        content = text[start_idx:end_idx]
        
        # Clean up the content (remove the header itself from the start)
        # Simple split by newline
        content_lines = content.split('\n')
        if content_lines:
             # Remove lines that look like the header
             cleaned_content = []
             for line in content_lines:
                 if any(h in line.lower() for h in headers[section_key]):
                     continue
                 cleaned_content.append(line)
             data[section_key] = "\n".join(cleaned_content).strip()
        else:
             data[section_key] = content.strip()

    return data
