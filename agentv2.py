"""
KYC Document Processing (Extraction + SIMPLE Verification)

- Extraction: Gemini Vision (your current flow)
- Verification: Rule-based (stable + less rejection)
- Output status: only "verified" | "not_verified"
"""

import os
import json
import base64
import threading
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List

from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


# ============================================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================================
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("❌ GEMINI_API_KEY not found in environment variables!")

print("✅ Environment variables loaded successfully")


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))

    EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "gemini-2.5-flash")

    TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "30"))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))

    # SIMPLE verification knobs (easy mode)
    REQUIRE_AADHAAR_MATCH_PAN_NAME = os.getenv("REQUIRE_NAME_MATCH", "true").lower() == "true"
    REQUIRE_DOB_MATCH = os.getenv("REQUIRE_DOB_MATCH", "true").lower() == "true"
    MIN_MATCH_SCORE = float(os.getenv("MIN_MATCH_SCORE", "0.67"))  # 0..1

    @classmethod
    def validate(cls):
        print(
            f"✅ Config: workers={cls.MAX_WORKERS}, retries={cls.MAX_RETRIES}, "
            f"threshold={cls.CONFIDENCE_THRESHOLD}, min_match={cls.MIN_MATCH_SCORE}"
        )


Config.validate()


# ============================================================================
# HELPERS
# ============================================================================
def encode_image(image_path: str) -> str:
    if not image_path or image_path.strip() == "":
        raise ValueError("Image path is empty")
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_json_from_text(text: str) -> Dict[str, Any]:
    """
    Robust JSON extraction:
    - supports ``````
    - supports JSON embedded in text
    """
    try:
        if not text or not isinstance(text, str):
            return {"error": "Empty response", "extraction_confidence": 0.0}

        t = text.strip()

        if "```" in t:
            cleaned = t.replace("```json", "```", "```")
            parts = cleaned.split("```")
            for p in parts:
                p = p.strip()
                if p.startswith("{") and p.endswith("}"):
                    return json.loads(p)

        start = t.find("{")
        end = t.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(t[start:end])

        return {"error": "No valid JSON found", "extraction_confidence": 0.0}
    except Exception as e:
        return {"error": f"Parse error: {str(e)}", "extraction_confidence": 0.0}


def digits_only(s: str) -> str:
    if not s:
        return ""
    return "".join(ch for ch in str(s) if ch.isdigit())


def normalize_name(name: str) -> str:
    if not name:
        return ""
    # Keep letters + spaces only, collapse spaces, uppercase
    cleaned = re.sub(r"[^A-Za-z ]+", " ", str(name))
    cleaned = re.sub(r"\s+", " ", cleaned).strip().upper()
    return cleaned


def normalize_dob(dob: str) -> str:
    """
    Normalize DOB to DD/MM/YYYY if possible.
    Accepts: DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD (basic), etc.
    """
    if not dob:
        return ""
    s = str(dob).strip()
    s = s.replace("-", "/").replace(".", "/")
    parts = s.split("/")
    if len(parts) == 3:
        # if YYYY/MM/DD
        if len(parts[0]) == 4:
            y, m, d = parts
            return f"{d.zfill(2)}/{m.zfill(2)}/{y}"
        # if DD/MM/YYYY
        d, m, y = parts
        if len(y) == 4:
            return f"{d.zfill(2)}/{m.zfill(2)}/{y}"
    return s


def string_match_score(a: str, b: str) -> float:
    """
    Simple similarity: token overlap ratio (stable + fast).
    """
    a = normalize_name(a)
    b = normalize_name(b)
    if not a or not b:
        return 0.0
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def heuristic_confidence(doc_type: str, data: Dict[str, Any]) -> float:
    """
    If model forgets extraction_confidence, infer it from key fields.
    """
    if not isinstance(data, dict):
        return 0.0

    if doc_type == "aadhaar_front":
        num = digits_only(data.get("aadhaar_number"))
        name = (data.get("name") or "").strip()
        if len(num) == 12 and name:
            return 0.90
        return 0.0

    if doc_type == "aadhaar_back":
        addr = (data.get("address") or "").strip()
        if len(addr) >= 15:
            return 0.85
        if data.get("qr_code_present") is True or data.get("barcode_visible") is True:
            return 0.75
        return 0.0

    if doc_type == "pan_front":
        pan = (data.get("pan_number") or "").strip()
        if len(pan) == 10:
            return 0.85
        return 0.0

    if doc_type == "passport":
        pno = (data.get("passport_number") or "").strip()
        if pno:
            return 0.80
        return 0.0

    return 0.0


# ============================================================================
# MODEL POOL (Thread-safe)
# ============================================================================
class ModelPool:
    _lock = threading.Lock()
    _gemini_models = []
    _pool_size = Config.MAX_WORKERS

    @classmethod
    def get_gemini_vision(cls):
        with cls._lock:
            if not cls._gemini_models:
                for _ in range(cls._pool_size):
                    model = ChatGoogleGenerativeAI(
                        google_api_key=Config.GEMINI_API_KEY,
                        model=Config.EXTRACTION_MODEL,
                        temperature=0.1,
                        max_output_tokens=1500,
                    )
                    cls._gemini_models.append(model)
            return cls._gemini_models[threading.get_ident() % len(cls._gemini_models)]


# ============================================================================
# EXTRACTORS
# ============================================================================
class DocumentExtractor:
    @staticmethod
    def extract_with_retry(doc_type: str, image_path: str, prompt: str, strict_prompt: str) -> Dict[str, Any]:
        print(f"🔍 [{doc_type}] Starting extraction...")

        if not image_path or image_path.strip() == "":
            return {"status": "failed", "error": "Empty image path", "data": {}, "attempts": 0}

        last_error = None

        for attempt in range(Config.MAX_RETRIES):
            try:
                base64_image = encode_image(image_path)
                model = ModelPool.get_gemini_vision()

                use_prompt = strict_prompt if attempt > 0 else prompt
                messages = [
                    SystemMessage(content=use_prompt),
                    HumanMessage(content=[
                        {"type": "text", "text": f"Extract {doc_type}. Return ONLY JSON."},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"},
                    ]),
                ]

                resp = model.invoke(messages)
                raw = getattr(resp, "content", "") or ""
                extracted = parse_json_from_text(raw)

                if extracted.get("error") == "No valid JSON found" and attempt < Config.MAX_RETRIES - 1:
                    last_error = "Model returned non-JSON"
                    print(f"⚠️ [{doc_type}] Non-JSON output; retrying with strict JSON prompt...")
                    continue

                conf = extracted.get("extraction_confidence", None)
                if conf is None:
                    conf = heuristic_confidence(doc_type, extracted)
                    extracted["extraction_confidence"] = conf
                else:
                    try:
                        conf = float(conf)
                    except Exception:
                        conf = heuristic_confidence(doc_type, extracted)
                        extracted["extraction_confidence"] = conf

                status = "success" if conf >= Config.CONFIDENCE_THRESHOLD else "failed"
                print(f"✅ [{doc_type}] status={status} confidence={conf:.2f} attempt={attempt+1}")

                return {"status": status, "data": extracted, "attempts": attempt + 1}

            except Exception as e:
                last_error = str(e)
                print(f"❌ [{doc_type}] Error attempt {attempt + 1}: {last_error}")

        return {"status": "failed", "error": last_error or "Max retries exceeded", "data": {}, "attempts": Config.MAX_RETRIES}


class AadhaarFrontExtractor(DocumentExtractor):
    PROMPT = """Return ONLY JSON:
{
  "is_aadhaar_card": true,
  "aadhaar_number": "12-digit number or null",
  "name": "full name or null",
  "dob": "DD/MM/YYYY or null",
  "mobile_no": "10-digit or null",
  "gender": "Male/Female/Other or null",
  "extraction_confidence": 0.0,
  "quality_issues": []
}
"""
    STRICT = """Return ONLY strict JSON (no markdown, no text):
{"is_aadhaar_card":true,"aadhaar_number":null,"name":null,"dob":null,"mobile_no":null,"gender":null,"extraction_confidence":0.0,"quality_issues":[]}
"""

    @classmethod
    def extract(cls, image_path: str) -> Dict[str, Any]:
        return cls.extract_with_retry("aadhaar_front", image_path, cls.PROMPT, cls.STRICT)


class AadhaarBackExtractor(DocumentExtractor):
    PROMPT = """Return ONLY JSON:
{
  "qr_code_present": true,
  "barcode_visible": true,
  "address": "complete address or null",
  "extraction_confidence": 0.0
}
"""
    STRICT = """Return ONLY strict JSON:
{"qr_code_present":null,"barcode_visible":null,"address":null,"extraction_confidence":0.0}
"""

    @classmethod
    def extract(cls, image_path: str) -> Dict[str, Any]:
        return cls.extract_with_retry("aadhaar_back", image_path, cls.PROMPT, cls.STRICT)


class PANFrontExtractor(DocumentExtractor):
    PROMPT = """Return ONLY JSON:
{
  "is_pan_card": true,
  "name": "full name or null",
  "pan_number": "10-character PAN or null",
  "fathers_name": "father name or null",
  "dob": "DD/MM/YYYY or null",
  "extraction_confidence": 0.0,
  "quality_issues": []
}
"""
    STRICT = """Return ONLY strict JSON:
{"is_pan_card":true,"name":null,"pan_number":null,"fathers_name":null,"dob":null,"extraction_confidence":0.0,"quality_issues":[]}
"""

    @classmethod
    def extract(cls, image_path: str) -> Dict[str, Any]:
        return cls.extract_with_retry("pan_front", image_path, cls.PROMPT, cls.STRICT)


class PassportExtractor(DocumentExtractor):
    PROMPT = """Return ONLY JSON:
{
  "name": "full name or null",
  "nationality": "nationality or null",
  "dob": "DD/MM/YYYY or null",
  "place_of_birth": "place or null",
  "issue_date": "DD/MM/YYYY or null",
  "expiry_date": "DD/MM/YYYY or null",
  "gender": "M/F/X or null",
  "passport_number": "passport number or null",
  "extraction_confidence": 0.0,
  "quality_issues": []
}
"""
    STRICT = """Return ONLY strict JSON:
{"name":null,"nationality":null,"dob":null,"place_of_birth":null,"issue_date":null,"expiry_date":null,"gender":null,"passport_number":null,"extraction_confidence":0.0,"quality_issues":[]}
"""

    @classmethod
    def extract(cls, image_path: str) -> Dict[str, Any]:
        return cls.extract_with_retry("passport", image_path, cls.PROMPT, cls.STRICT)


# ============================================================================
# SIMPLE VERIFIER (2 status only)
# ============================================================================
def verify_simple(extracted: Dict[str, Any]) -> Tuple[str, float, List[str]]:
    """
    Returns:
      status: "verified" | "not_verified"
      score: 0..1
      reasons: list of strings
    """
    reasons = []
    score_parts = []

    aadhaar_f = extracted.get("aadhaar_front", {}) or {}
    aadhaar_b = extracted.get("aadhaar_back", {}) or {}
    pan_f = extracted.get("pan_front", {}) or {}
    passport = extracted.get("passport", {}) or {}

    # 1) Basic format checks
    aadhaar_num = digits_only(aadhaar_f.get("aadhaar_number"))
    if len(aadhaar_num) != 12:
        reasons.append("Aadhaar number format invalid or missing.")
        score_parts.append(0.0)
    else:
        score_parts.append(1.0)

    pan_num = (pan_f.get("pan_number") or "").strip().upper()
    if not re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]{1}$", pan_num):
        reasons.append("PAN format invalid or missing.")
        score_parts.append(0.0)
    else:
        score_parts.append(1.0)

    # 2) Address presence (easy requirement)
    addr = (aadhaar_b.get("address") or "").strip()
    if len(addr) < 10:
        reasons.append("Aadhaar back address missing/too short.")
        score_parts.append(0.0)
    else:
        score_parts.append(1.0)

    # 3) Name match (optional strictness)
    aadhaar_name = aadhaar_f.get("name") or ""
    pan_name = pan_f.get("name") or ""
    name_score = string_match_score(aadhaar_name, pan_name)

    if Config.REQUIRE_AADHAAR_MATCH_PAN_NAME:
        if name_score < Config.MIN_MATCH_SCORE:
            reasons.append(f"Name mismatch between Aadhaar and PAN (score={name_score:.2f}).")
            score_parts.append(0.0)
        else:
            score_parts.append(1.0)
    else:
        # not required, still contributes softly
        score_parts.append(min(1.0, name_score))

    # 4) DOB match (optional strictness)
    aadhaar_dob = normalize_dob(aadhaar_f.get("dob"))
    pan_dob = normalize_dob(pan_f.get("dob"))
    dob_ok = bool(aadhaar_dob and pan_dob and aadhaar_dob == pan_dob)

    if Config.REQUIRE_DOB_MATCH:
        if not dob_ok:
            reasons.append(f"DOB mismatch between Aadhaar and PAN ({aadhaar_dob} vs {pan_dob}).")
            score_parts.append(0.0)
        else:
            score_parts.append(1.0)
    else:
        score_parts.append(1.0 if dob_ok else 0.5)

    # 5) Passport is optional: only check if provided
    if passport:
        pass_name = passport.get("name") or ""
        pass_dob = normalize_dob(passport.get("dob"))
        pass_name_score = string_match_score(aadhaar_name, pass_name)
        if pass_name and pass_name_score < 0.50:
            reasons.append(f"Passport name differs (score={pass_name_score:.2f}).")
            score_parts.append(0.5)
        if pass_dob and aadhaar_dob and pass_dob != aadhaar_dob:
            reasons.append(f"Passport DOB differs ({pass_dob} vs {aadhaar_dob}).")
            score_parts.append(0.5)

    # Final score
    if not score_parts:
        score = 0.0
    else:
        score = sum(score_parts) / len(score_parts)

    status = "verified" if score >= 0.80 and len(reasons) == 0 else "not_verified"
    return status, score, reasons


# ============================================================================
# PARALLEL PROCESSOR
# ============================================================================
def process_documents_parallel(image_paths: Dict[str, str]) -> Dict[str, Any]:
    extractors = {
        "aadhaar_front": AadhaarFrontExtractor,
        "aadhaar_back": AadhaarBackExtractor,
        "pan_front": PANFrontExtractor,
        "passport": PassportExtractor,
    }

    results = {}
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        future_to_doc = {
            executor.submit(extractors[doc_type].extract, path): doc_type
            for doc_type, path in image_paths.items()
            if doc_type in extractors and path and str(path).strip()
        }

        for future in as_completed(future_to_doc):
            doc_type = future_to_doc[future]
            try:
                results[doc_type] = future.result(timeout=Config.TIMEOUT_SECONDS)
            except Exception as e:
                results[doc_type] = {"status": "failed", "error": str(e), "data": {}}

    return results


# ============================================================================
# MAIN FUNCTION (2-status output)
# ============================================================================
def process_kyc_documents(image_paths: Dict[str, str]) -> Dict[str, Any]:
    start_time = datetime.now()

    valid_images = {k: v for k, v in image_paths.items() if v and str(v).strip()}

    required_docs = ["aadhaar_front", "aadhaar_back", "pan_front"]
    missing = [doc for doc in required_docs if doc not in valid_images]
    if missing:
        return {
            "status": "not_verified",
            "verified": False,
            "reasons": [f"Missing required documents: {', '.join(missing)}"],
            "kycData": {},
            "processingTime": 0.0,
        }

    extraction_results = process_documents_parallel(valid_images)

    extracted_data = {}
    extraction_status = {}
    total_attempts = 0

    for doc_type, result in extraction_results.items():
        extracted_data[doc_type] = result.get("data", {}) or {}
        extraction_status[doc_type] = result.get("status", "failed")
        total_attempts += int(result.get("attempts", 0) or 0)

    failed_required = [doc for doc in required_docs if extraction_status.get(doc) != "success"]
    if failed_required:
        return {
            "status": "not_verified",
            "verified": False,
            "reasons": [f"Extraction failed for required documents: {', '.join(failed_required)}"],
            "kycData": {
                "extractionStatus": extraction_status,
                "extractedData": extracted_data,
            },
            "processingTime": (datetime.now() - start_time).total_seconds(),
        }

    # SIMPLE verification
    status, score, reasons = verify_simple(extracted_data)

    processing_time = (datetime.now() - start_time).total_seconds()

    # Map minimal output for your controller
    kyc_data = {
        "aadhaarFrontUrl": valid_images.get("aadhaar_front"),
        "aadhaarBackUrl": valid_images.get("aadhaar_back"),
        "panCardUrl": valid_images.get("pan_front"),
        "passportUrl": valid_images.get("passport", ""),

        "aadhaarNumber": extracted_data.get("aadhaar_front", {}).get("aadhaar_number"),
        "aadhaarName": extracted_data.get("aadhaar_front", {}).get("name"),
        "aadhaarDOB": extracted_data.get("aadhaar_front", {}).get("dob"),
        "aadhaarGender": extracted_data.get("aadhaar_front", {}).get("gender"),
        "aadhaarAddress": extracted_data.get("aadhaar_back", {}).get("address"),

        "panNumber": extracted_data.get("pan_front", {}).get("pan_number"),
        "panName": extracted_data.get("pan_front", {}).get("name"),
        "panDOB": extracted_data.get("pan_front", {}).get("dob"),
        "panFatherName": extracted_data.get("pan_front", {}).get("fathers_name"),

        "passportNumber": extracted_data.get("passport", {}).get("passport_number"),
        "passportName": extracted_data.get("passport", {}).get("name"),
        "passportDOB": extracted_data.get("passport", {}).get("dob"),

        "extractedData": extracted_data,
        "extractionStatus": extraction_status,

        "verification": {
            "score": score,
            "status": status,
            "reasons": reasons,
        },

        "extractionMetadata": {
            "processingTime": processing_time,
            "totalAttempts": total_attempts,
            "modelUsed": Config.EXTRACTION_MODEL,
            "confidenceThreshold": Config.CONFIDENCE_THRESHOLD,
        },

        "lastCheckedAt": datetime.now().isoformat(),
    }

    return {
        "status": status,                    # "verified" | "not_verified"
        "verified": status == "verified",    # boolean
        "reasons": reasons,                  # list (empty if verified)
        "kycData": kyc_data,
        "processingTime": processing_time,
    }
