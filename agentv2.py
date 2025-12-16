"""
Optimized KYC Document Processing System with Parallel Execution
Multi-agent system with concurrent document processing for maximum speed
"""

import os
import json
from typing import TypedDict, Dict, Any, Literal
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import base64
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import threading

# ============================================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================================

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("❌ GROQ_API_KEY not found in environment variables!")
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("❌ GEMINI_API_KEY not found in environment variables!")

print("✅ Environment variables loaded successfully")

# ============================================================================
# CONFIGURATION
# ============================================================================


class Config:
    """Configuration for AI models and parallel processing"""
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))  # Reduced for speed
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))  # Parallel workers
    EXTRACTION_MODEL = os.getenv(
        "EXTRACTION_MODEL", "gemini-2.5-flash")  # Corrected model name
    VERIFICATION_MODEL = os.getenv(
        "VERIFICATION_MODEL", "groq/compound")  # Corrected Groq model
    TIMEOUT_SECONDS = 30

    @classmethod
    def validate(cls):
        if not cls.GROQ_API_KEY or not cls.GEMINI_API_KEY:
            raise ValueError("API keys missing!")
        print(
            f"✅ Config: {cls.MAX_WORKERS} parallel workers, {cls.MAX_RETRIES} max retries")


Config.validate()

# ============================================================================
# DOCUMENT TYPE ENUM
# ============================================================================


class DocumentType(str, Enum):
    AADHAAR_FRONT = "aadhaar_front"
    AADHAAR_BACK = "aadhaar_back"
    PAN_FRONT = "pan_front"
    PAN_BACK = "pan_back"
    PASSPORT = "passport"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def encode_image(image_path: str) -> str:
    """Encode image to base64 - optimized with caching"""
    if not image_path or image_path.strip() == "":
        raise ValueError("Image path is empty")

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def parse_extraction_response(response: str) -> Dict[str, Any]:
    """Parse JSON response from extraction agents"""
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end > start:
            json_str = response[start:end]
            return json.loads(json_str)
        return {"error": "No valid JSON found", "extraction_confidence": 0.0}
    except Exception as e:
        return {"error": f"Parse error: {str(e)}", "extraction_confidence": 0.0}

# ============================================================================
# AI MODEL POOL (Thread-safe)
# ============================================================================


class ModelPool:
    """Thread-safe model pool for parallel processing"""
    _lock = threading.Lock()
    _gemini_models = []
    _groq_models = []
    _pool_size = Config.MAX_WORKERS

    @classmethod
    def get_gemini_vision(cls):
        """Get or create Gemini vision model"""
        with cls._lock:
            if not cls._gemini_models:
                for _ in range(cls._pool_size):
                    model = ChatGoogleGenerativeAI(
                        google_api_key=Config.GEMINI_API_KEY,
                        model=Config.EXTRACTION_MODEL,
                        temperature=0.1,
                        max_output_tokens=1500
                    )
                    cls._gemini_models.append(model)
            return cls._gemini_models[threading.get_ident() % len(cls._gemini_models)]

    @classmethod
    def get_groq_verifier(cls):
        """Get or create Groq model"""
        with cls._lock:
            if not cls._groq_models:
                for _ in range(min(2, cls._pool_size)):  # Fewer verifier instances
                    model = ChatGroq(
                        groq_api_key=Config.GROQ_API_KEY,
                        model_name=Config.VERIFICATION_MODEL,
                        temperature=0.2,
                        max_tokens=2500
                    )
                    cls._groq_models.append(model)
            return cls._groq_models[0]  # Single verifier is sufficient

# ============================================================================
# PARALLEL DOCUMENT EXTRACTORS
# ============================================================================


class DocumentExtractor:
    """Base class for all document extractors with retry logic"""

    @staticmethod
    def extract_with_retry(doc_type: str, image_path: str, prompt: str) -> Dict[str, Any]:
        """Extract data with automatic retry logic"""
        print(f"🔍 [{doc_type}] Starting extraction...")

        if not image_path or image_path.strip() == "":
            return {
                "status": "failed",
                "error": "Empty image path",
                "data": {}
            }

        for attempt in range(Config.MAX_RETRIES):
            try:
                base64_image = encode_image(image_path)
                model = ModelPool.get_gemini_vision()

                messages = [
                    SystemMessage(content=prompt),
                    HumanMessage(content=[
                        {"type": "text", "text": f"Extract {doc_type} information"},
                        {"type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"}
                    ])
                ]

                response = model.invoke(messages)
                extracted = parse_extraction_response(response.content)

                confidence = extracted.get("extraction_confidence", 0)

                if confidence >= 0.7 or attempt == Config.MAX_RETRIES - 1:
                    print(
                        f"✅ [{doc_type}] Extracted (confidence: {confidence:.2f}, attempt: {attempt + 1})")
                    return {
                        "status": "success",
                        "data": extracted,
                        "attempts": attempt + 1
                    }

                print(
                    f"⚠️  [{doc_type}] Low confidence ({confidence:.2f}), retry {attempt + 1}/{Config.MAX_RETRIES}")

            except Exception as e:
                print(f"❌ [{doc_type}] Error on attempt {attempt + 1}: {str(e)}")
                if attempt == Config.MAX_RETRIES - 1:
                    return {
                        "status": "failed",
                        "error": str(e),
                        "data": {},
                        "attempts": attempt + 1
                    }

        return {"status": "failed", "error": "Max retries exceeded", "data": {}}


class AadhaarFrontExtractor(DocumentExtractor):
    PROMPT = """Extract from Aadhaar Card FRONT. Return ONLY JSON:
{
    "is_aadhaar_card": true,
    "aadhaar_number": "12-digit number",
    "name": "full name",
    "dob": "DD/MM/YYYY",
    "mobile_no": "10-digit or null",
    "gender": "Male/Female",
    "extraction_confidence": 0.95,
    "quality_issues": []
}"""

    @classmethod
    def extract(cls, image_path: str) -> Dict[str, Any]:
        return cls.extract_with_retry("aadhaar_front", image_path, cls.PROMPT)


class AadhaarBackExtractor(DocumentExtractor):
    PROMPT = """Extract from Aadhaar BACK. Return ONLY JSON:
{
    "qr_code_present": true,
    "barcode_visible": true,
    "address": "complete address",
    "extraction_confidence": 0.95
}"""

    @classmethod
    def extract(cls, image_path: str) -> Dict[str, Any]:
        return cls.extract_with_retry("aadhaar_back", image_path, cls.PROMPT)


class PANFrontExtractor(DocumentExtractor):
    PROMPT = """Extract from PAN Card. Return ONLY JSON:
{
    "is_pan_card": true,
    "name": "full name",
    "pan_number": "10-character PAN number",
    "fathers_name": "father's name",
    "dob": "DD/MM/YYYY",
    "extraction_confidence": 0.95,
    "quality_issues": []
}"""

    @classmethod
    def extract(cls, image_path: str) -> Dict[str, Any]:
        return cls.extract_with_retry("pan_front", image_path, cls.PROMPT)


class PANBackExtractor(DocumentExtractor):
    @classmethod
    def extract(cls, image_path: str) -> Dict[str, Any]:
        if not image_path or image_path.strip() == "":
            return {
                "status": "success",
                "data": {"note": "PAN back not provided (optional)"},
                "attempts": 0
            }

        prompt = """Extract from PAN BACK. Return ONLY JSON:
{
    "signature_present": true,
    "additional_info": "text visible",
    "extraction_confidence": 0.85
}"""
        return cls.extract_with_retry("pan_back", image_path, prompt)


class PassportExtractor(DocumentExtractor):
    PROMPT = """Extract from Passport. Return ONLY JSON:
{
    "name": "full name",
    "nationality": "nationality",
    "dob": "DD/MM/YYYY",
    "place_of_birth": "place",
    "issue_date": "DD/MM/YYYY",
    "expiry_date": "DD/MM/YYYY",
    "gender": "M/F",
    "passport_number": "passport number",
    "extraction_confidence": 0.95,
    "quality_issues": []
}"""

    @classmethod
    def extract(cls, image_path: str) -> Dict[str, Any]:
        return cls.extract_with_retry("passport", image_path, cls.PROMPT)

# ============================================================================
# VERIFICATION AGENT
# ============================================================================


class VerificationAgent:
    """Fast verification with cross-checking"""

    @staticmethod
    def verify(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        print("\n🔍 VERIFICATION: Cross-checking extracted data...")

        prompt = f"""Cross-validate this KYC data. Return ONLY JSON:

Data:
{json.dumps(extracted_data, indent=2)}

Validate:
1. Name consistency across documents
2. DOB consistency
3. Gender consistency
4. Document format (Aadhaar: 12 digits, PAN: 10 chars, Passport valid)
5. Quality and completeness

Return:
{{
    "verification_status": "approved/rejected/needs_review",
    "confidence_score": 0.95,
    "inconsistencies": [{{"field": "name", "issue": "description", "severity": "high/medium/low"}}],
    "validated_data": {{
        "name": "verified name",
        "dob": "verified DOB",
        "gender": "verified gender",
    }},
    "recommendations": ["actions needed"],
    "quality_score": 0.95
}}"""

        try:
            model = ModelPool.get_groq_verifier()
            messages = [
                SystemMessage(content="You are a KYC verification expert."),
                HumanMessage(content=prompt)
            ]

            response = model.invoke(messages)
            result = parse_extraction_response(response.content)

            print(
                f"✅ VERIFICATION COMPLETE: {result.get('verification_status', 'unknown')}")
            return result

        except Exception as e:
            print(f"❌ VERIFICATION FAILED: {str(e)}")
            return {
                "verification_status": "needs_review",
                "confidence_score": 0.0,
                "error": str(e)
            }

# ============================================================================
# PARALLEL PROCESSOR
# ============================================================================


def process_documents_parallel(image_paths: Dict[str, str]) -> Dict[str, Any]:
    """Process all documents in parallel"""

    extractors = {
        "aadhaar_front": AadhaarFrontExtractor,
        "aadhaar_back": AadhaarBackExtractor,
        "pan_front": PANFrontExtractor,
        "pan_back": PANBackExtractor,
        "passport": PassportExtractor
    }

    results = {}

    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        future_to_doc = {
            executor.submit(extractors[doc_type].extract, path): doc_type
            for doc_type, path in image_paths.items()
            if doc_type in extractors
        }

        for future in as_completed(future_to_doc):
            doc_type = future_to_doc[future]
            try:
                result = future.result(timeout=Config.TIMEOUT_SECONDS)
                results[doc_type] = result
            except Exception as e:
                print(f"❌ [{doc_type}] Failed: {str(e)}")
                results[doc_type] = {"status": "failed",
                                     "error": str(e), "data": {}}

    return results

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================


def process_kyc_documents(image_paths: Dict[str, str]) -> Dict[str, Any]:
    """Main optimized KYC processing function"""

    start_time = datetime.now()
    print("=" * 80)
    print("🚀 OPTIMIZED KYC PROCESSING - PARALLEL EXECUTION")
    print("=" * 80)

    # Filter valid paths
    valid_images = {k: v for k, v in image_paths.items() if v and v.strip()}

    # Check required documents
    required_docs = ["aadhaar_front", "aadhaar_back", "pan_front"]
    missing = [doc for doc in required_docs if doc not in valid_images]

    if missing:
        error_msg = f"Missing required documents: {', '.join(missing)}"
        print(f"\n❌ {error_msg}")
        return {
            "kycStatus": "rejected",
            "kycData": {},
            "error": error_msg,
            "processingTime": 0
        }

    print(f"\n📋 Processing {len(valid_images)} documents in parallel...")
    print(
        f"🔧 Workers: {Config.MAX_WORKERS}, Max retries: {Config.MAX_RETRIES}\n")

    try:
        # STEP 1: Parallel extraction
        extraction_results = process_documents_parallel(valid_images)

        # STEP 2: Compile extracted data
        extracted_data = {}
        extraction_status = {}
        total_attempts = 0

        for doc_type, result in extraction_results.items():
            extracted_data[doc_type] = result.get("data", {})
            extraction_status[doc_type] = result.get("status", "failed")
            total_attempts += result.get("attempts", 0)

        # Check if required documents failed extraction
        failed_required = [
            doc for doc in required_docs if extraction_status.get(doc) != "success"]
        if failed_required:
            error_msg = f"Extraction failed for required documents: {', '.join(failed_required)}"
            print(f"\n❌ {error_msg}")
            return {
                "kycStatus": "rejected",
                "kycData": {
                    "extractionStatus": extraction_status,
                    "extractedData": extracted_data
                },
                "error": error_msg,
                "processingTime": (datetime.now() - start_time).total_seconds()
            }

        # STEP 3: Verification
        verification_result = VerificationAgent.verify(extracted_data)

        # STEP 4: Format output to match Student schema
        processing_time = (datetime.now() - start_time).total_seconds()

        # Map to Student model kycData structure
        kyc_data = {
            # Document URLs (kept as is from input)
            "aadhaarFrontUrl": valid_images.get("aadhaar_front"),
            "aadhaarBackUrl": valid_images.get("aadhaar_back"),
            "panCardUrl": valid_images.get("pan_front"),
            "panCardBackUrl": valid_images.get("pan_back", ""),
            "passportUrl": valid_images.get("passport", ""),

            # Extracted Aadhaar data
            "aadhaarNumber": extracted_data.get("aadhaar_front", {}).get("aadhaar_number"),
            "aadhaarName": extracted_data.get("aadhaar_front", {}).get("name"),
            "aadhaarDOB": extracted_data.get("aadhaar_front", {}).get("dob"),
            "aadhaarAddress": extracted_data.get("aadhaar_back", {}).get("address"),
            "aadhaarGender": extracted_data.get("aadhaar_front", {}).get("gender"),

            # Extracted PAN data
            "panNumber": extracted_data.get("pan_front", {}).get("pan_number"),
            "panName": extracted_data.get("pan_front", {}).get("name"),
            "panDOB": extracted_data.get("pan_front", {}).get("dob"),
            "panFatherName": extracted_data.get("pan_front", {}).get("fathers_name"),

            # Extracted Passport data
            "passportNumber": extracted_data.get("passport", {}).get("passport_number"),
            "passportName": extracted_data.get("passport", {}).get("name"),
            "passportDOB": extracted_data.get("passport", {}).get("dob"),
            "passportIssueDate": extracted_data.get("passport", {}).get("issue_date"),
            "passportExpiryDate": extracted_data.get("passport", {}).get("expiry_date"),
            "passportPlaceOfIssue": extracted_data.get("passport", {}).get("place_of_birth"),
            "passportPlaceOfBirth": extracted_data.get("passport", {}).get("place_of_birth"),

            # Verification metadata
            "verificationSource": "aiextractiongroq",
            "lastVerifiedAt": datetime.now().isoformat(),
            "failureCount": 0 if verification_result.get("verification_status") == "approved" else 1,
            "failedAt": None if verification_result.get("verification_status") == "approved" else datetime.now().isoformat(),

            # Extracted data (full raw data)
            "extractedData": extracted_data,

            # Confidence and verification
            "verificationConfidence": int(verification_result.get("confidence_score", 0) * 100),
            "verificationLevel": verification_result.get("verification_status"),
            "verificationReason": ", ".join(verification_result.get("recommendations", [])),
            "validationScore": verification_result.get("quality_score", 0),

            # Metadata
            "extractionMetadata": {
                "processingTime": processing_time,
                "totalAttempts": total_attempts,
                "extractionStatus": extraction_status,
                "modelUsed": Config.EXTRACTION_MODEL,
                "verificationModel": Config.VERIFICATION_MODEL
            },

            # Document completeness
            "documentCompleteness": {
                "aadhaarFront": extraction_status.get("aadhaar_front") == "success",
                "aadhaarBack": extraction_status.get("aadhaar_back") == "success",
                "panFront": extraction_status.get("pan_front") == "success",
                "panBack": extraction_status.get("pan_back") == "success",
                "passport": extraction_status.get("passport") == "success"
            },

            # Identity confirmation
            "identityConfirmation": verification_result.get("validated_data", {}),

            # Compliance checks
            "complianceChecks": {
                "nameMatch": len([i for i in verification_result.get("inconsistencies", []) if i.get("field") == "name"]) == 0,
                "dobMatch": len([i for i in verification_result.get("inconsistencies", []) if i.get("field") == "dob"]) == 0,
                "genderMatch": len([i for i in verification_result.get("inconsistencies", []) if i.get("field") == "gender"]) == 0,
                "documentFormatValid": verification_result.get("quality_score", 0) > 0.7
            },

            # Risk assessment
            "riskAssessment": {
                "overallRisk": "low" if verification_result.get("confidence_score", 0) > 0.85 else "medium",
                "inconsistencyCount": len(verification_result.get("inconsistencies", [])),
                "qualityScore": verification_result.get("quality_score", 0)
            },

            # Issues
            "validationIssues": [i.get("issue") for i in verification_result.get("inconsistencies", []) if i.get("severity") in ["high", "medium"]],
            "verificationIssues": verification_result.get("recommendations", [])
        }

        # Determine KYC status
        status_map = {
            "approved": "verified",
            "rejected": "rejected",
            "needs_review": "manual_review"
        }
        kyc_status = status_map.get(verification_result.get(
            "verification_status"), "manual_review")

        result = {
            "kycStatus": kyc_status,
            "kycData": kyc_data,
            "kycVerifiedAt": datetime.now().isoformat() if kyc_status == "verified" else None,
            "kycRejectedAt": datetime.now().isoformat() if kyc_status == "rejected" else None,
            "processingTime": processing_time,
            "summary": {
                "totalDocuments": len(valid_images),
                "successfulExtractions": sum(1 for s in extraction_status.values() if s == "success"),
                "failedExtractions": sum(1 for s in extraction_status.values() if s == "failed"),
                "totalAttempts": total_attempts,
                "verificationStatus": verification_result.get("verification_status"),
                "confidenceScore": verification_result.get("confidence_score")
            }
        }

        print("\n" + "=" * 80)
        print("✅ KYC PROCESSING COMPLETE")
        print("=" * 80)
        print(f"Status: {kyc_status}")
        print(f"Time: {processing_time:.2f}s")
        print(
            f"Success: {result['summary']['successfulExtractions']}/{result['summary']['totalDocuments']}")
        print(f"Confidence: {result['summary']['confidenceScore']:.2f}")

        return result

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            "kycStatus": "rejected",
            "kycData": {},
            "error": str(e),
            "processingTime": processing_time
        }

# ============================================================================
# USAGE EXAMPLE
# ============================================================================


if __name__ == "__main__":
    image_paths = {
        "aadhaar_front": r"C:\Users\sansk\OneDrive\Desktop\loan\loanbackend-2\backendv2\kycdata\bull.jpg",
        "aadhaar_back": r"C:\Users\sansk\OneDrive\Desktop\loan\loanbackend-2\backendv2\kycdata\Aadharfrontform1_page-0002.jpg",
        "pan_front": r"C:\Users\sansk\OneDrive\Desktop\loan\loanbackend-2\backendv2\kycdata\kritipan.jpg",
        "pan_back": "",  # Optional
        "passport": r"C:\Users\sansk\OneDrive\Desktop\loan\loanbackend-2\backendv2\kycdata\kritipass.jpg"
    }

    result = process_kyc_documents(image_paths)

    # Save result
    with open("kyc_result_optimized.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n💾 Results saved to kyc_result_optimized.json")
    print(f"\n📊 KYC Status: {result['kycStatus']}")
    print(f"⏱️  Processing Time: {result['processingTime']:.2f}s")
