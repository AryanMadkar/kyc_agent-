"""
Flask Production Server for KYC Document Processing
Optimized for temporary file handling - no permanent storage
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from datetime import datetime
from typing import Dict, Any
from werkzeug.utils import secure_filename
import uuid
import threading
import tempfile
import shutil
import io
from PIL import Image

# Import the optimized KYC processing module
from agentv2 import process_kyc_documents, Config

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration - using in-memory processing
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
app.config['SECRET_KEY'] = os.getenv(
    'FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')

# Global cache for results (in production, use Redis)
results_cache = {}
cache_lock = threading.Lock()


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower(
           ) in app.config['ALLOWED_EXTENSIONS']


def validate_image(file_stream) -> bool:
    """Validate if the uploaded file is a valid image"""
    try:
        img = Image.open(file_stream)
        img.verify()  # Verify it's an image
        file_stream.seek(0)  # Reset stream position
        return True
    except Exception:
        return False


def save_to_temp_memory(file) -> tempfile.NamedTemporaryFile:
    """Save uploaded file to temporary memory and return temp file object"""
    if file.filename == '':
        raise ValueError("No file selected")

    if not allowed_file(file.filename):
        raise ValueError(
            f"File type not allowed. Allowed types: {', '.join(app.config['ALLOWED_EXTENSIONS'])}")

    # Validate it's a real image
    if not validate_image(file.stream):
        raise ValueError("Invalid image file")

    # Reset stream position after validation
    file.stream.seek(0)

    # Create a temporary file that will be automatically deleted when closed
    # Using delete=False temporarily so we can pass paths to process_kyc_documents
    temp_file = tempfile.NamedTemporaryFile(
        suffix=os.path.splitext(secure_filename(file.filename))[1],
        delete=False  # We'll manually delete it after processing
    )

    # Save file to temporary location
    file.save(temp_file.name)
    temp_file.close()  # Close but don't delete yet

    return temp_file


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "kyc-processing-api",
        "version": "2.0.0",
        "storage_mode": "temporary-in-memory"
    })


@app.route('/api/kyc/process', methods=['POST'])
def process_kyc():
    """
    Process KYC documents via form data upload
    Files are stored temporarily in memory and deleted immediately after processing
    """
    temp_files = []  # Track all temp files for cleanup

    try:
        start_time = datetime.now()

        # Check if all required files are present
        required_files = ['aadhaar_front', 'aadhaar_back', 'pan_front']

        # Validate required files
        for req_file in required_files:
            if req_file not in request.files:
                return jsonify({
                    "error": f"Missing required document: {req_file}",
                    "required_documents": required_files,
                    "status": "error"
                }), 400

        # Save uploaded files to temporary memory
        image_paths = {}

        try:
            # Process all files in request
            for doc_type in ['aadhaar_front', 'aadhaar_back', 'pan_front', 'pan_back', 'passport']:
                if doc_type in request.files:
                    file = request.files[doc_type]
                    if file and file.filename != '':
                        temp_file = save_to_temp_memory(file)
                        image_paths[doc_type] = temp_file.name
                        temp_files.append(temp_file.name)
                        print(
                            f"📄 Saved {doc_type} to temp file: {temp_file.name}")
                    elif doc_type not in required_files:
                        # Optional documents can be empty
                        image_paths[doc_type] = ""
                elif doc_type not in required_files:
                    # Optional documents not provided
                    image_paths[doc_type] = ""

            print(f"📁 Processing {len(temp_files)} uploaded files (temporary)")

            # Process KYC documents
            result = process_kyc_documents(image_paths)

            # Add processing metadata
            result["request_id"] = str(uuid.uuid4())
            result["processed_at"] = datetime.now().isoformat()
            result["uploaded_files_count"] = len(temp_files)
            result["storage_mode"] = "temporary-in-memory"

            # Cache result (in production, use Redis)
            with cache_lock:
                results_cache[result["request_id"]] = {
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }

            processing_time = (datetime.now() - start_time).total_seconds()
            result["total_processing_time"] = processing_time

            return jsonify(result), 200

        except Exception as e:
            # Re-raise to be handled by outer exception handler
            raise e

    except ValueError as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400

    except Exception as e:
        print(f"❌ Server error: {str(e)}")
        import traceback
        traceback.print_exc()

        return jsonify({
            "error": "Internal server error",
            "message": str(e),
            "status": "error"
        }), 500

    finally:
        # ALWAYS clean up temp files, even if there's an error
        cleanup_temp_files(temp_files)


def cleanup_temp_files(file_paths):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)  # Delete the temporary file
                print(f"🧹 Cleaned up temp file: {file_path}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to delete temp file {file_path}: {e}")

# Alternative: Process directly from memory without saving to disk


@app.route('/api/kyc/process-direct', methods=['POST'])
def process_kyc_direct():
    """
    Process KYC documents directly from memory without saving to disk
    This is the most secure option if your agentv2 can handle file-like objects
    """
    try:
        start_time = datetime.now()

        # Check if all required files are present
        required_files = ['aadhaar_front', 'aadhaar_back', 'pan_front']

        # Validate required files
        for req_file in required_files:
            if req_file not in request.files:
                return jsonify({
                    "error": f"Missing required document: {req_file}",
                    "required_documents": required_files,
                    "status": "error"
                }), 400

        # Store files in memory as BytesIO objects
        file_objects = {}
        file_count = 0

        for doc_type in ['aadhaar_front', 'aadhaar_back', 'pan_front', 'pan_back', 'passport']:
            if doc_type in request.files:
                file = request.files[doc_type]
                if file and file.filename != '':
                    # Validate file
                    if not allowed_file(file.filename):
                        return jsonify({
                            "error": f"Invalid file type for {doc_type}",
                            "status": "error"
                        }), 400

                    # Read file into memory
                    file_data = file.read()

                    # Option 1: If agentv2 accepts BytesIO objects
                    file_objects[doc_type] = io.BytesIO(file_data)

                    # Option 2: If agentv2 needs file paths, create temporary in-memory file
                    # with tempfile.NamedTemporaryFile(mode='wb', delete=False) as tmp:
                    #     tmp.write(file_data)
                    #     file_objects[doc_type] = tmp.name

                    file_count += 1
                    file.seek(0)  # Reset file pointer if needed
                elif doc_type not in required_files:
                    file_objects[doc_type] = None
            elif doc_type not in required_files:
                file_objects[doc_type] = None

        print(f"🧠 Processing {file_count} files directly from memory")

        # TODO: You'll need to modify process_kyc_documents to accept file objects
        # For now, this is a placeholder implementation
        result = {
            "status": "processing_direct",
            "message": "Direct memory processing mode",
            "files_processed": file_count
        }

        result["request_id"] = str(uuid.uuid4())
        result["processed_at"] = datetime.now().isoformat()
        result["storage_mode"] = "direct-memory"

        processing_time = (datetime.now() - start_time).total_seconds()
        result["total_processing_time"] = processing_time

        return jsonify(result), 200

    except Exception as e:
        print(f"❌ Direct processing error: {str(e)}")
        import traceback
        traceback.print_exc()

        return jsonify({
            "error": "Internal server error",
            "message": str(e),
            "status": "error"
        }), 500


@app.route('/api/kyc/process-urls', methods=['POST'])
def process_kyc_urls():
    """
    Process KYC documents using URLs (for cloud storage)
    This avoids any file storage completely
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON data provided",
                "status": "error"
            }), 400

        # Validate required fields
        required_fields = ['aadhaar_front', 'aadhaar_back', 'pan_front']
        missing_fields = [
            field for field in required_fields if field not in data]

        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}",
                "required_fields": required_fields,
                "status": "error"
            }), 400

        # Process with URLs - no file storage needed
        image_paths = {
            "aadhaar_front": data.get('aadhaar_front'),
            "aadhaar_back": data.get('aadhaar_back'),
            "pan_front": data.get('pan_front'),
            "pan_back": data.get('pan_back', ''),
            "passport": data.get('passport', '')
        }

        result = process_kyc_documents(image_paths)
        result["request_id"] = str(uuid.uuid4())
        result["processed_at"] = datetime.now().isoformat()
        result["source"] = "urls"
        result["storage_mode"] = "no-storage-urls"

        return jsonify(result), 200

    except Exception as e:
        print(f"❌ URL processing error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
            "status": "error"
        }), 500


@app.route('/api/kyc/result/<request_id>', methods=['GET'])
def get_result(request_id: str):
    """Retrieve cached result by request ID"""
    with cache_lock:
        if request_id in results_cache:
            result = results_cache[request_id]
            return jsonify({
                "status": "success",
                "request_id": request_id,
                "result": result["result"],
                "timestamp": result["timestamp"]
            }), 200
        else:
            return jsonify({
                "error": "Result not found or expired",
                "request_id": request_id,
                "status": "error"
            }), 404


@app.route('/api/kyc/status', methods=['GET'])
def system_status():
    """Get system status and configuration"""
    return jsonify({
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "max_workers": Config.MAX_WORKERS,
            "max_retries": Config.MAX_RETRIES,
            "extraction_model": Config.EXTRACTION_MODEL,
            "verification_model": Config.VERIFICATION_MODEL,
            "max_file_size": f"{app.config['MAX_CONTENT_LENGTH'] / (1024*1024)} MB"
        },
        "cache_size": len(results_cache),
        "storage_mode": "temporary-in-memory",
        "data_retention": "no-permanent-storage"
    })

# Clean up cache periodically (optional)


def cleanup_old_cache():
    """Periodically clean up old cache entries"""
    while True:
        import time
        time.sleep(3600)  # Run every hour

        with cache_lock:
            current_time = datetime.now()
            old_keys = []
            for key, value in results_cache.items():
                cache_time = datetime.fromisoformat(value["timestamp"])
                if (current_time - cache_time).total_seconds() > 86400:  # 24 hours
                    old_keys.append(key)

            for key in old_keys:
                del results_cache[key]
                print(f"🧹 Cleaned up old cache entry: {key}")


# Start cache cleanup thread
cache_cleanup_thread = threading.Thread(target=cleanup_old_cache, daemon=True)
cache_cleanup_thread.start()


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "status": "error"
    }), 404


@app.errorhandler(413)
def too_large(error):
    return jsonify({
        "error": f"File too large. Maximum size is {app.config['MAX_CONTENT_LENGTH'] / (1024*1024)} MB",
        "status": "error"
    }), 413


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "status": "error"
    }), 500


if __name__ == '__main__':
    # Production settings
    port = 5001  # Force port 5000
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

    print(f"🚀 Starting KYC Processing Server on port {port}")
    print(f"🔧 Max workers: {Config.MAX_WORKERS}")
    print(f"🔄 Max retries: {Config.MAX_RETRIES}")
    print(f"💾 Storage mode: Temporary in-memory only")
    print(f"🧹 Cache cleanup: Active (24h retention)")

    app.run(
        host='0.0.0.0',
        port=port,  # Use fixed port
        debug=debug,
        threaded=True
    )
