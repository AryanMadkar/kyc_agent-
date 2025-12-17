"""
Flask Production Server for KYC Document Processing
With Redis Queue Support and OpenRouter Fallback
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
from redis import Redis
from rq import Queue
from rq.job import Job
from rq.registry import StartedJobRegistry, FinishedJobRegistry, FailedJobRegistry

# Import the optimized KYC processing module
from agentv2 import process_kyc_documents, Config

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['REDIS_URL'] = os.getenv('REDIS_URL')

# Redis connection and queue setup
try:
    redis_conn = Redis.from_url(app.config['REDIS_URL'],ssl_cert_reqs=None)
    redis_conn.ping()
    kyc_queue = Queue('kyc_processing', connection=redis_conn)
    print("✅ Redis connected successfully")
except Exception as e:
    print(f"⚠️ Redis connection failed: {e}")
    print("⚠️ Queue features will be disabled")
    redis_conn = None
    kyc_queue = None

# Global cache for results (in production, use Redis)
results_cache = {}
cache_lock = threading.Lock()

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(
        suffix=os.path.splitext(secure_filename(file.filename))[1],
        delete=False  # We'll manually delete it after processing
    )

    # Save file to temporary location
    file.save(temp_file.name)
    temp_file.close()  # Close but don't delete yet

    return temp_file

def cleanup_temp_files(file_paths):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)  # Delete the temporary file
                print(f"🧹 Cleaned up temp file: {file_path}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to delete temp file {file_path}: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    redis_status = "disconnected"
    queue_length = 0

    if redis_conn:
        try:
            redis_conn.ping()
            redis_status = "connected"
            if kyc_queue:
                queue_length = len(kyc_queue)
        except:
            redis_status = "error"

    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "kyc-processing-api",
        "version": "3.0.0-fallback",
        "storage_mode": "temporary-in-memory",
        "queue_status": redis_status,
        "jobs_in_queue": queue_length,
        "fallback_enabled": True
    })

@app.route('/api/kyc/process', methods=['POST'])
def process_kyc():
    """
    Process KYC documents synchronously (immediate processing)
    Files are stored temporarily and deleted immediately after processing
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
                        print(f"📄 Saved {doc_type} to temp file: {temp_file.name}")
                elif doc_type not in required_files:
                    # Optional documents can be empty
                    image_paths[doc_type] = ""

            print(f"📁 Processing {len(temp_files)} uploaded files (synchronous)")

            # Process KYC documents
            result = process_kyc_documents(image_paths)

            # Add processing metadata
            result["request_id"] = str(uuid.uuid4())
            result["processed_at"] = datetime.now().isoformat()
            result["uploaded_files_count"] = len(temp_files)
            result["storage_mode"] = "temporary-in-memory"
            result["processing_mode"] = "synchronous"

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

@app.route('/api/kyc/process-async', methods=['POST'])
def process_kyc_async():
    """
    Process KYC documents asynchronously using Redis Queue
    Use this for high-volume workloads
    """
    if not redis_conn or not kyc_queue:
        return jsonify({
            "error": "Queue service unavailable",
            "message": "Redis is not connected. Use /api/kyc/process for synchronous processing.",
            "status": "error"
        }), 503

    temp_files = []

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
        for doc_type in ['aadhaar_front', 'aadhaar_back', 'pan_front', 'pan_back', 'passport']:
            if doc_type in request.files:
                file = request.files[doc_type]
                if file and file.filename != '':
                    temp_file = save_to_temp_memory(file)
                    image_paths[doc_type] = temp_file.name
                    temp_files.append(temp_file.name)
                    print(f"📄 Saved {doc_type} to temp file: {temp_file.name}")
            elif doc_type not in required_files:
                # Optional documents not provided
                image_paths[doc_type] = ""

        print(f"📁 Queuing job with {len(temp_files)} uploaded files")

        # Enqueue job
        job = kyc_queue.enqueue(
            'agentv2.process_kyc_documents',
            image_paths,
            job_timeout='5m',
            result_ttl=3600  # Keep result for 1 hour
        )

        print(f"📋 Job {job.id} queued successfully")

        return jsonify({
            "status": "queued",
            "job_id": job.id,
            "position_in_queue": len(kyc_queue),
            "check_status_url": f"/api/kyc/status/{job.id}",
            "message": "Job queued for processing. Use the status URL to check progress.",
            "processing_mode": "asynchronous"
        }), 202

    except Exception as e:
        cleanup_temp_files(temp_files)
        print(f"❌ Queue error: {str(e)}")
        import traceback
        traceback.print_exc()

        return jsonify({
            "error": "Failed to queue job",
            "message": str(e),
            "status": "error"
        }), 500

@app.route('/api/kyc/status/<job_id>', methods=['GET'])
def check_job_status(job_id):
    """Check status of queued job"""
    if not redis_conn:
        return jsonify({
            "error": "Queue service unavailable",
            "status": "error"
        }), 503

    try:
        job = Job.fetch(job_id, connection=redis_conn)

        response = {
            "job_id": job_id,
            "status": job.get_status(),
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "ended_at": job.ended_at.isoformat() if job.ended_at else None
        }

        if job.is_finished:
            response["result"] = job.result
            response["message"] = "Processing complete"
            response["status_code"] = 200
        elif job.is_failed:
            response["error"] = str(job.exc_info) if job.exc_info else "Unknown error"
            response["message"] = "Processing failed"
            response["status_code"] = 500
        elif job.is_started:
            response["message"] = "Processing in progress"
            response["status_code"] = 202
        else:
            response["message"] = "Job queued, waiting for worker"
            response["position"] = job.get_position()
            response["status_code"] = 202

        return jsonify(response), response.get("status_code", 200)

    except Exception as e:
        return jsonify({
            "error": "Job not found or expired",
            "job_id": job_id,
            "message": str(e),
            "status": "error"
        }), 404

@app.route('/api/kyc/result/<request_id>', methods=['GET'])
def get_result(request_id: str):
    """Retrieve cached result by request ID (for synchronous processing)"""
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

@app.route('/api/kyc/queue/stats', methods=['GET'])
def queue_stats():
    """Get queue statistics"""
    if not redis_conn or not kyc_queue:
        return jsonify({
            "error": "Queue service unavailable",
            "status": "error"
        }), 503

    try:
        from rq import Worker

        started_registry = StartedJobRegistry('kyc_processing', connection=redis_conn)
        finished_registry = FinishedJobRegistry('kyc_processing', connection=redis_conn)
        failed_registry = FailedJobRegistry('kyc_processing', connection=redis_conn)

        return jsonify({
            "queued_jobs": len(kyc_queue),
            "started_jobs": len(started_registry),
            "finished_jobs": len(finished_registry),
            "failed_jobs": len(failed_registry),
            "workers": Worker.count(connection=redis_conn),
            "queue_name": "kyc_processing",
            "redis_url": app.config['REDIS_URL']
        })
    except Exception as e:
        return jsonify({
            "error": "Failed to get queue stats",
            "message": str(e),
            "status": "error"
        }), 500

@app.route('/api/kyc/system/status', methods=['GET'])
def system_status():
    """Get system status and configuration"""
    redis_status = "disconnected"
    if redis_conn:
        try:
            redis_conn.ping()
            redis_status = "connected"
        except:
            redis_status = "error"

    return jsonify({
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "max_workers": Config.MAX_WORKERS,
            "max_retries": Config.MAX_RETRIES,
            "extraction_model": Config.EXTRACTION_MODEL,
            "extraction_fallback": Config.EXTRACTION_FALLBACK_MODEL,
            "verification_model": Config.VERIFICATION_MODEL,
            "verification_fallback": Config.VERIFICATION_FALLBACK_MODEL,
            "max_file_size": f"{app.config['MAX_CONTENT_LENGTH'] / (1024*1024)} MB"
        },
        "cache_size": len(results_cache),
        "storage_mode": "temporary-in-memory",
        "data_retention": "no-permanent-storage",
        "queue_status": redis_status,
        "fallback_enabled": True
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
    port = int(os.getenv('PORT', 5001))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

    print("=" * 80)
    print("🚀 KYC Processing Server Starting")
    print("=" * 80)
    print(f"🌐 Port: {port}")
    print(f"🔧 Max workers: {Config.MAX_WORKERS}")
    print(f"🔄 Max retries: {Config.MAX_RETRIES}")
    print(f"💾 Storage mode: Temporary in-memory only")
    print(f"🧹 Cache cleanup: Active (24h retention)")
    print(f"🔄 Fallback: OpenRouter models enabled")
    print(f"📋 Queue: {'Enabled' if redis_conn else 'Disabled (Redis not connected)'}")
    print("=" * 80)
    print()
    print("📍 Available Endpoints:")
    print("  GET  /health                    - Health check")
    print("  POST /api/kyc/process           - Synchronous processing")
    print("  POST /api/kyc/process-async     - Asynchronous processing (queue)")
    print("  GET  /api/kyc/status/<job_id>  - Check job status")
    print("  GET  /api/kyc/queue/stats       - Queue statistics")
    print("  GET  /api/kyc/system/status     - System status")
    print("=" * 80)

    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )