# gunicorn_config.py
import os
import multiprocessing

# Bind to Render's PORT env
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"

# Render free tier has limited RAM (512MB)
workers = 1  # Keep it at 1 for free tier
worker_class = "gevent"
worker_connections = 1000

# Timeouts
timeout = 120
keepalive = 2

# Graceful shutdown
graceful_timeout = 30

# Memory management
max_requests = 100
max_requests_jitter = 20

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

proc_name = "kyc-processing-api"