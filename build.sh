#!/bin/bash
# build.sh

# Build Docker image
docker build -t kyc-processing-api .

# Run locally for testing
docker run -p 5000:5000 \
  -e GROQ_API_KEY=$GROQ_API_KEY \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  -e FLASK_SECRET_KEY=$FLASK_SECRET_KEY \
  kyc-processing-api