#!/bin/bash
# Setup script for LongContext-ICL-Annotation
# Run once before evaluation to install dependencies and download spacy models.

set -e

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Downloading spacy English model..."
python3 -m spacy download en_core_web_sm

echo "Setup complete."
