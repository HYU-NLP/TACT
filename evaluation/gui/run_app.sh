#!/bin/bash

set -a
source .env
set +a

PORT=${FLASK_PORT:-11001}
HOST=${FLASK_HOST:-0.0.0.0}

echo "Running Flask app on $HOST:$PORT"
python app.py