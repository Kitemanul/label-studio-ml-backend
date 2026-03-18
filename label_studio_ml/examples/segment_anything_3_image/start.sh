#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec gunicorn --bind :${PORT:-9090} --workers ${WORKERS:-1} --threads ${THREADS:-4} --timeout 0 --pythonpath "${SCRIPT_DIR}" _wsgi:app
