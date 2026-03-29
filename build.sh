#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

# Ensure media directories exist for X-ray storage
mkdir -p media/uploads/originals media/uploads/ 2>/dev/null || true

# Execution permissions for the startup script
chmod +x start.sh

# Clean up runs/ folder to save disk space on Render
rm -rf runs/ detect/ 2>/dev/null || true

python manage.py collectstatic --noinput
python manage.py migrate --noinput
