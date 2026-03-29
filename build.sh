#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

# Clean up runs/ folder to save disk space on Render
rm -rf runs/ detect/ 2>/dev/null || true

python manage.py collectstatic --noinput
python manage.py makemigrations admins users
python manage.py migrate
