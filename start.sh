#!/usr/bin/env bash

# Exit on error
set -o errexit

echo "--- 🚀 BONEAI STARTUP INITIATED ---"

# 1. Final Schema Synchronization
echo "--- 🛠️ Applying Database Migrations ---"
python manage.py makemigrations admins users --noinput
python manage.py migrate --noinput

# 2. BoneAI Directory Readiness
echo "--- 📂 Ensuring Media Folders Exist ---"
mkdir -p media/uploads/originals media/uploads/

# 3. Production Server Start
echo "--- 🔥 Starting Gunicorn Production Server ---"
gunicorn Bone_Abnormality_Detection.wsgi:application --timeout 120 --workers 2
