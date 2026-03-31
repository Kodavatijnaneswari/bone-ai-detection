# Gunicorn configuration file to prevent Render Deployment Timeouts
import os

PORT = os.environ.get('PORT', '10000')
bind = f"0.0.0.0:{PORT}"

# Increase timeout severely for heavy AI models and cv2 imports
timeout = 200

# Use threads to prevent a single slow request or import from freezing the worker
workers = 2
threads = 4

# Log level to help debug future issues
loglevel = "info"
