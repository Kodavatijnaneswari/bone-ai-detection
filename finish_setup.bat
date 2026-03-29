@echo off
echo Copying new high-resolution YOLO model to the web server...
PowerShell -Command "$latest_run = Get-ChildItem -Path 'runs\detect' -Filter 'clean_bone_model*' -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1; $source = Join-Path $latest_run.FullName 'weights\best.pt'; $destination = 'media\YOLOv8x-best.pt'; Copy-Item -Path $source -Destination $destination -Force"
echo.
echo ==================================================
echo ✅ Model Successfully Installed!
echo.
echo IMPORTANT: To load the new model into memory, you MUST 
echo restart the Django server. 
echo.
echo 1. Go to your running Django terminal.
echo 2. Press Ctrl+C to kill the server.
echo 3. Type: python manage.py runserver
echo ==================================================
pause
