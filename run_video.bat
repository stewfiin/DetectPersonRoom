@echo off
cd /d "%~dp0"
.\venv\Scripts\python.exe detector.py --source "C:\Users\HP\Downloads\gemini_generated_video_0de34cb5.mp4" --loop %*
