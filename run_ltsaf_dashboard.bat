@echo off
cd /d "%~dp0"

if exist .venv312\Scripts\activate.bat (
    call .venv312\Scripts\activate.bat
) else (
    if exist venv312\Scripts\activate.bat (
        call venv312\Scripts\activate.bat
    )
)

python -m python -m streamlit run dashboard\ltsaf_live_dashboard.py

pause