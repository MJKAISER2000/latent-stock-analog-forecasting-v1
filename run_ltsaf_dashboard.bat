@echo off
cd /d C:\ResearchCode\latent_market_twin

if exist .venv312\Scripts\activate.bat (
    call .venv312\Scripts\activate.bat
) else (
    if exist venv312\Scripts\activate.bat (
        call venv312\Scripts\activate.bat
    )
)

streamlit run dashboard\ltsaf_live_dashboard.py

pause