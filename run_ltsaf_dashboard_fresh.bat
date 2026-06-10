@echo off
cd /d C:\ResearchCode\latent_market_twin

if exist .venv312\Scripts\activate.bat (
    call .venv312\Scripts\activate.bat
) else (
    if exist venv312\Scripts\activate.bat (
        call venv312\Scripts\activate.bat
    )
)

echo.
echo ============================================================
echo Updating LTSAF live portfolio value...
echo ============================================================
python scripts\check_ltsaf_live_value.py

echo.
echo ============================================================
echo Launching LTSAF dashboard...
echo ============================================================
streamlit run dashboard\ltsaf_live_dashboard.py

pause