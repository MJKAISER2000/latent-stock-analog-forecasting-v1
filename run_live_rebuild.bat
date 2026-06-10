@echo off
cd /d C:\ResearchCode\latent_market_twin

if exist .venv312\Scripts\activate.bat (
    call .venv312\Scripts\activate.bat
) else (
    if exist venv312\Scripts\activate.bat (
        call venv312\Scripts\activate.bat
    )
)

python scripts\run_live_rebuild_pipeline.py

pause