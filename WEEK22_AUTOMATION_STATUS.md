\# Week 22 Automation Status



\## Automated Now



The following is automated through Windows Task Scheduler:



\- Monthly pipeline launch

\- Final model run

\- Original ranker branch training

\- Latent-neighbor branch training

\- Final blended portfolio generation

\- Regime filter check

\- Paper trade order sheet

\- Signal ledger update

\- Run summary ledger update

\- Order ledger update

\- Current holdings mark-to-market

\- Monthly update log creation

\- Portfolio value summary in monthly logs



\## Manual / Not Automated Yet



The following is not automated yet:



\- Downloading fresh adjusted close prices

\- Updating the stock universe

\- Rebuilding monthly returns

\- Rebuilding model features

\- Rebuilding stock latent PCA embeddings

\- Rebuilding latent-neighbor features

\- Evaluating prior-month performance

\- Creating rebalance orders from current holdings vs target holdings



\## Main Monthly Command



Manual command:



```powershell

python scripts\\run\_monthly\_update.py

