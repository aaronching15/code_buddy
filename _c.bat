@echo off
cd /d c:\rc_202X\rc_202X\ciss_web\CISS_rc\apps\agent\code_buddy
git add -A
git commit -m "fix: nan guard in bs price opt pnl cumulative stats and daily table"
git push
