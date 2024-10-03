@echo off
:loop
python code/datasets/data_engineering.py
python code/models/model.py
timeout /t 300
goto loop
