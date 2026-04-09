@echo off
REM Windows helper to run main.py in the current Python environment (e.g. conda base with CUDA)

REM UTF-8 环境变量（与 run.sh 保持一致）
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM 自愈：如果缺少核心依赖则自动安装（使用当前环境的 python）
python -c "import torch, transformers, mediapipe" >nul 2>nul
if errorlevel 1 (
    if exist requirements.txt (
        echo [INFO] Installing missing dependencies from requirements.txt
        python -m pip install -r requirements.txt
    ) else (
        echo [WARN] requirements.txt not found, please install dependencies manually
    )
)

REM 始终默认启用 tts 配置；用户传 --config 可覆盖（argparse 后出现的同名参数优先）
python main.py --config tts %*
