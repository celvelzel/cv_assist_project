#!/usr/bin/env python3
"""
CV 视觉辅助系统 - 主入口
"""

import sys
import os
import faulthandler

# 原生扩展(C++/CUDA 等)崩溃时尽量在 stderr 打出 Python 栈，便于排查「突然卡退无日志」
faulthandler.enable()

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.system import main

if __name__ == "__main__":
    main()
