from __future__ import annotations

import os
import sys

# 确保项目根目录在 sys.path 中，避免测试收集期导入失败
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


