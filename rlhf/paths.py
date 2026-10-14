"""路径解析 —— 同一份代码在两种布局下都能跑。

  <开发目录>/code/llm/rlhf/   模块根 = <开发目录>/code/llm   含 data/ 的根 = <开发目录>
  <仓库根>/rlhf/              模块根 = <仓库根>              含 data/ 的根 = <仓库根>

模块根判据 = 「同时含 llama/ 与 pipeline/」；含 data/ 的根的判据 = 「含 data/」。
两者都不写死层级数，所以任何一侧加一层目录都不会把脚本改死。
"""
from pathlib import Path

_here = Path(__file__).resolve()


def _module_root() -> Path:
    for anc in _here.parents:
        if (anc / "llama").is_dir() and (anc / "pipeline").is_dir():
            return anc
    raise RuntimeError(f"找不到同时含 llama/ 与 pipeline/ 的祖先目录：{_here}")


MODULE_ROOT = _module_root()

# 含 data/ 的那一级。本地还没建 data/ 时回落到模块根 —— repo 里两者本就是同一个目录
ROOT = MODULE_ROOT
if not (ROOT / "data").is_dir():
    for _anc in _here.parents:
        if (_anc / "data").is_dir():
            ROOT = _anc
            break
