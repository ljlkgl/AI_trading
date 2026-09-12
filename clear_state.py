"""清除历史残留状态脚本。

删除本地持久化的「模型状态」文件，让系统以崭新状态启动：
- state/theses.json             操作理由列表（历史残留的持仓/挂单理由）
- state/experience_library.json 自主经验库（历史经验）
- state/watch_triggers.json     唤醒条件（价格触发）
- state/rounds.json             轮次历史（状态面板展示用）
- state/state.json              运行时状态（含上一轮执行反馈 last_feedback）

不会删除 .env（API Key 等配置保留），也不会删除 logs/（日志仅展示用）。
建议在系统停止运行后执行；运行中的进程会重新生成这些文件。

用法：
  python clear_state.py          # 先列出待删文件并请求确认
  python clear_state.py --yes    # 直接删除，不询问
  python clear_state.py --list   # 仅列出待删文件，不删除
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 系统状态文件（相对项目根目录）。web_ctl.json 为 web 服务瞬时控制文件，不属于模型状态，不清理。
_STATE_FILES = (
    "state/theses.json",
    "state/experience_library.json",
    "state/watch_triggers.json",
    "state/rounds.json",
    "state/state.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="清除交易系统历史残留状态（不影响 .env）")
    parser.add_argument("--yes", "-y", action="store_true", help="不询问，直接删除")
    parser.add_argument("--list", action="store_true", help="仅列出待删文件，不删除")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    existing = [p for p in (root / rel for rel in _STATE_FILES) if p.exists()]

    if not existing:
        print("没有需要清理的状态文件（系统状态目录为空或文件不存在）。")
        return 0

    print("将删除以下状态文件：")
    for p in existing:
        print(f"  - {p}")
    if args.list:
        return 0

    if not args.yes:
        try:
            ans = input("确认删除以上文件？[y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans not in ("y", "yes"):
            print("已取消，未删除任何文件。")
            return 1

    removed = 0
    for p in existing:
        try:
            p.unlink()
            removed += 1
            print(f"已删除 {p}")
        except OSError as exc:
            print(f"删除失败 {p}: {exc}", file=sys.stderr)

    print(f"共删除 {removed} 个状态文件。系统下次启动将以崭新状态运行；.env 未受影响。")
    if removed:
        print("提示：请确保系统已停止运行；运行中的进程会重新生成这些文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
