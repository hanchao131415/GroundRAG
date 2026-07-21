"""Prepare deterministic sample documents and print a reviewer walkthrough."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DemoPrompt:
    user_id: str
    label: str
    question: str


@dataclass(frozen=True)
class DemoResult:
    created: int
    existing: int
    walkthrough: tuple[DemoPrompt, ...]


WALKTHROUGH = (
    DemoPrompt("zhangsan", "HR employee", "工作满3年年假几天？"),
    DemoPrompt("lisi", "Finance employee", "一线城市住宿费报销上限是多少？"),
    DemoPrompt("wangwu", "IT employee", "公司密码多久更换一次？"),
    DemoPrompt("admin", "Administrator", "检索全部部门的制度文档并比较来源。"),
)


def _document_files(root: Path) -> set[Path]:
    extensions = {".pdf", ".docx", ".md", ".txt", ".xlsx"}
    return {path.relative_to(root) for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions}


def prepare_demo(root: str | Path) -> DemoResult:
    output = Path(root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    before = _document_files(output)
    env = os.environ.copy()
    env["GROUNDRAG_DOCS_PATH"] = str(output)
    generator = Path(__file__).with_name("make_sample_docs.py")
    subprocess.run([sys.executable, str(generator)], check=True, env=env)
    after = _document_files(output)
    return DemoResult(created=len(after - before), existing=len(after & before), walkthrough=WALKTHROUGH)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare GroundRAG sample documents and reviewer prompts")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "docs")
    args = parser.parse_args()
    result = prepare_demo(args.output)
    print(f"Demo documents ready: {result.created} created, {result.existing} existing")
    for item in result.walkthrough:
        print(f"- {item.user_id} ({item.label}): {item.question}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
