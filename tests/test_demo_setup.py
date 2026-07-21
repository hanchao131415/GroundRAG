from pathlib import Path


def test_prepare_demo_is_idempotent_and_returns_reviewer_walkthrough(tmp_path):
    from scripts.demo_setup import prepare_demo

    first = prepare_demo(tmp_path)
    second = prepare_demo(tmp_path)

    files = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert len(files) >= 10
    assert first.created >= 10
    assert second.created == 0
    assert second.existing == len(files)
    assert {item.user_id for item in second.walkthrough} == {"zhangsan", "lisi", "wangwu", "admin"}
    assert all(item.question for item in second.walkthrough)
    assert Path(tmp_path, "HR", "年假管理制度.md").exists()
