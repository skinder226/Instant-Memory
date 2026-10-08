from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_python_sources_compile():
    files = [ROOT / "main.py", *((ROOT / "src").rglob("*.py"))]
    for path in files:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_prompt_uses_runtime_timestamp_marker():
    prompt = (ROOT / "src/memori/prompts.py").read_text(encoding="utf-8")
    assert '__CURRENT_DATETIME__' in prompt
    assert 'datetime.now(timezone.utc).isoformat()' not in prompt
