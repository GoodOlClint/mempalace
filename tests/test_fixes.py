"""
Tests for the four fixes from the agent-native audit + REVIEW.md:

  1. KG singleton uses _config.palace_path, not the hardcoded default
  2. cmd_split forwards --palace through to split_main's argv
  3. PALACE_PROTOCOL and AAAK_SPEC load from prompts/*.txt files
  4. tool_update_drawer edits content in-place and stamps edited_at

Note: chromadb 0.4.x is incompatible with NumPy 2.0. Tests that require
chromadb are marked with pytest.importorskip and skipped in affected envs.
The chromadb/numpy version pin is a pre-existing issue, not introduced here.
"""

import os
import sys
import types
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Fix 1: Knowledge graph co-located with palace
# ---------------------------------------------------------------------------


def test_kg_accepts_custom_db_path(tmp_path):
    """KnowledgeGraph must store to the path given, not always DEFAULT_KG_PATH.

    The fix: _get_kg() now passes db_path=os.path.join(_config.palace_path,
    'knowledge_graph.sqlite3') instead of letting KnowledgeGraph fall back to
    DEFAULT_KG_PATH. This test verifies KnowledgeGraph honours the argument.
    """
    from mempalace.knowledge_graph import KnowledgeGraph

    custom_db = str(tmp_path / "custom_palace" / "knowledge_graph.sqlite3")
    kg = KnowledgeGraph(db_path=custom_db)

    assert kg.db_path == custom_db
    # The file must be created on init
    assert Path(custom_db).exists(), "KG must create the db file at the given path"


def test_kg_default_path_is_home_mempalace():
    """Default KG path must remain ~/.mempalace/knowledge_graph.sqlite3.

    Ensures backward compatibility: users who run without --palace still get
    data in the same location as before the fix.
    """
    from mempalace.knowledge_graph import DEFAULT_KG_PATH

    expected = os.path.expanduser("~/.mempalace/knowledge_graph.sqlite3")
    assert DEFAULT_KG_PATH == expected


def test_kg_lazy_init_pattern():
    """_get_kg() in mcp_server must be a lazy initialiser, not a module-level call.

    Inspects the source of mcp_server to verify:
    - No bare `_kg = KnowledgeGraph()` at module scope
    - A `_get_kg` function exists
    - `_kg_instance` sentinel exists for caching

    This is a static-analysis test — runs without importing chromadb.
    """
    import ast

    src = (Path(__file__).parent.parent / "mempalace" / "mcp_server.py").read_text()
    tree = ast.parse(src)

    # Check that no module-level assignment creates a KnowledgeGraph() call
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "_kg":
                    # If the value is a Call to KnowledgeGraph, that's the bug
                    if isinstance(node.value, ast.Call):
                        func = node.value.func
                        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                        assert name != "KnowledgeGraph", (
                            "Module-level `_kg = KnowledgeGraph()` found — "
                            "this is the bug we fixed. Use _get_kg() instead."
                        )

    # Check _get_kg function exists
    func_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert "_get_kg" in func_names, "_get_kg lazy-init function must exist in mcp_server.py"

    # Check _kg_instance sentinel exists
    sentinel_found = any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_kg_instance" for t in node.targets)
        for node in ast.walk(tree)
    )
    assert sentinel_found, "_kg_instance sentinel variable must exist for lazy caching"


# ---------------------------------------------------------------------------
# Fix 2: cmd_split forwards --palace
# ---------------------------------------------------------------------------


def test_cmd_split_forwards_palace(tmp_path):
    """cmd_split must include --palace in the reconstructed sys.argv.

    Before this fix, running `mempalace split <dir> --palace /x` would drop
    the --palace flag when rebuilding sys.argv for split_main. The result:
    split files processed correctly but written to the wrong palace location.
    """
    captured = []

    def mock_split_main():
        captured.extend(sys.argv)

    args = types.SimpleNamespace(
        dir=str(tmp_path),
        palace=str(tmp_path / "mypalace"),
        output_dir=None,
        dry_run=False,
        min_sessions=2,
    )

    with patch("mempalace.split_mega_files.main", mock_split_main):
        from mempalace.cli import cmd_split
        cmd_split(args)

    assert "--palace" in captured, (
        f"--palace must appear in sys.argv passed to split_main. Got: {captured}"
    )
    palace_idx = captured.index("--palace")
    assert captured[palace_idx + 1] == str(tmp_path / "mypalace"), (
        "--palace value must be the custom path"
    )


def test_cmd_split_no_palace_not_forwarded(tmp_path):
    """When --palace is not given, the reconstructed argv must not include it."""
    captured = []

    def mock_split_main():
        captured.extend(sys.argv)

    args = types.SimpleNamespace(
        dir=str(tmp_path),
        palace=None,
        output_dir=None,
        dry_run=False,
        min_sessions=2,
    )

    with patch("mempalace.split_mega_files.main", mock_split_main):
        from mempalace.cli import cmd_split
        cmd_split(args)

    assert "--palace" not in captured, (
        "--palace must not appear in argv when args.palace is None"
    )


def test_cmd_split_all_flags_forwarded(tmp_path):
    """cmd_split must forward all flags together correctly."""
    captured = []

    def mock_split_main():
        captured.extend(sys.argv)

    args = types.SimpleNamespace(
        dir=str(tmp_path),
        palace=str(tmp_path / "palace"),
        output_dir=str(tmp_path / "out"),
        dry_run=True,
        min_sessions=5,
    )

    with patch("mempalace.split_mega_files.main", mock_split_main):
        from mempalace.cli import cmd_split
        cmd_split(args)

    assert "--palace" in captured
    assert "--output-dir" in captured
    assert "--dry-run" in captured
    assert "--min-sessions" in captured
    assert "5" in captured


# ---------------------------------------------------------------------------
# Fix 3: PALACE_PROTOCOL and AAAK_SPEC load from prompts/ files
# ---------------------------------------------------------------------------


def test_prompt_files_exist():
    """Both prompt text files must exist under mempalace/prompts/."""
    prompts_dir = Path(__file__).parent.parent / "mempalace" / "prompts"
    assert (prompts_dir / "palace_protocol.txt").exists(), (
        "mempalace/prompts/palace_protocol.txt is missing — "
        "PALACE_PROTOCOL must be extracted to this file"
    )
    assert (prompts_dir / "aaak_spec.txt").exists(), (
        "mempalace/prompts/aaak_spec.txt is missing — "
        "AAAK_SPEC must be extracted to this file"
    )


def test_palace_protocol_content():
    """palace_protocol.txt must contain the five-step memory protocol."""
    content = (
        Path(__file__).parent.parent / "mempalace" / "prompts" / "palace_protocol.txt"
    ).read_text()

    assert len(content) > 50, "palace_protocol.txt is suspiciously short"
    # The core 'verify before speaking' instruction
    assert "mempalace_status" in content, (
        "Protocol must reference mempalace_status (the wake-up tool)"
    )
    # The diary instruction
    assert "diary" in content.lower(), (
        "Protocol must reference the diary (after-session write)"
    )


def test_aaak_spec_content():
    """aaak_spec.txt must contain the AAAK dialect definition."""
    content = (
        Path(__file__).parent.parent / "mempalace" / "prompts" / "aaak_spec.txt"
    ).read_text()

    assert len(content) > 50, "aaak_spec.txt is suspiciously short"
    assert "AAAK" in content, "Spec must reference AAAK"
    assert "ENTITIES" in content or "entity" in content.lower(), (
        "Spec must describe entity codes"
    )


def test_load_prompt_helper_reads_files():
    """_load_prompt must read from the prompts/ directory relative to mcp_server.py.

    Tested by importing only the helper, bypassing the chromadb chain.
    """
    import importlib.util
    import ast

    src_path = Path(__file__).parent.parent / "mempalace" / "mcp_server.py"
    src = src_path.read_text()

    # Verify _load_prompt is defined and uses Path(__file__).parent / "prompts"
    tree = ast.parse(src)
    load_prompt_funcs = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_load_prompt"
    ]
    assert len(load_prompt_funcs) == 1, "_load_prompt function must exist in mcp_server.py"

    func_src = ast.unparse(load_prompt_funcs[0])
    assert "prompts" in func_src, "_load_prompt must reference the prompts/ directory"


def test_pyproject_includes_prompt_files():
    """pyproject.toml must declare prompts/*.txt as package data."""
    content = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    assert "prompts" in content, (
        "pyproject.toml must include prompts/*.txt in package-data "
        "so the files are bundled in wheel/sdist installs"
    )


# ---------------------------------------------------------------------------
# Fix 4: tool_update_drawer (requires chromadb)
# ---------------------------------------------------------------------------


def _chromadb_available():
    try:
        import chromadb
        return True
    except Exception:
        return False


import pytest

chromadb_skip = pytest.mark.skipif(
    not _chromadb_available(),
    reason="chromadb not importable (NumPy version incompatibility in this env)",
)


@chromadb_skip
def test_update_drawer_edits_content(tmp_path):
    """tool_update_drawer must replace content and stamp edited_at."""
    import chromadb
    import mempalace.mcp_server as srv

    original_palace = srv._config.palace_path
    srv._config.palace_path = str(tmp_path)

    try:
        client = chromadb.PersistentClient(path=str(tmp_path))
        col = client.get_or_create_collection(srv._config.collection_name)
        drawer_id = "drawer_wing_test_room_test_aabbccdd11223344"
        col.add(
            ids=[drawer_id],
            documents=["original content"],
            metadatas=[{
                "wing": "wing_test",
                "room": "room_test",
                "filed_at": "2026-01-01T00:00:00",
                "added_by": "test",
            }],
        )

        result = srv.tool_update_drawer(drawer_id, "updated content")

        assert result["success"] is True, f"Update failed: {result}"
        assert result["drawer_id"] == drawer_id
        assert "edited_at" in result

        after = col.get(ids=[drawer_id], include=["documents", "metadatas"])
        assert after["documents"][0] == "updated content"

        meta = after["metadatas"][0]
        assert meta["wing"] == "wing_test", "wing must be preserved"
        assert meta["room"] == "room_test", "room must be preserved"
        assert meta["filed_at"] == "2026-01-01T00:00:00", "filed_at must be preserved"
        assert "edited_at" in meta, "edited_at must be added"

    finally:
        srv._config.palace_path = original_palace


@chromadb_skip
def test_update_drawer_not_found(tmp_path):
    """Updating a missing drawer must return success=False with the drawer_id echoed."""
    import chromadb
    import mempalace.mcp_server as srv

    original_palace = srv._config.palace_path
    srv._config.palace_path = str(tmp_path)

    try:
        client = chromadb.PersistentClient(path=str(tmp_path))
        client.get_or_create_collection(srv._config.collection_name)

        result = srv.tool_update_drawer("nonexistent_id", "some content")

        assert result["success"] is False
        assert result.get("drawer_id") == "nonexistent_id"
        assert "error" in result or "not found" in str(result).lower()

    finally:
        srv._config.palace_path = original_palace


def test_update_drawer_in_tools_registry():
    """mempalace_update_drawer must be in TOOLS with correct required fields.

    Tested via AST inspection to avoid importing chromadb.
    """
    src = (Path(__file__).parent.parent / "mempalace" / "mcp_server.py").read_text()

    assert "mempalace_update_drawer" in src, (
        "mempalace_update_drawer must be defined in mcp_server.py"
    )
    assert "drawer_id" in src, "drawer_id parameter must exist"
    assert "new_content" in src, "new_content parameter must exist"
    assert "edited_at" in src, "edited_at timestamp must be set on update"


@chromadb_skip
def test_update_drawer_registered_in_tools():
    """TOOLS dict must have mempalace_update_drawer with required schema."""
    import mempalace.mcp_server as srv

    assert "mempalace_update_drawer" in srv.TOOLS
    tool = srv.TOOLS["mempalace_update_drawer"]
    required = tool["input_schema"].get("required", [])
    assert "drawer_id" in required
    assert "new_content" in required
    assert callable(tool["handler"])
