"""
test_closets.py — Tests for the closet (searchable index) layer and the
features that ride on top of it: mine_lock serialization, entity metadata,
hybrid BM25+vector search, and diary ingest.

Coverage map:
  * mine_lock — acquire/release, blocks concurrent acquisition.
  * build_closet_lines — pointer-line shape, header pickup, entity stoplist
    (regression for "When/After/The"), real-name survival, fallback line.
  * upsert_closet_lines — pure overwrite (regression for the append bug),
    char-limit packing without splitting a line.
  * purge_file_closets — scoped to source_file.
  * Project-miner end-to-end rebuild — re-mining with fewer topics fully
    purges leftover numbered closets from a larger prior run.
  * _extract_drawer_ids_from_closet — pointer parsing + dedup.
  * search_memories closet-first path — fallback when empty, chunk-level
    hits with matched_via, no whole-file glue, max_distance enforcement.
  * Entity metadata — extracted, stoplist applied, registry cached by mtime.
  * Real BM25 — real IDF over candidate corpus, hybrid rerank.
  * Diary ingest — drawers + closets created, incremental skips, state
    file lives outside the diary dir, wing-prefixed drawer IDs prevent
    cross-diary collisions, force=True purges leftover closets.
"""

import json
import os
import threading
import time

import yaml

from mempalace.miner import (
    _extract_entities_for_metadata,
    _load_known_entities,
    mine,
)
from mempalace.palace import (
    CLOSET_CHAR_LIMIT,
    build_closet_lines,
    get_closets_collection,
    get_collection,
    mine_lock,
    purge_file_closets,
    upsert_closet_lines,
)
from mempalace.searcher import (
    _bm25_scores,
    _extract_drawer_ids_from_closet,
    _hybrid_rank,
    search_memories,
)


# ── mine_lock ────────────────────────────────────────────────────────────


class TestMineLock:
    def test_lock_acquires_and_releases(self, tmp_path):
        target = str(tmp_path / "lock_target.txt")
        with mine_lock(target):
            lock_dir = os.path.expanduser("~/.mempalace/locks")
            assert os.path.isdir(lock_dir)
        # Re-acquire after release should succeed instantly.
        start = time.time()
        with mine_lock(target):
            pass
        assert time.time() - start < 1.0

    def test_lock_blocks_concurrent_access(self, tmp_path):
        target = str(tmp_path / "concurrent_lock.txt")
        results = []

        def worker(name):
            start = time.time()
            with mine_lock(target):
                results.append((name, time.time() - start))
                time.sleep(0.2)

        t1 = threading.Thread(target=worker, args=("a",))
        t2 = threading.Thread(target=worker, args=("b",))
        t1.start()
        time.sleep(0.05)  # ensure t1 acquires first
        t2.start()
        t1.join()
        t2.join()

        # The second worker must have waited at least most of t1's hold time.
        wait_times = sorted(r[1] for r in results)
        assert (
            wait_times[1] > 0.1
        ), f"second thread should block on mine_lock, waited only {wait_times[1]:.3f}s"


# ── build_closet_lines ─────────────────────────────────────────────────


class TestBuildClosetLines:
    def test_emits_pointer_line_shape(self):
        content = (
            "# Auth rewrite\n\n"
            "Decided we need to migrate to passkeys. "
            "Built the prototype with WebAuthn. "
            "Reviewed the API surface."
        )
        lines = build_closet_lines(
            "/proj/auth.md",
            ["drawer_proj_backend_aaa", "drawer_proj_backend_bbb"],
            content,
            wing="proj",
            room="backend",
        )
        assert lines, "should always emit at least one line"
        for line in lines:
            assert "→" in line, f"line missing pointer arrow: {line!r}"
            parts = line.split("|")
            assert len(parts) == 3, f"expected topic|entities|→refs, got {line!r}"
            assert parts[2].startswith("→")

    def test_extracts_section_headers_as_topics(self):
        content = "# First Header\nbody\n## Second Header\nmore body"
        lines = build_closet_lines("/x.md", ["d1"], content, "w", "r")
        joined = "\n".join(lines).lower()
        assert "first header" in joined
        assert "second header" in joined

    def test_entity_stoplist_filters_sentence_starters(self):
        # "When", "After", "The" repeat 3+ times — old code would index them
        # as entities. Stoplist drops them.
        content = (
            "When the pipeline ran, the result was good. "
            "When the user logged in, the token was issued. "
            "After the migration, the latency dropped. "
            "After the rollback, the latency rose. "
            "The new flow is stable. The audit cleared."
        )
        lines = build_closet_lines("/x.md", ["d1"], content, "w", "r")
        entity_segments = [line.split("|")[1] for line in lines]
        for seg in entity_segments:
            tokens = set(seg.split(";")) if seg else set()
            assert "When" not in tokens
            assert "After" not in tokens
            assert "The" not in tokens

    def test_real_proper_nouns_survive_stoplist(self):
        content = (
            "Igor reviewed the diff. Milla wrote the spec. "
            "Igor pushed the fix. Milla approved the PR. "
            "Igor and Milla shipped together."
        )
        lines = build_closet_lines("/x.md", ["d1"], content, "w", "r")
        joined_entities = ";".join(line.split("|")[1] for line in lines)
        assert "Igor" in joined_entities
        assert "Milla" in joined_entities

    def test_emits_fallback_line_when_nothing_extractable(self):
        content = "lorem ipsum dolor sit amet consectetur adipiscing elit"
        lines = build_closet_lines("/x/notes.txt", ["d1"], content, "wing", "room")
        assert len(lines) == 1
        assert "wing/room/notes" in lines[0]
        assert "→d1" in lines[0]

    def test_pointer_references_first_three_drawers(self):
        ids = [f"drawer_{i}" for i in range(10)]
        lines = build_closet_lines("/x.md", ids, "# A\n# B", "w", "r")
        assert all("→drawer_0,drawer_1,drawer_2" in line for line in lines)


# ── upsert_closet_lines ───────────────────────────────────────────────


class TestUpsertClosetLines:
    def test_overwrites_existing_closet_does_not_append(self, palace_path):
        col = get_closets_collection(palace_path)
        base = "closet_test_room_abc"
        meta = {"wing": "test", "room": "room", "source_file": "/x.md"}

        upsert_closet_lines(col, base, ["alpha|;|→d1", "beta|;|→d2", "gamma|;|→d3"], meta)
        first = col.get(ids=[f"{base}_01"])
        assert "alpha" in first["documents"][0]

        # Second mine — entirely different lines. Must replace, not append.
        upsert_closet_lines(col, base, ["delta|;|→d4", "epsilon|;|→d5"], meta)
        second = col.get(ids=[f"{base}_01"])
        doc = second["documents"][0]
        assert "delta" in doc
        assert "epsilon" in doc
        assert "alpha" not in doc, "old closet line leaked into rebuild"
        assert "beta" not in doc

    def test_packs_into_multiple_closets_without_splitting_lines(self, palace_path):
        col = get_closets_collection(palace_path)
        base = "closet_pack_room_def"
        meta = {"wing": "test", "room": "room", "source_file": "/y.md"}

        line = "x" * 600  # well under CLOSET_CHAR_LIMIT
        n_written = upsert_closet_lines(col, base, [line, line, line, line], meta)
        # 4 lines @ 601 chars each = 2404 — should pack into 2 closets
        assert n_written == 2

        for i in range(1, n_written + 1):
            doc = col.get(ids=[f"{base}_{i:02d}"])["documents"][0]
            for chunk in doc.split("\n"):
                assert len(chunk) == 600, f"line was truncated in closet {i}"
            assert len(doc) <= CLOSET_CHAR_LIMIT


# ── purge_file_closets ────────────────────────────────────────────────


class TestPurgeFileClosets:
    def test_deletes_only_the_targeted_source(self, palace_path):
        col = get_closets_collection(palace_path)
        col.upsert(
            ids=["closet_a_01", "closet_b_01"],
            documents=["a|;|→d1", "b|;|→d2"],
            metadatas=[
                {"source_file": "/keep.md", "wing": "w", "room": "r"},
                {"source_file": "/drop.md", "wing": "w", "room": "r"},
            ],
        )
        purge_file_closets(col, "/drop.md")
        remaining_ids = set(col.get()["ids"])
        assert "closet_a_01" in remaining_ids
        assert "closet_b_01" not in remaining_ids


# ── project miner: closet rebuild end-to-end ──────────────────────────


class TestMinerClosetRebuild:
    def test_remine_replaces_closets_completely(self, tmp_path):
        project = tmp_path / "proj"
        project.mkdir()
        (project / "mempalace.yaml").write_text(
            yaml.dump({"wing": "proj", "rooms": [{"name": "general", "description": "x"}]})
        )
        target = project / "doc.md"

        # First mine — long content produces multiple numbered closets.
        first_topics = "\n\n".join(f"# Topic {i}\n" + ("filler text " * 30) for i in range(15))
        target.write_text(first_topics)
        palace = tmp_path / "palace"
        mine(str(project), str(palace), wing_override="proj", agent="test")

        col = get_closets_collection(str(palace))
        first_pass = col.get(where={"source_file": str(target)})
        assert first_pass["ids"], "first mine should have written closets"
        first_ids = set(first_pass["ids"])
        assert any("topic 0" in (d or "").lower() for d in first_pass["documents"])

        # Touch mtime + shrink content so the rebuild produces fewer closets.
        target.write_text("# Only Topic Now\n" + ("short body " * 5))
        new_mtime = os.path.getmtime(target) + 60
        os.utime(target, (new_mtime, new_mtime))
        time.sleep(0.01)

        mine(str(project), str(palace), wing_override="proj", agent="test")

        col = get_closets_collection(str(palace))
        second_pass = col.get(where={"source_file": str(target)})
        second_docs = "\n".join(second_pass["documents"]).lower()
        assert "only topic now" in second_docs
        for i in range(15):
            assert (
                f"topic {i}\n" not in second_docs
            ), f"stale 'Topic {i}' from first mine survived the rebuild"
        # Numbered closets that existed only in the larger first run must be gone.
        leftover = first_ids - set(second_pass["ids"])
        for stale_id in leftover:
            assert not col.get(ids=[stale_id])[
                "ids"
            ], f"orphan closet {stale_id} from larger first run survived purge"


# ── _extract_drawer_ids_from_closet ───────────────────────────────────


class TestExtractDrawerIds:
    def test_parses_single_pointer(self):
        assert _extract_drawer_ids_from_closet("topic|;|→drawer_x") == ["drawer_x"]

    def test_parses_multiple_pointers_per_line(self):
        line = "topic|ent|→drawer_a,drawer_b,drawer_c"
        assert _extract_drawer_ids_from_closet(line) == ["drawer_a", "drawer_b", "drawer_c"]

    def test_dedupes_across_lines(self):
        doc = "one|;|→drawer_a,drawer_b\ntwo|;|→drawer_b,drawer_c"
        assert _extract_drawer_ids_from_closet(doc) == ["drawer_a", "drawer_b", "drawer_c"]

    def test_empty_doc_returns_empty(self):
        assert _extract_drawer_ids_from_closet("") == []
        assert _extract_drawer_ids_from_closet("no arrows here") == []


# ── search_memories closet-first path ────────────────────────────────


class TestSearchMemoriesClosetFirst:
    def test_falls_back_to_direct_when_no_closets(self, palace_path, seeded_collection):
        result = search_memories("JWT authentication", palace_path)
        assert result["results"], "should still find drawer hits via fallback"
        for hit in result["results"]:
            assert hit.get("matched_via") == "drawer"

    def test_closet_first_returns_chunk_level_hits(self, palace_path, seeded_collection):
        closets = get_closets_collection(palace_path)
        closets.upsert(
            ids=["closet_proj_backend_aaa_01"],
            documents=["JWT auth tokens|;|→drawer_proj_backend_aaa"],
            metadatas=[{"wing": "project", "room": "backend", "source_file": "auth.py"}],
        )

        result = search_memories("JWT authentication", palace_path)
        assert result["results"], "closet-first search should hydrate the drawer"
        top = result["results"][0]
        assert top["matched_via"] == "closet"
        assert "JWT" in top["text"]
        # Chunk-level — must NOT glue every drawer in the file together.
        assert "Database migrations" not in top["text"]
        assert "→drawer_proj_backend_aaa" in top["closet_preview"]

    def test_max_distance_filters_closet_hits(self, palace_path, seeded_collection):
        closets = get_closets_collection(palace_path)
        closets.upsert(
            ids=["closet_proj_backend_aaa_01"],
            documents=["JWT auth tokens|;|→drawer_proj_backend_aaa"],
            metadatas=[{"wing": "project", "room": "backend", "source_file": "auth.py"}],
        )
        result = search_memories(
            "completely unrelated query about quantum gardening",
            palace_path,
            max_distance=0.001,
        )
        for hit in result["results"]:
            assert hit["distance"] <= 0.001


# ── entity metadata ──────────────────────────────────────────────────


class TestEntityMetadata:
    def test_extracts_capitalized_names(self):
        text = "Ben reviewed the code. Ben approved it. Igor flagged two issues. Igor fixed them."
        entities = _extract_entities_for_metadata(text)
        assert "Ben" in entities
        assert "Igor" in entities

    def test_empty_for_no_entities(self):
        text = "this is all lowercase with no proper nouns at all"
        assert _extract_entities_for_metadata(text) == ""

    def test_semicolon_separated(self):
        text = "Alice and Bob met Charlie. Alice said hello. Bob agreed. Charlie laughed."
        entities = _extract_entities_for_metadata(text)
        assert ";" in entities

    def test_stoplist_filters_sentence_starters(self):
        # Same regression as the closet entity test — "When/After/The" must
        # not become entities just because they're capitalized 2+ times.
        text = (
            "When the build broke, the team paged. "
            "When the fix landed, the alarm cleared. "
            "After the rollback, the queue drained. "
            "After the deploy, the latency normalized."
        )
        entities = _extract_entities_for_metadata(text)
        tokens = set(entities.split(";")) if entities else set()
        assert "When" not in tokens
        assert "After" not in tokens
        assert "The" not in tokens

    def test_capped_list_never_truncates_a_name(self):
        # 30 distinct repeated proper nouns — extraction should cap the list
        # before joining so a name never gets cut in half.
        # Use morphologically distinct stems so the [A-Z][a-z]+ regex sees
        # each as its own token.
        names = [
            "Anna",
            "Brian",
            "Carol",
            "David",
            "Elena",
            "Frank",
            "Grace",
            "Harold",
            "Iris",
            "Julian",
            "Kira",
            "Liam",
            "Maya",
            "Noah",
            "Oscar",
            "Penny",
            "Quinn",
            "Rosa",
            "Sergei",
            "Tara",
            "Umar",
            "Vera",
            "Walter",
            "Xander",
            "Yvonne",
            "Zachary",
            "Amelia",
            "Boris",
            "Clara",
            "Dmitri",
        ]
        text = " ".join(f"{n} met {n}." for n in names)
        entities = _extract_entities_for_metadata(text)
        extracted = [n for n in entities.split(";") if n]
        assert extracted, "should have extracted some entities"
        for name in extracted:
            assert name in names, f"truncation produced a partial token: {name!r}"

    def test_known_registry_is_cached_by_mtime(self, monkeypatch, tmp_path):
        # Point the registry at a temp file we control, exercise the cache.
        registry = tmp_path / "known_entities.json"
        registry.write_text(json.dumps({"people": ["Zelda"]}))
        from mempalace import miner

        monkeypatch.setattr(miner, "_ENTITY_REGISTRY_PATH", str(registry))
        miner._ENTITY_REGISTRY_CACHE["mtime"] = None
        miner._ENTITY_REGISTRY_CACHE["names"] = frozenset()

        first = _load_known_entities()
        assert "Zelda" in first

        # Second call without changing mtime: must reuse cache, not re-read.
        read_count = {"n": 0}
        original_open = open

        def counting_open(path, *a, **kw):
            if str(path) == str(registry):
                read_count["n"] += 1
            return original_open(path, *a, **kw)

        monkeypatch.setattr("builtins.open", counting_open)
        _load_known_entities()
        assert read_count["n"] == 0, "registry should not be re-read when mtime unchanged"

        # Bump mtime → cache must invalidate.
        new_mtime = os.path.getmtime(registry) + 5
        os.utime(registry, (new_mtime, new_mtime))
        registry.write_text(json.dumps({"people": ["Zelda", "Link"]}))
        os.utime(registry, (new_mtime, new_mtime))
        names = _load_known_entities()
        assert "Link" in names


# ── BM25 hybrid search (real IDF over candidate corpus) ──────────────


class TestBM25:
    def test_scores_positive_for_matching_doc(self):
        scores = _bm25_scores(
            "database migration",
            ["We migrated the database to Postgres.", "unrelated cookery tips"],
        )
        assert scores[0] > 0
        assert scores[1] == 0.0

    def test_scores_zero_when_no_overlap(self):
        scores = _bm25_scores("quantum physics", ["We built a web app in React"])
        assert scores == [0.0]

    def test_idf_downweights_terms_present_in_every_doc(self):
        # "database" appears in every candidate → low IDF → low contribution.
        # "vacuum" is unique to one → high IDF → that doc dominates.
        scores = _bm25_scores(
            "database vacuum",
            [
                "database backup nightly schedule",
                "database vacuum scheduled weekly",
                "database failover plan",
            ],
        )
        assert scores[1] == max(scores), "doc with the rare query term should win on IDF"

    def test_empty_inputs_return_zeros(self):
        assert _bm25_scores("", ["hello world"]) == [0.0]
        assert _bm25_scores("query here", []) == []
        assert _bm25_scores("query", [""]) == [0.0]

    def test_hybrid_rank_promotes_keyword_match(self):
        results = [
            {"text": "database schema design for Postgres", "distance": 0.5},
            {"text": "unrelated topic about cooking", "distance": 0.3},
        ]
        ranked = _hybrid_rank(results, "database Postgres schema")
        # The keyword-rich result outranks the closer-vector but irrelevant one.
        assert "database" in ranked[0]["text"]
        # bm25_score field is exposed for debugging.
        assert "bm25_score" in ranked[0]
        # No internal scoring leak.
        assert "_hybrid_score" not in ranked[0]

    def test_hybrid_rank_absolute_normalization(self):
        # Adding a much-worse result to the candidate set must NOT reshuffle
        # the top two — proves we're using absolute (1 - dist) and not
        # dist / max_dist normalization.
        base = [
            {"text": "alpha alpha alpha", "distance": 0.1},
            {"text": "beta beta beta", "distance": 0.4},
        ]
        ranked_short = _hybrid_rank([dict(r) for r in base], "alpha")
        with_outlier = base + [{"text": "gamma gamma gamma", "distance": 1.9}]
        ranked_long = _hybrid_rank([dict(r) for r in with_outlier], "alpha")
        assert ranked_short[0]["text"] == ranked_long[0]["text"]
        assert ranked_short[1]["text"] == ranked_long[1]["text"]


# ── diary ingest ─────────────────────────────────────────────────────


class TestDiaryIngest:
    def test_ingest_creates_drawers_and_closets(self, tmp_path):
        diary_dir = tmp_path / "diaries"
        diary_dir.mkdir()
        (diary_dir / "2026-04-13.md").write_text(
            "# 2026-04-13\n\n## 10:00 PDT — Test\n\nBuilt the auth system.\n"
        )
        palace_dir = tmp_path / "palace"

        from mempalace.diary_ingest import ingest_diaries

        result = ingest_diaries(str(diary_dir), str(palace_dir), force=True)
        assert result["days_updated"] >= 1
        assert get_collection(str(palace_dir)).count() >= 1

    def test_ingest_skips_unchanged_on_second_run(self, tmp_path):
        diary_dir = tmp_path / "diaries"
        diary_dir.mkdir()
        (diary_dir / "2026-04-13.md").write_text(
            "# 2026-04-13\n\n## 10:00 — Test\n\nContent here that's long enough.\n"
        )
        palace_dir = tmp_path / "palace"

        from mempalace.diary_ingest import ingest_diaries

        ingest_diaries(str(diary_dir), str(palace_dir), force=True)
        result = ingest_diaries(str(diary_dir), str(palace_dir))
        assert result["days_updated"] == 0

    def test_state_file_lives_outside_diary_dir(self, tmp_path):
        # Regression: the original implementation wrote
        # ``.diary_ingest_state.json`` *inside* the user's diary directory,
        # polluting their content folder. State must live under
        # ``~/.mempalace/state/`` instead.
        diary_dir = tmp_path / "diaries"
        diary_dir.mkdir()
        (diary_dir / "2026-04-13.md").write_text(
            "# 2026-04-13\n\n## 10:00 — Test\n\nBody content here long enough.\n"
        )
        palace_dir = tmp_path / "palace"

        from mempalace.diary_ingest import _state_file_for, ingest_diaries

        ingest_diaries(str(diary_dir), str(palace_dir), force=True)

        # No state file inside the user's diary dir.
        for entry in diary_dir.iterdir():
            assert (
                "diary_ingest" not in entry.name
            ), f"state file leaked into user diary dir: {entry}"

        # State file does exist under ~/.mempalace/state/.
        state_path = _state_file_for(str(palace_dir), diary_dir.resolve())
        assert state_path.exists()
        assert "/.mempalace/state/" in str(state_path)

    def test_wing_prefixed_drawer_id_prevents_cross_diary_collision(self, tmp_path):
        # Regression: the original implementation used
        # ``drawer_diary_{date_str}`` regardless of wing — two diaries with
        # the same date in different wings would clobber each other.
        date_md = "# 2026-04-13\n\n## 10:00 — entry\n\nThis is the day's content.\n"

        # Two separate diary dirs, ingested into the same palace under
        # different wings. Each must produce a distinct drawer.
        personal_dir = tmp_path / "personal"
        personal_dir.mkdir()
        (personal_dir / "2026-04-13.md").write_text(date_md + "Personal-only marker.\n")

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "2026-04-13.md").write_text(date_md + "Work-only marker.\n")

        palace_dir = tmp_path / "palace"

        from mempalace.diary_ingest import _diary_drawer_id, ingest_diaries

        ingest_diaries(str(personal_dir), str(palace_dir), wing="personal", force=True)
        ingest_diaries(str(work_dir), str(palace_dir), wing="work", force=True)

        col = get_collection(str(palace_dir))
        personal_id = _diary_drawer_id("personal", "2026-04-13")
        work_id = _diary_drawer_id("work", "2026-04-13")
        assert personal_id != work_id

        personal = col.get(ids=[personal_id])
        work = col.get(ids=[work_id])
        assert personal["ids"] == [personal_id]
        assert work["ids"] == [work_id]
        assert "Personal-only marker." in personal["documents"][0]
        assert "Work-only marker." in work["documents"][0]
