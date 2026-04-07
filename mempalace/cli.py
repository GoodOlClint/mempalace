#!/usr/bin/env python3
"""
MemPalace — Give your AI a memory. No API key required.

Two ways to ingest:
  Projects:      mempalace mine ~/projects/my_app          (code, docs, notes)
  Conversations: mempalace mine ~/chats/ --mode convos     (Claude, ChatGPT, Slack)

Same palace. Same search. Different ingest strategies.

Commands:
    mempalace init <dir>                  Detect rooms from folder structure
    mempalace split <dir>                 Split concatenated mega-files into per-session files
    mempalace mine <dir>                  Mine project files (default)
    mempalace mine <dir> --mode convos    Mine conversation exports
    mempalace search "query"              Find anything, exact words
    mempalace wake-up                     Show L0 + L1 wake-up context
    mempalace wake-up --wing my_app       Wake-up for a specific project
    mempalace status                      Show what's been filed

Examples:
    mempalace init ~/projects/my_app
    mempalace mine ~/projects/my_app
    mempalace mine ~/chats/claude-sessions --mode convos
    mempalace search "why did we switch to GraphQL"
    mempalace search "pricing discussion" --wing my_app --room costs
"""

import os
import sys
import json
import argparse
from pathlib import Path

from .config import MempalaceConfig


def cmd_init(args):
    import json
    from pathlib import Path
    from .room_detector_local import save_config

    project_path = Path(args.dir).expanduser().resolve()
    wing_name = args.wing or project_path.name.lower().replace(" ", "_").replace("-", "_")

    remote = getattr(args, "remote", None)

    if args.blank:
        # Blank init — skip all detection, create minimal config
        rooms = [{"name": "general", "description": "All project files"}]
        save_config(str(project_path), wing_name, rooms)
        MempalaceConfig().init(remote=remote)
        if remote:
            print(f"  Remote palace: {remote}")
        return

    from .entity_detector import scan_for_detection, detect_entities, confirm_entities
    from .room_detector_local import detect_rooms_local

    # Pass 1: auto-detect people and projects from file content
    print(f"\n  Scanning for entities in: {args.dir}")
    files = scan_for_detection(args.dir)
    if files:
        print(f"  Reading {len(files)} files...")
        detected = detect_entities(files)
        total = len(detected["people"]) + len(detected["projects"]) + len(detected["uncertain"])
        if total > 0:
            confirmed = confirm_entities(detected, yes=getattr(args, "yes", False))
            # Save confirmed entities to <project>/entities.json for the miner
            if confirmed["people"] or confirmed["projects"]:
                entities_path = project_path / "entities.json"
                with open(entities_path, "w") as f:
                    json.dump(confirmed, f, indent=2)
                print(f"  Entities saved: {entities_path}")
        else:
            print("  No entities detected — proceeding with directory-based rooms.")

    # Pass 2: detect rooms from folder structure
    detect_rooms_local(project_dir=args.dir, yes=getattr(args, "yes", False), wing_override=wing_name)
    MempalaceConfig().init(remote=remote)
    if remote:
        print(f"  Remote palace: {remote}")


def cmd_mine(args):
    remote = getattr(args, "remote", None) or MempalaceConfig().remote

    if remote:
        from .remote_client import RemotePalaceClient

        client = RemotePalaceClient(remote)
        print(f"\n  Connecting to remote palace at {remote}...")
        try:
            status = client.verify()
            print(f"  Connected. Remote palace has {status.get('total_drawers', '?')} drawers.\n")
        except Exception as e:
            print(f"  ERROR: Could not connect to {remote}: {e}")
            return

        collection = client.collection()

        if args.mode == "convos":
            from .convo_miner import mine_convos

            mine_convos(
                convo_dir=args.dir,
                palace_path=None,
                wing=args.wing,
                agent=args.agent,
                limit=args.limit,
                dry_run=args.dry_run,
                extract_mode=args.extract,
                collection=collection,
            )
        else:
            from .miner import mine

            mine(
                project_dir=args.dir,
                palace_path=None,
                wing_override=args.wing,
                agent=args.agent,
                limit=args.limit,
                dry_run=args.dry_run,
                collection=collection,
            )
    else:
        palace_path = os.path.expanduser(args.palace) if args.palace else MempalaceConfig().palace_path

        if args.mode == "convos":
            from .convo_miner import mine_convos

            mine_convos(
                convo_dir=args.dir,
                palace_path=palace_path,
                wing=args.wing,
                agent=args.agent,
                limit=args.limit,
                dry_run=args.dry_run,
                extract_mode=args.extract,
            )
        else:
            from .miner import mine

            mine(
                project_dir=args.dir,
                palace_path=palace_path,
                wing_override=args.wing,
                agent=args.agent,
                limit=args.limit,
                dry_run=args.dry_run,
                respect_gitignore=not args.no_gitignore,
            )


def cmd_search(args):
    remote = MempalaceConfig().remote

    if remote:
        from .remote_client import RemotePalaceClient
        client = RemotePalaceClient(remote)
        result = client.collection()._call("mempalace_search", {
            "query": args.query,
            "wing": args.wing or "",
            "room": args.room or "",
            "limit": args.results,
        })
        if "error" in result:
            print(f"\n  Error: {result['error']}")
            return
        hits = result.get("results", [])
        print(f"\n{'=' * 56}")
        print(f"  Search: \"{args.query}\"  ({len(hits)} results)")
        print(f"{'=' * 56}\n")
        for h in hits:
            sim = h.get("similarity", "?")
            wing = h.get("wing", "?")
            room = h.get("room", "?")
            src = h.get("source_file", "?")
            print(f"  [{sim}] {wing}/{room}  ← {src}")
            print(f"  {h.get('text', '')[:300]}")
            print(f"  {'─' * 56}")
    else:
        from .searcher import search

        palace_path = os.path.expanduser(args.palace) if args.palace else MempalaceConfig().palace_path
        try:
            search(
                query=args.query,
                palace_path=palace_path,
                wing=args.wing,
                room=args.room,
                n_results=args.results,
            )
        except RuntimeError as e:
            print(f"\n  Error: {e}")
            sys.exit(1)


def cmd_wakeup(args):
    """Show L0 (identity) + L1 (essential story) — the wake-up context."""
    from .layers import MemoryStack

    palace_path = os.path.expanduser(args.palace) if args.palace else MempalaceConfig().palace_path
    stack = MemoryStack(palace_path=palace_path)

    text = stack.wake_up(wing=args.wing)
    tokens = len(text) // 4
    print(f"Wake-up text (~{tokens} tokens):")
    print("=" * 50)
    print(text)


def cmd_split(args):
    """Split concatenated transcript mega-files into per-session files."""
    from .split_mega_files import main as split_main
    import sys

    # Rebuild argv for split_mega_files argparse
    argv = [args.dir]
    if args.output_dir:
        argv += ["--output-dir", args.output_dir]
    if args.dry_run:
        argv.append("--dry-run")
    if args.min_sessions != 2:
        argv += ["--min-sessions", str(args.min_sessions)]
    if args.palace:
        argv += ["--palace", args.palace]

    old_argv = sys.argv
    sys.argv = ["mempalace split"] + argv
    try:
        split_main()
    finally:
        sys.argv = old_argv


def cmd_serve(args):
    from .mcp_server import handle_request

    import socket
    import threading
    import logging

    logger = logging.getLogger("mempalace_serve")
    port = args.tcp_port or args.port
    host = args.host
    lock = threading.Lock()

    def handle_client(conn, addr):
        try:
            buf = b""
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buf += chunk
                if b"\n" in buf:
                    break
            line = buf.decode("utf-8").strip()
            if not line:
                return
            request = json.loads(line)
            with lock:
                response = handle_request(request)
            if response is not None:
                conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
        except Exception as e:
            logger.error(f"Client error ({addr}): {e}")
            try:
                error_resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32000, "message": str(e)}}
                conn.sendall((json.dumps(error_resp) + "\n").encode("utf-8"))
            except Exception:
                pass
        finally:
            conn.close()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(16)
    print(f"MemPalace MCP Server listening on {host}:{port}")

    try:
        while True:
            conn, addr = server.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.close()


def cmd_status(args):
    remote = MempalaceConfig().remote

    if remote:
        from .remote_client import RemotePalaceClient
        client = RemotePalaceClient(remote)
        try:
            result = client.collection()._call("mempalace_status", {})
        except Exception as e:
            print(f"\n  Error connecting to {remote}: {e}")
            return
        if "error" in result:
            print(f"\n  {result['error']}")
            print(f"  Palace: {result.get('palace_path', remote)}")
            return
        print(f"\n{'=' * 55}")
        print(f"  MemPalace Status — {result.get('total_drawers', '?')} drawers (remote: {remote})")
        print(f"{'=' * 55}\n")
        wing_rooms = result.get("wing_rooms", {})
        wings = result.get("wings", {})
        for wing_name in sorted(wings.keys()):
            print(f"  WING: {wing_name}")
            if wing_name in wing_rooms:
                for room, count in sorted(wing_rooms[wing_name].items(), key=lambda x: x[1], reverse=True):
                    print(f"    ROOM: {room:20} {count:5} drawers")
            else:
                print(f"    {wings[wing_name]} drawers")
            print()
        print(f"{'=' * 55}\n")
    else:
        from .miner import status

        palace_path = os.path.expanduser(args.palace) if args.palace else MempalaceConfig().palace_path
        status(palace_path=palace_path)


def cmd_compress(args):
    """Compress drawers in a wing using AAAK Dialect."""
    import chromadb
    from .dialect import Dialect

    palace_path = os.path.expanduser(args.palace) if args.palace else MempalaceConfig().palace_path

    # Load dialect (with optional entity config)
    config_path = args.config
    if not config_path:
        for candidate in ["entities.json", os.path.join(palace_path, "entities.json")]:
            if os.path.exists(candidate):
                config_path = candidate
                break

    if config_path and os.path.exists(config_path):
        dialect = Dialect.from_config(config_path)
        print(f"  Loaded entity config: {config_path}")
    else:
        dialect = Dialect()

    # Connect to palace
    try:
        client = chromadb.PersistentClient(path=palace_path)
        col = client.get_collection("mempalace_drawers")
    except Exception:
        print(f"\n  No palace found at {palace_path}")
        print("  Run: mempalace init <dir> then mempalace mine <dir>")
        sys.exit(1)

    # Query drawers in batches to avoid SQLite variable limit (~999)
    where = {"wing": args.wing} if args.wing else None
    _BATCH = 500
    docs, metas, ids = [], [], []
    offset = 0
    while True:
        try:
            kwargs = {"include": ["documents", "metadatas"], "limit": _BATCH, "offset": offset}
            if where:
                kwargs["where"] = where
            batch = col.get(**kwargs)
        except Exception as e:
            if not docs:
                print(f"\n  Error reading drawers: {e}")
                sys.exit(1)
            break
        batch_docs = batch.get("documents", [])
        if not batch_docs:
            break
        docs.extend(batch_docs)
        metas.extend(batch.get("metadatas", []))
        ids.extend(batch.get("ids", []))
        offset += len(batch_docs)
        if len(batch_docs) < _BATCH:
            break

    if not docs:
        wing_label = f" in wing '{args.wing}'" if args.wing else ""
        print(f"\n  No drawers found{wing_label}.")
        return

    print(
        f"\n  Compressing {len(docs)} drawers"
        + (f" in wing '{args.wing}'" if args.wing else "")
        + "..."
    )
    print()

    total_original = 0
    total_compressed = 0
    compressed_entries = []

    for doc, meta, doc_id in zip(docs, metas, ids):
        compressed = dialect.compress(doc, metadata=meta)
        stats = dialect.compression_stats(doc, compressed)

        total_original += stats["original_chars"]
        total_compressed += stats["compressed_chars"]

        compressed_entries.append((doc_id, compressed, meta, stats))

        if args.dry_run:
            wing_name = meta.get("wing", "?")
            room_name = meta.get("room", "?")
            source = Path(meta.get("source_file", "?")).name
            print(f"  [{wing_name}/{room_name}] {source}")
            print(
                f"    {stats['original_tokens']}t -> {stats['compressed_tokens']}t ({stats['ratio']:.1f}x)"
            )
            print(f"    {compressed}")
            print()

    # Store compressed versions (unless dry-run)
    if not args.dry_run:
        try:
            comp_col = client.get_or_create_collection("mempalace_compressed")
            for doc_id, compressed, meta, stats in compressed_entries:
                comp_meta = dict(meta)
                comp_meta["compression_ratio"] = round(stats["ratio"], 1)
                comp_meta["original_tokens"] = stats["original_tokens"]
                comp_col.upsert(
                    ids=[doc_id],
                    documents=[compressed],
                    metadatas=[comp_meta],
                )
            print(
                f"  Stored {len(compressed_entries)} compressed drawers in 'mempalace_compressed' collection."
            )
        except Exception as e:
            print(f"  Error storing compressed drawers: {e}")
            sys.exit(1)

    # Summary
    ratio = total_original / max(total_compressed, 1)
    orig_tokens = Dialect.count_tokens("x" * total_original)
    comp_tokens = Dialect.count_tokens("x" * total_compressed)
    print(f"  Total: {orig_tokens:,}t -> {comp_tokens:,}t ({ratio:.1f}x compression)")
    if args.dry_run:
        print("  (dry run -- nothing stored)")


def main():
    parser = argparse.ArgumentParser(
        description="MemPalace — Give your AI a memory. No API key required.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--palace",
        default=None,
        help="Where the palace lives (default: from ~/.mempalace/config.json or ~/.mempalace/palace)",
    )

    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", help="Detect rooms from your folder structure")
    p_init.add_argument("dir", help="Project directory to set up")
    p_init.add_argument(
        "--yes", "-y", action="store_true", help="Auto-accept all detected entities and rooms (fully non-interactive)"
    )
    p_init.add_argument(
        "--blank", action="store_true",
        help="Skip entity/room detection entirely — create minimal config with a single 'general' room",
    )
    p_init.add_argument(
        "--wing", default=None,
        help="Override wing name (default: directory name)",
    )
    p_init.add_argument(
        "--remote", default=None,
        help="Remote MCP server address (host:port) — store in config for all subsequent commands",
    )

    # mine
    p_mine = sub.add_parser("mine", help="Mine files into the palace")
    p_mine.add_argument("dir", help="Directory to mine")
    p_mine.add_argument(
        "--mode",
        choices=["projects", "convos"],
        default="projects",
        help="Ingest mode: 'projects' for code/docs (default), 'convos' for chat exports",
    )
    p_mine.add_argument("--wing", default=None, help="Wing name (default: directory name)")
    p_mine.add_argument(
        "--agent",
        default="mempalace",
        help="Your name — recorded on every drawer (default: mempalace)",
    )
    p_mine.add_argument("--limit", type=int, default=0, help="Max files to process (0 = all)")
    p_mine.add_argument(
        "--dry-run", action="store_true", help="Show what would be filed without filing"
    )
    p_mine.add_argument(
        "--extract",
        choices=["exchange", "general"],
        default="exchange",
        help="Extraction strategy for convos mode: 'exchange' (default) or 'general' (5 memory types)",
    )
    p_mine.add_argument(
        "--no-gitignore",
        action="store_true",
        help="Don't respect .gitignore files when scanning (index everything)",
    )
    p_mine.add_argument(
        "--remote",
        default=None,
        help="Remote MCP server address (host:port) — mine locally, ingest remotely",
    )

    # search
    p_search = sub.add_parser("search", help="Find anything, exact words")
    p_search.add_argument("query", help="What to search for")
    p_search.add_argument("--wing", default=None, help="Limit to one project")
    p_search.add_argument("--room", default=None, help="Limit to one room")
    p_search.add_argument("--results", type=int, default=5, help="Number of results")

    # compress
    p_compress = sub.add_parser(
        "compress", help="Compress drawers using AAAK Dialect (~30x reduction)"
    )
    p_compress.add_argument("--wing", default=None, help="Wing to compress (default: all wings)")
    p_compress.add_argument(
        "--dry-run", action="store_true", help="Preview compression without storing"
    )
    p_compress.add_argument(
        "--config", default=None, help="Entity config JSON (e.g. entities.json)"
    )

    # wake-up
    p_wakeup = sub.add_parser("wake-up", help="Show L0 + L1 wake-up context (~600-900 tokens)")
    p_wakeup.add_argument("--wing", default=None, help="Wake-up for a specific project/wing")

    # split
    p_split = sub.add_parser(
        "split",
        help="Split concatenated transcript mega-files into per-session files (run before mine)",
    )
    p_split.add_argument("dir", help="Directory containing transcript files")
    p_split.add_argument(
        "--output-dir",
        default=None,
        help="Write split files here (default: same directory as source files)",
    )
    p_split.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be split without writing files",
    )
    p_split.add_argument(
        "--min-sessions",
        type=int,
        default=2,
        help="Only split files containing at least N sessions (default: 2)",
    )

    # status
    sub.add_parser("status", help="Show what's been filed")

    # serve
    p_serve = sub.add_parser("serve", help="Run the MCP server (TCP mode for remote access)")
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=8765, help="TCP port (default: 8765)")
    p_serve.add_argument("--tcp-port", type=int, default=None, help="Alias for --port (backwards compat)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    dispatch = {
        "init": cmd_init,
        "mine": cmd_mine,
        "split": cmd_split,
        "search": cmd_search,
        "compress": cmd_compress,
        "wake-up": cmd_wakeup,
        "status": cmd_status,
        "serve": cmd_serve,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
