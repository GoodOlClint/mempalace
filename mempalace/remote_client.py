#!/usr/bin/env python3
"""
remote_client.py — Remote palace client for mining over the network.

Implements the same interface as a ChromaDB collection (add, get, query)
but sends requests to a MemPalace MCP server over TCP via JSON-RPC 2.0.

Supports:
  - Batch writes (50+ drawers per TCP call)
  - Local embedding computation (Apple Silicon >> VM CPU)
  - Connection reuse within a batch

Usage:
    from mempalace.remote_client import RemotePalaceClient

    client = RemotePalaceClient("172.16.10.115:8765")
    collection = client.collection(local_embeddings=True, batch_size=50)

    # Same interface as chromadb collection — works with miner.py and convo_miner.py
    collection.add(documents=[...], ids=[...], metadatas=[...])
"""

import json
import socket
import itertools


class RemotePalaceError(Exception):
    """Raised when the remote MCP server returns an error."""


class RemoteCollection:
    """Drop-in replacement for a ChromaDB collection that forwards to a remote MCP server."""

    def __init__(
        self,
        host: str,
        port: int,
        timeout: float = 60.0,
        local_embeddings: bool = False,
        batch_size: int = 50,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.local_embeddings = local_embeddings
        self.batch_size = batch_size
        self._id_counter = itertools.count(1)
        self._embed_fn = None

    def _get_embed_fn(self):
        """Lazy-load the embedding function."""
        if self._embed_fn is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            self._embed_fn = DefaultEmbeddingFunction()
        return self._embed_fn

    def _call(self, tool_name: str, arguments: dict) -> dict:
        """Send a JSON-RPC tools/call request and return the parsed result."""
        request = {
            "jsonrpc": "2.0",
            "id": next(self._id_counter),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        payload = json.dumps(request) + "\n"

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect((self.host, self.port))
            sock.sendall(payload.encode("utf-8"))

            buf = b""
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
                if b"\n" in buf:
                    break

            response = json.loads(buf.decode("utf-8").strip())

            if "error" in response:
                raise RemotePalaceError(response["error"].get("message", str(response["error"])))

            content = response.get("result", {}).get("content", [])
            if content and content[0].get("type") == "text":
                return json.loads(content[0]["text"])
            return response.get("result", {})
        finally:
            sock.close()

    def add(self, documents: list, ids: list, metadatas: list):
        """Add drawers to the remote palace using batch writes.

        When local_embeddings is enabled, computes embeddings on the client
        (Apple Silicon) and sends them with the batch to skip server-side
        embedding (VM CPU).
        """
        # Build drawer dicts for batch API
        all_drawers = []
        for doc, drawer_id, meta in zip(documents, ids, metadatas):
            drawer = {
                "wing": meta["wing"],
                "room": meta["room"],
                "content": doc,
                "source_file": meta.get("source_file", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "added_by": meta.get("added_by", "remote-miner"),
            }
            all_drawers.append(drawer)

        # Compute local embeddings if enabled
        if self.local_embeddings:
            ef = self._get_embed_fn()
            texts = [d["content"] for d in all_drawers]
            # Batch embed all at once for efficiency
            embeddings = ef(texts)
            for drawer, emb in zip(all_drawers, embeddings):
                drawer["embedding"] = emb.tolist() if hasattr(emb, 'tolist') else list(emb)

        # Send in batches
        for i in range(0, len(all_drawers), self.batch_size):
            batch = all_drawers[i:i + self.batch_size]
            result = self._call("mempalace_add_drawer_batch", {"drawers": batch})
            if not result.get("success"):
                error = result.get("error", "unknown error")
                raise RemotePalaceError(f"Batch write failed: {error}")

    def get(self, where: dict = None, limit: int = 1, **kwargs) -> dict:
        """Check if documents exist. Used by file_already_mined()."""
        source_file = None
        if where and "source_file" in where:
            source_file = where["source_file"]

        if source_file:
            result = self._call("mempalace_file_already_mined", {
                "source_file": source_file,
            })
            if result.get("mined"):
                return {"ids": ["exists"]}
            return {"ids": []}

        return {"ids": []}

    def count(self) -> int:
        """Get total drawer count."""
        result = self._call("mempalace_status", {})
        return result.get("total_drawers", 0)


class RemotePalaceClient:
    """Client that connects to a remote MemPalace MCP server over TCP."""

    def __init__(self, address: str, timeout: float = 60.0):
        """
        Args:
            address: "host:port" string (e.g., "172.16.10.115:8765")
            timeout: Socket timeout in seconds
        """
        host, port = address.rsplit(":", 1)
        self.host = host
        self.port = int(port)
        self.timeout = timeout

    def collection(self, local_embeddings: bool = False, batch_size: int = 50) -> RemoteCollection:
        """Return a RemoteCollection that mimics a ChromaDB collection."""
        return RemoteCollection(
            self.host,
            self.port,
            self.timeout,
            local_embeddings=local_embeddings,
            batch_size=batch_size,
        )

    def verify(self) -> dict:
        """Verify connectivity by calling status."""
        col = RemoteCollection(self.host, self.port, self.timeout)
        return col._call("mempalace_status", {})
