#!/usr/bin/env python3
"""
remote_client.py — Remote palace client for mining over the network.

Implements the same interface as a ChromaDB collection (add, get, query)
but sends requests to a MemPalace MCP server over TCP via JSON-RPC 2.0.

Usage:
    from mempalace.remote_client import RemotePalaceClient

    client = RemotePalaceClient("172.16.10.115:8765")
    collection = client.collection()

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

    def __init__(self, host: str, port: int, timeout: float = 30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._id_counter = itertools.count(1)

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

            # Read response — accumulate until we get a complete JSON line
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

            # MCP wraps results in content[0].text as JSON string
            content = response.get("result", {}).get("content", [])
            if content and content[0].get("type") == "text":
                return json.loads(content[0]["text"])
            return response.get("result", {})
        finally:
            sock.close()

    def add(self, documents: list, ids: list, metadatas: list):
        """Add drawers to the remote palace. Mirrors ChromaDB collection.add()."""
        for doc, drawer_id, meta in zip(documents, ids, metadatas):
            result = self._call("mempalace_add_drawer", {
                "wing": meta["wing"],
                "room": meta["room"],
                "content": doc,
                "source_file": meta.get("source_file", ""),
                "added_by": meta.get("added_by", "remote-miner"),
            })
            if not result.get("success") and result.get("reason") != "duplicate":
                error = result.get("error", "unknown error")
                raise RemotePalaceError(f"Failed to add drawer {drawer_id}: {error}")

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

        # Fallback — can't do arbitrary get() remotely without a dedicated tool
        return {"ids": []}

    def count(self) -> int:
        """Get total drawer count."""
        result = self._call("mempalace_status", {})
        return result.get("total_drawers", 0)


class RemotePalaceClient:
    """Client that connects to a remote MemPalace MCP server over TCP."""

    def __init__(self, address: str, timeout: float = 30.0):
        """
        Args:
            address: "host:port" string (e.g., "172.16.10.115:8765")
            timeout: Socket timeout in seconds
        """
        host, port = address.rsplit(":", 1)
        self.host = host
        self.port = int(port)
        self.timeout = timeout

    def collection(self) -> RemoteCollection:
        """Return a RemoteCollection that mimics a ChromaDB collection."""
        return RemoteCollection(self.host, self.port, self.timeout)

    def verify(self) -> dict:
        """Verify connectivity by calling status."""
        col = self.collection()
        return col._call("mempalace_status", {})
