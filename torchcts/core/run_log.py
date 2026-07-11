# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software.

"""Compact rolling run log for crash and hang diagnosis."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import TextIO

from torchcts.core.result_sanitization import sanitize_text


class RollingRunLog:
    """Keep recent test starts without retaining one line for every test."""

    def __init__(self, path: str | Path, *, window_size: int = 32):
        if window_size < 1:
            raise ValueError("Run-log window size must be positive")
        self.path = Path(path)
        self.window_size = window_size
        self._entries: deque[str] = deque(maxlen=window_size)
        self._handle: TextIO | None = None
        self._started = 0

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def _header(self) -> str:
        return (
            "# TorchCTS rolling run log\n"
            f"# tests started: {self._started}; retained: {len(self._entries)}\n"
        )

    def _rewrite(self) -> None:
        if self._handle is None:
            return
        self._handle.seek(0)
        self._handle.write(self._header())
        self._handle.writelines(self._entries)
        self._handle.truncate()
        self._handle.flush()

    def record_start(self, elapsed_seconds: float, nodeid: str) -> None:
        if self._handle is None:
            raise RuntimeError("Run log is not open")
        self._started += 1
        line = f"{elapsed_seconds:8.1f}s  {sanitize_text(nodeid)}\n"
        self._entries.append(line)
        if self._started == 1 or (self._started - 1) % self.window_size == 0:
            self._rewrite()
        else:
            self._handle.write(line)
            self._handle.flush()

    def close(self) -> None:
        if self._handle is None:
            return
        self._rewrite()
        self._handle.close()
        self._handle = None

    def __enter__(self) -> "RollingRunLog":
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
