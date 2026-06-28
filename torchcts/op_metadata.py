# Copyright (c) 2026 Kris Bailey <kris@krisbailey.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies or substantial portions of the Software.

from __future__ import annotations

from functools import lru_cache
import json
from importlib import resources


@lru_cache(maxsize=1)
def load_op_metadata() -> dict:
    """Load TorchCTS-owned generic PyTorch op metadata."""

    text = resources.files("torchcts").joinpath("op_metadata.json").read_text(encoding="utf-8")
    return json.loads(text)


def get_op_metadata(dispatcher_name: str) -> dict:
    """Return metadata for an exact dispatcher name, if present."""

    name = dispatcher_name if dispatcher_name.startswith("aten::") else f"aten::{dispatcher_name}"
    return dict(load_op_metadata().get("ops", {}).get(name, {}))


__all__ = ["get_op_metadata", "load_op_metadata"]
