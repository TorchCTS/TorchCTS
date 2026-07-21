#!/usr/bin/env python3
"""Emit the reviewed direct-surface disposition table from the frozen inventory."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "tests/oracles/cases/inventory.json"


def main():
    snapshot = json.loads(INVENTORY.read_text(encoding="ascii"))
    records = snapshot["inventory"]["direct_specs"]
    payload = {
        "schema_version": 1,
        "records": {"CP-DIRECT-DISPOSITIONS": records},
        "counts": {
            "surfaces": len(records),
            "oracle_ids": len({record["oracle_id"] for record in records}),
            "status_groups": len({
                (
                    record["oracle_id"], record["coverage_status"], record["runner"],
                    record["backend_gate"], record["contract_ref"],
                )
                for record in records
            }),
        },
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


if __name__ == "__main__":
    main()
