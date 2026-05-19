"""Single CLI entrypoint for every pipeline step.

Usage:
    python -m pipeline.cli migrate
    python -m pipeline.cli ingest [--source PATH] [--batch-id ID]
    python -m pipeline.cli quality [--batch-id ID]
    python -m pipeline.cli preprocess [--batch-id ID]
    python -m pipeline.cli split [--batch-id ID]
    python -m pipeline.cli train [--batch-id ID]
    python -m pipeline.cli promote
    python -m pipeline.cli all [--source PATH] [--batch-id ID]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from pipeline import ingest, preprocess, promote, quality, split, train
from pipeline.db import migrations


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("migrate", help="apply DDL migrations")

    p_ing = sub.add_parser("ingest", help="ingest CSV into raw layer")
    p_ing.add_argument("--source")
    p_ing.add_argument("--batch-id", dest="batch_id")

    for name in ("quality", "preprocess", "split"):
        sp = sub.add_parser(name)
        sp.add_argument("--batch-id", dest="batch_id")

    p_train = sub.add_parser("train", help="train one or all candidates")
    p_train.add_argument("--batch-id", dest="batch_id")
    p_train.add_argument(
        "--model",
        choices=["lr", "rf", "logistic_regression", "random_forest"],
        help="train only this candidate (default: train both)",
    )

    sub.add_parser("promote", help="(needs a candidate; use `all` instead)")

    p_all = sub.add_parser("all", help="end-to-end pipeline (migrate->promote)")
    p_all.add_argument("--source")
    p_all.add_argument("--batch-id", dest="batch_id")

    return p


def _print(obj) -> None:
    print(json.dumps(obj, default=str, indent=2))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parser().parse_args(argv)

    if args.command == "migrate":
        migrations.run()
        _print({"ok": True})
        return 0

    if args.command == "ingest":
        _print(ingest.load_batch(path=args.source, batch_id=args.batch_id))
        return 0

    if args.command == "quality":
        _print(quality.run(batch_id=args.batch_id))
        return 0

    if args.command == "preprocess":
        _print(preprocess.run(batch_id=args.batch_id))
        return 0

    if args.command == "split":
        _print(split.run(batch_id=args.batch_id))
        return 0

    if args.command == "train":
        _print(train.run(batch_id=args.batch_id, model=getattr(args, "model", None)))
        return 0

    if args.command == "promote":
        print("error: promote requires a candidate; use `all` to chain train->promote", file=sys.stderr)
        return 2

    if args.command == "all":
        migrations.run()
        ingest_summary = ingest.load_batch(path=args.source, batch_id=args.batch_id)
        batch = ingest_summary["batch_id"]
        quality.run(batch_id=batch)
        preprocess.run(batch_id=batch)
        split.run(batch_id=batch)
        candidate = train.run(batch_id=batch)
        result = promote.promote(candidate)
        _print({"ingest": ingest_summary, "candidate": candidate, "promotion": result})
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
