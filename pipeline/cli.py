"""Punto de entrada CLI para todas las etapas del pipeline.

Uso:
    python -m pipeline.cli migrate
    python -m pipeline.cli ingest [--api-url URL] [--group GROUP]
    python -m pipeline.cli quality [--batch-id ID]
    python -m pipeline.cli decide [--batch-id ID]
    python -m pipeline.cli preprocess [--batch-id ID]
    python -m pipeline.cli split [--batch-id ID]
    python -m pipeline.cli train [--batch-id ID] [--model NAME]
    python -m pipeline.cli promote
    python -m pipeline.cli audit
    python -m pipeline.cli all [--api-url URL] [--group GROUP]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from pipeline import audit, decide, ingest, preprocess, promote, quality, split, train
from pipeline.db import migrations


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("migrate", help="aplica las migraciones DDL")

    p_ing = sub.add_parser("ingest", help="ingesta un lote desde la API")
    p_ing.add_argument("--api-url", dest="api_url")
    p_ing.add_argument("--group", type=int, dest="group_number")

    for name in ("quality", "preprocess", "split"):
        sp = sub.add_parser(name)
        sp.add_argument("--batch-id", dest="batch_id")

    p_decide = sub.add_parser("decide", help="decide si entrenar")
    p_decide.add_argument("--batch-id", dest="batch_id")

    p_train = sub.add_parser("train", help="entrena modelos candidatos")
    p_train.add_argument("--batch-id", dest="batch_id")
    p_train.add_argument(
        "--model",
        choices=["linear_regression", "random_forest", "gradient_boosting"],
        help="entrena solo este candidato",
    )

    sub.add_parser("promote", help="(requiere candidato; usar 'all')")
    sub.add_parser("audit", help="muestra historial de auditoria")

    p_all = sub.add_parser("all", help="pipeline completo")
    p_all.add_argument("--api-url", dest="api_url")
    p_all.add_argument("--group", type=int, dest="group_number")

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
        _print(ingest.run(api_url=args.api_url, group_number=args.group_number))
        return 0

    if args.command == "quality":
        _print(quality.run(batch_id=args.batch_id))
        return 0

    if args.command == "decide":
        quality_report = quality.run(batch_id=args.batch_id)
        _print(decide.run(quality_report))
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
        print("error: promote requiere un candidato; usar 'all'", file=sys.stderr)
        return 2

    if args.command == "audit":
        _print(audit.get_history())
        return 0

    if args.command == "all":
        migrations.run()

        ingest_summary = ingest.run(api_url=args.api_url, group_number=args.group_number)
        batch = ingest_summary["batch_id"]

        audit_id = audit.create_entry(batch, ingest_summary["total_rows"])
        audit.update_entry(audit_id, rows_after_dedup=ingest_summary["inserted"])

        quality_report = quality.run(batch_id=batch)
        audit.update_entry(
            audit_id,
            schema_ok=quality_report["schema_ok"],
            quality_ok=quality_report["quality_ok"],
            drift_detected=quality_report["drift_detected"],
        )

        decision = decide.run(quality_report)
        audit.update_entry(
            audit_id,
            training_decision=decision["should_train"],
            training_reason=decision["reason"],
        )

        if not decision["should_train"]:
            audit.update_entry(audit_id, status="completed")
            _print({
                "ingest": ingest_summary,
                "quality": quality_report,
                "decision": decision,
                "training": "skipped",
            })
            return 0

        preprocess.run(batch_id=batch)
        split.run(batch_id=batch)
        candidate = train.run(batch_id=batch, training_reason=decision["reason"])

        audit.update_entry(
            audit_id,
            mlflow_run_id=candidate.get("run_id"),
            model_registered=candidate.get("version") is not None,
        )

        result = promote.promote(candidate)
        audit.update_entry(
            audit_id,
            promotion_decision=result.get("promoted", False),
            promotion_reason=result.get("reason", ""),
            champion_metric_before=result.get("champion_mae"),
            champion_metric_after=candidate.get("metric"),
            status="completed",
        )

        _print({
            "ingest": ingest_summary,
            "decision": decision,
            "candidate": candidate,
            "promotion": result,
        })
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
