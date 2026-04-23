from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from .version import __version__
from .logging_config import configure_logging
from .paths import runs_root
from .contracts.run_config import RunConfig, save_run_config, load_run_config
from .contracts.status import write_status
from .io.dataset_loader import preview_headers

from .pipeline.preprocess import run_preprocess
from .pipeline.eda import run_eda
from .pipeline.split_scale import run_split_scale
from .pipeline.train import run_training
from .pipeline.modelica_export import export_modelica
from .pipeline.report import generate_report
from .pipeline.orchestrator import run_full_attempt

from .attempts.manager import next_attempt_number, list_attempts, select_best_attempt, update_latest_attempt


def _cmd_preview_headers(args, logger) -> int:
    dataset = Path(args.dataset)
    headers = preview_headers(dataset, sheet_name=args.sheet)
    out = {"dataset": str(dataset), "sheet": args.sheet, "headers": headers, "count": len(headers)}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    else:
        print(json.dumps(out, indent=2))
    return 0


def _cmd_make_run(args, logger) -> int:
    rr = runs_root()
    rr.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id
    if not run_id:
        from datetime import datetime
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = rr / run_id
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    dataset_src = Path(args.dataset)
    if not dataset_src.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_src}")

    dataset_dst = input_dir / dataset_src.name
    shutil.copy2(dataset_src, dataset_dst)

    cfg = RunConfig(
        run_id=run_id,
        dataset_path=str(dataset_dst),
        dataset_format="csv" if dataset_dst.suffix.lower() == ".csv" else "xlsx",
        sheet_name=args.sheet,
        package_name=args.package,
        n_inputs=0,
        n_outputs=0,
        input_columns=[],
        output_columns=[],
        engine_version=__version__,
    ).normalized()

    cfg_path = input_dir / "run_config.json"
    save_run_config(cfg, cfg_path)

    status_path = run_dir / "status.json"
    write_status(status_path, state="READY", message="Run workspace created.", progress=0.0)

    latest = rr / "latest_run.json"
    latest.write_text(json.dumps({"run_id": run_id, "run_dir": str(run_dir)}, indent=2), encoding="utf-8")
    print(run_id)
    return 0


def _cmd_validate_config(args, logger) -> int:
    cfg = load_run_config(Path(args.config))
    print(json.dumps(cfg.model_dump(), indent=2))
    return 0


def _cmd_preprocess(args, logger) -> int:
    inputs = [s.strip() for s in (args.inputs or "").split(",") if s.strip()] if args.inputs else None
    outputs = [s.strip() for s in (args.outputs or "").split(",") if s.strip()] if args.outputs else None
    paths = run_preprocess(run_id=args.run_id, attempt_num=int(args.attempt), inputs_override=inputs, outputs_override=outputs)
    print(json.dumps({"cleaned_csv": str(paths.cleaned_csv), "preprocess_report": str(paths.report_json)}, indent=2))
    return 0


def _cmd_eda(args, logger) -> int:
    result = run_eda(run_id=args.run_id, attempt_num=int(args.attempt))
    print(json.dumps(result, indent=2))
    return 0


def _cmd_split_scale(args, logger) -> int:
    result = run_split_scale(run_id=args.run_id, attempt_num=int(args.attempt))
    print(json.dumps(result, indent=2))
    return 0


def _cmd_train(args, logger) -> int:
    hidden = [int(x.strip()) for x in args.hidden.split(",")] if args.hidden else None
    result = run_training(
        run_id=args.run_id,
        attempt_num=int(args.attempt),
        hidden=hidden,
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        patience=int(args.patience),
    )
    print(json.dumps(result, indent=2))
    return 0


def _cmd_export_modelica(args, logger) -> int:
    result = export_modelica(run_id=args.run_id, attempt_num=int(args.attempt))
    print(json.dumps(result, indent=2))
    return 0


def _cmd_report(args, logger) -> int:
    result = generate_report(run_id=args.run_id, attempt_num=int(args.attempt))
    print(json.dumps(result, indent=2))
    return 0


def _cmd_list_attempts(args, logger) -> int:
    infos = list_attempts(args.run_id)
    out = []
    for i in infos:
        out.append({
            "attempt": i.attempt_name,
            "attempt_dir": str(i.attempt_dir),
            "metrics_exists": i.metrics_path.exists(),
            "report_exists": i.report_html.exists(),
        })
    print(json.dumps(out, indent=2))
    return 0


def _cmd_best_attempt(args, logger) -> int:
    latest = select_best_attempt(args.run_id)
    print(json.dumps(latest, indent=2))
    return 0


def _cmd_retrain(args, logger) -> int:
    run_id = args.run_id
    attempt_num = next_attempt_number(run_id)
    # update pointer immediately
    update_latest_attempt(run_id, attempt_num)

    # Run entire pipeline
    result = run_full_attempt(run_id=run_id, attempt_num=attempt_num)

    # After pipeline completes, compute best attempt
    latest = select_best_attempt(run_id)

    print(json.dumps({
        "run_id": run_id,
        "new_attempt": f"attempt_{attempt_num:03d}",
        "latest_json": str((runs_root()/run_id/"latest.json")),
        "best": latest.get("best_attempt"),
        "best_score": latest.get("best_score"),
        "report": result.get("report"),
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="surrogate_tool", description="Offline surrogate generator engine (Excel-driven).")
    parser.add_argument("--hello", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--version", action="store_true")

    sub = parser.add_subparsers(dest="cmd")

    p_prev = sub.add_parser("preview-headers")
    p_prev.add_argument("--dataset", required=True)
    p_prev.add_argument("--sheet", default=None)
    p_prev.add_argument("--out", default=None)
    p_prev.set_defaults(func=_cmd_preview_headers)

    p_run = sub.add_parser("make-run")
    p_run.add_argument("--dataset", required=True)
    p_run.add_argument("--sheet", default=None)
    p_run.add_argument("--package", required=True)
    p_run.add_argument("--run-id", default=None)
    p_run.set_defaults(func=_cmd_make_run)

    p_val = sub.add_parser("validate-config")
    p_val.add_argument("--config", required=True)
    p_val.set_defaults(func=_cmd_validate_config)

    p_pre = sub.add_parser("preprocess")
    p_pre.add_argument("--run-id", required=True)
    p_pre.add_argument("--attempt", default="1")
    p_pre.add_argument("--inputs", default=None)
    p_pre.add_argument("--outputs", default=None)
    p_pre.set_defaults(func=_cmd_preprocess)

    p_eda = sub.add_parser("eda")
    p_eda.add_argument("--run-id", required=True)
    p_eda.add_argument("--attempt", default="1")
    p_eda.set_defaults(func=_cmd_eda)

    p_ss = sub.add_parser("split-scale")
    p_ss.add_argument("--run-id", required=True)
    p_ss.add_argument("--attempt", default="1")
    p_ss.set_defaults(func=_cmd_split_scale)

    p_tr = sub.add_parser("train")
    p_tr.add_argument("--run-id", required=True)
    p_tr.add_argument("--attempt", default="1")
    p_tr.add_argument("--hidden", default=None)
    p_tr.add_argument("--lr", default="1e-3")
    p_tr.add_argument("--batch-size", default="32")
    p_tr.add_argument("--epochs", default="500")
    p_tr.add_argument("--patience", default="30")
    p_tr.set_defaults(func=_cmd_train)

    p_ex = sub.add_parser("export-modelica")
    p_ex.add_argument("--run-id", required=True)
    p_ex.add_argument("--attempt", default="1")
    p_ex.set_defaults(func=_cmd_export_modelica)

    p_rp = sub.add_parser("report")
    p_rp.add_argument("--run-id", required=True)
    p_rp.add_argument("--attempt", default="1")
    p_rp.set_defaults(func=_cmd_report)

    # ✅ NEW in Checkpoint 9
    p_la = sub.add_parser("list-attempts")
    p_la.add_argument("--run-id", required=True)
    p_la.set_defaults(func=_cmd_list_attempts)

    p_ba = sub.add_parser("best-attempt")
    p_ba.add_argument("--run-id", required=True)
    p_ba.set_defaults(func=_cmd_best_attempt)

    p_rt = sub.add_parser("retrain")
    p_rt.add_argument("--run-id", required=True)
    p_rt.set_defaults(func=_cmd_retrain)

    rr = runs_root()
    rr.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(log_path=rr / "engine_smoke.log", verbose=False)

    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    if args.hello:
        logger.info("✅ surrogate_tool engine is alive.")
        logger.info(f"Version: {__version__}")
        logger.info(f"Python: {os.sys.version.split()[0]}")
        logger.info(f"Runs root: {rr}")
        print("HELLO_OK")
        return 0

    if hasattr(args, "func"):
        return int(args.func(args, logger))

    parser.print_help()
    return 0
