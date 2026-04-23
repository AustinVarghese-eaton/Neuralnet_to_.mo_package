from __future__ import annotations

from surrogate_tool.pipeline.preprocess import run_preprocess
from surrogate_tool.pipeline.eda import run_eda
from surrogate_tool.pipeline.split_scale import run_split_scale
from surrogate_tool.pipeline.train import run_training
from surrogate_tool.pipeline.modelica_export import export_modelica
from surrogate_tool.pipeline.report import generate_report


def run_full_attempt(run_id: str, attempt_num: int) -> dict:
    """
    Run the full pipeline for an attempt in deterministic order.
    Assumes run_config.json exists and includes input/output columns.
    """
    # preprocess uses cfg input/output (or overrides via CLI earlier). Here we do not override.
    run_preprocess(run_id=run_id, attempt_num=attempt_num, inputs_override=None, outputs_override=None)

    run_eda(run_id=run_id, attempt_num=attempt_num)

    run_split_scale(run_id=run_id, attempt_num=attempt_num)

    run_training(run_id=run_id, attempt_num=attempt_num)

    export_modelica(run_id=run_id, attempt_num=attempt_num)

    rep = generate_report(run_id=run_id, attempt_num=attempt_num)

    return {"report": rep}
