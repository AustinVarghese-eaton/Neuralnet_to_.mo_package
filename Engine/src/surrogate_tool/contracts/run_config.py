from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


ModelicaName = str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_modelica_name(name: str) -> str:
    """
    Modelica identifier rule requested:
      ^[A-Za-z][A-Za-z0-9_]*$
    """
    import re
    if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", name or ""):
        raise ValueError(
            "Invalid Modelica package name. Must match ^[A-Za-z][A-Za-z0-9_]*$ "
            "(start with a letter; then letters/digits/underscore only)."
        )
    return name


class RunConfig(BaseModel):
    """
    File contract for Excel -> Engine input.
    This schema will expand in later checkpoints (attempts, hyperparams, etc.).
    """
    run_id: str = Field(..., description="Unique run workspace id (e.g., 20260203_0930_A1)")
    created_utc: str = Field(default_factory=utc_now_iso)

    dataset_path: str = Field(..., description="Original dataset file path (CSV/XLSX).")
    dataset_format: Literal["csv", "xlsx"] = Field(..., description="Dataset type.")
    sheet_name: Optional[str] = Field(default=None, description="Required for XLSX if not default.")

    # Engineer-selected I/O columns (Excel will populate these)
    input_columns: list[str] = Field(default_factory=list)
    output_columns: list[str] = Field(default_factory=list)

    # UX fields
    n_inputs: int = 0
    n_outputs: int = 0

    # Modelica export
    package_name: ModelicaName = Field(..., description="Top-level Modelica package name.")
    random_seed: int = 42
    engine_version: str = "0.1.0"

    # FMU compilation
    gcc_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to gcc executable for FMU DLL compilation. "
            r"Auto-detected from OpenModelica install (e.g. C:\OpenModelica1.24.0\tools\msys\mingw64\bin\gcc.exe). "
            "Leave None to use auto-detection."
        ),
    )

    @field_validator("package_name")
    @classmethod
    def _pkg_name(cls, v: str) -> str:
        return validate_modelica_name(v)

    @field_validator("dataset_path")
    @classmethod
    def _dataset_path(cls, v: str) -> str:
        p = Path(v)
        if not p.exists():
            raise ValueError(f"Dataset path does not exist: {v}")
        if p.suffix.lower() not in [".csv", ".xlsx"]:
            raise ValueError("Dataset must be .csv or .xlsx")
        return str(p)

    @field_validator("sheet_name")
    @classmethod
    def _sheet_name(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            return None
        return v

    def normalized(self) -> "RunConfig":
        """
        Return a normalized copy (format inferred from extension if needed).
        """
        p = Path(self.dataset_path)
        fmt = "csv" if p.suffix.lower() == ".csv" else "xlsx"
        data = self.model_dump()
        data["dataset_format"] = fmt
        return RunConfig(**data)


def load_run_config(path: Path) -> RunConfig:
    """
    Loads JSON that may contain UTF-8 BOM (common from Windows tools).
    """
    import json
    raw = path.read_text(encoding="utf-8-sig")
    obj = json.loads(raw)
    return RunConfig(**obj).normalized()


def save_run_config(cfg: RunConfig, path: Path) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.model_dump(), indent=2), encoding="utf-8")
