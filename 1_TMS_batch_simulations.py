#!/usr/bin/env python3
"""
Lorina SINANAJ, UNIGE, April 2026
TMS E-field simulations pipeline

"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys
import yaml

import pandas as pd


class PipelineError(ValueError):
    """Actionable configuration/input error."""


def _resolve_path(value, base):
    if value is None:
        return None
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path):
    """Load only the shared settings Stage 1 needs; Stage 2 validates the full file."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise PipelineError(f"Configuration not found: {path}. Copy config.example.yaml to config.yaml first.")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise PipelineError("Configuration must be a YAML mapping.")

    input_cfg = dict(raw.get("input") or {})
    sim_cfg = dict(raw.get("simulation") or {})
    input_cfg.setdefault("study_table", "studies.csv")
    input_cfg.setdefault("simulation_root", "simulations")
    input_cfg.setdefault("folder_template", "{Study_ID}_study")

    defaults = {
        "head_model_path": None,
        "coil_models_root": None,
        "didtmax_lookup_csv": None,
        "open_in_gmsh": False,
        "dry_run": True,
        "run_only_studies": None,
        "output_folder_suffix": "_study",
        "use_distance_from_csv": True,
        "default_distance_mm": 0.0,
        "preferred_eeg_positions_file": "EEG10-10_UI_Jurak_2007.csv",
    }
    for key, value in defaults.items():
        sim_cfg.setdefault(key, value)

    base = path.parent
    input_cfg["study_table"] = _resolve_path(input_cfg["study_table"], base)
    input_cfg["simulation_root"] = _resolve_path(input_cfg["simulation_root"], base)
    for key in ("head_model_path", "coil_models_root", "didtmax_lookup_csv"):
        sim_cfg[key] = _resolve_path(sim_cfg[key], base)

    return {"input": input_cfg, "simulation": sim_cfg, "config_path": path}


REQUIRED_COLUMNS = {
    "Study_ID",
    "Study_name",
    "region",
    "targeting_method",
    "coil_mt",
    "coil_center",
    "coil_name",
}

LEGACY_EEG_LABELS = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}


def normalize_text(value):
    return str(value).strip().lower()


def normalize_eeg_label(label):
    normalized = str(label).strip().upper()
    normalized = normalized.replace(" ", "").replace("-", "").replace("_", "")
    for old_label, new_label in LEGACY_EEG_LABELS.items():
        normalized = normalized.replace(old_label, new_label)
    return normalized


def parse_vector(value, column_name, study_id):
    try:
        parsed = ast.literal_eval(str(value))
        if not isinstance(parsed, (list, tuple)) or len(parsed) != 3:
            raise ValueError
        return [float(x) for x in parsed]
    except Exception as exc:
        raise ValueError(
            f"Invalid {column_name} for Study_ID={study_id}: {value}"
        ) from exc


def validate_dataframe(df, config):
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"CSV is missing required simulation columns: {sorted(missing_columns)}"
        )

    sim = config["simulation"]
    if sim["use_distance_from_csv"] and "distance" not in df.columns:
        raise ValueError(
            "CSV must contain a 'distance' column (in mm) when "
            "simulation.use_distance_from_csv = true."
        )

    for key, label in (
        ("coil_models_root", "Coil models root folder"),
        ("didtmax_lookup_csv", "dI/dtmax lookup CSV"),
        ("head_model_path", "Head model path"),
    ):
        path = sim[key]
        if path is None:
            raise ValueError(f"simulation.{key} is not configured in config.yaml.")
        if not path.exists():
            raise ValueError(f"{label} does not exist: {path}")

    if not sim["coil_models_root"].is_dir():
        raise ValueError(
            f"Coil models root is not a folder: {sim['coil_models_root']}"
        )
    if not sim["head_model_path"].is_dir():
        raise ValueError(
            f"Head model path is not a folder: {sim['head_model_path']}"
        )
    if not sim["didtmax_lookup_csv"].is_file():
        raise ValueError(
            f"dI/dtmax lookup is not a file: {sim['didtmax_lookup_csv']}"
        )

    expected_folder_template = "{Study_ID}" + sim["output_folder_suffix"]
    if config["input"]["folder_template"] != expected_folder_template:
        raise ValueError(
            "Simulation/analysis folder conventions differ. Set "
            f"input.folder_template to {expected_folder_template!r} so Stage 2 can find Stage 1 outputs."
        )


def find_column(columns, candidates):
    normalized_to_original = {
        str(column).strip().lower(): column for column in columns
    }
    for candidate in candidates:
        if candidate in normalized_to_original:
            return normalized_to_original[candidate]
    return None


# ========== dI/dtmax lookup functions ==========
def load_didtmax_lookup_dataframe(path):
    lookup_df = pd.read_csv(path)
    lookup_df.columns = [str(column).strip() for column in lookup_df.columns]

    coil_name_column = find_column(lookup_df.columns, ["coil_name", "coil", "name"])
    didtmax_column = find_column(
        lookup_df.columns, ["coil_didtmax", "didtmax", "didt_max", "di_dt_max"]
    )

    if coil_name_column is None or didtmax_column is None:
        raise ValueError(
            "Lookup CSV must contain coil name and dI/dtmax columns. "
            f"Found columns: {list(lookup_df.columns)}"
        )

    working_df = lookup_df[[coil_name_column, didtmax_column]].copy()
    working_df.columns = ["coil_name", "coil_didtmax"]
    working_df["coil_name"] = working_df["coil_name"].astype(str).str.strip()
    return working_df


def get_coil_didtmax(row, didtmax_lookup_df):
    coil_name = str(row["coil_name"]).strip()
    matches = didtmax_lookup_df[didtmax_lookup_df["coil_name"] == coil_name]

    if matches.empty:
        raise ValueError(
            f"coil_didtmax not found in lookup CSV for Study_ID={row['Study_ID']}: {coil_name}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple coil_didtmax matches found in lookup CSV for Study_ID={row['Study_ID']}: {coil_name}"
        )
    return float(matches.iloc[0]["coil_didtmax"])


def get_eeg_positions_file(head_model_path, study_id, preferred_file_name):
    eeg_positions_dir = head_model_path / "eeg_positions"
    if not eeg_positions_dir.exists() or not eeg_positions_dir.is_dir():
        raise ValueError(
            f"Missing eeg_positions folder for Study_ID={study_id}: {eeg_positions_dir}"
        )

    preferred_file = eeg_positions_dir / preferred_file_name
    if preferred_file.exists():
        return preferred_file

    eeg_10_10_matches = sorted(eeg_positions_dir.glob("EEG10-10*.csv"))
    if eeg_10_10_matches:
        return eeg_10_10_matches[0]

    csv_matches = sorted(eeg_positions_dir.glob("*.csv"))
    if len(csv_matches) == 1:
        return csv_matches[0]
    if len(csv_matches) == 0:
        raise ValueError(
            f"No EEG positions CSV file found for Study_ID={study_id} in {eeg_positions_dir}"
        )

    match_list = "\n".join(str(path) for path in csv_matches)
    raise ValueError(
        "Multiple EEG positions CSV files found for "
        f"Study_ID={study_id}. Set simulation.preferred_eeg_positions_file "
        f"to the correct file.\n{match_list}"
    )


def load_eeg_positions_dataframe(head_model_path, study_id, preferred_file_name):
    eeg_positions_file = get_eeg_positions_file(
        head_model_path, study_id, preferred_file_name
    )

    raw_df = pd.read_csv(
        eeg_positions_file,
        sep=None,
        engine="python",
        header=None,
    )

    if raw_df.empty:
        raise ValueError(
            f"EEG positions CSV is empty for Study_ID={study_id}: {eeg_positions_file}"
        )

    first_cell = str(raw_df.iloc[0, 0]).strip().lower()

    if first_cell in {"electrode", "fiducial", "referenceelectrode"}:
        if raw_df.shape[1] < 5:
            raise ValueError(
                f"EEG positions CSV has too few columns for Study_ID={study_id}: {eeg_positions_file}"
            )
        working_df = raw_df.iloc[:, :5].copy()
        working_df.columns = ["point_type", "x", "y", "z", "label"]
        return working_df

    header_df = pd.read_csv(eeg_positions_file, sep=None, engine="python")
    header_df.columns = [str(column).strip() for column in header_df.columns]
    return header_df


def get_eeg_coordinate_from_label(label, head_model_path, study_id, preferred_file_name):
    eeg_df = load_eeg_positions_dataframe(
        head_model_path, study_id, preferred_file_name
    )

    label_column = find_column(
        eeg_df.columns,
        ["label", "name", "electrode", "position", "channel"],
    )
    if label_column is None:
        if "point_type" in eeg_df.columns and eeg_df.shape[1] >= 5:
            label_column = eeg_df.columns[4]
        else:
            label_column = eeg_df.columns[0]

    x_column = find_column(eeg_df.columns, ["x"])
    y_column = find_column(eeg_df.columns, ["y"])
    z_column = find_column(eeg_df.columns, ["z"])

    if x_column is None or y_column is None or z_column is None:
        if "point_type" in eeg_df.columns and len(eeg_df.columns) >= 5:
            x_column = eeg_df.columns[1]
            y_column = eeg_df.columns[2]
            z_column = eeg_df.columns[3]
        elif len(eeg_df.columns) >= 4:
            x_column = eeg_df.columns[1]
            y_column = eeg_df.columns[2]
            z_column = eeg_df.columns[3]
        else:
            raise ValueError(
                f"Could not find X/Y/Z columns in EEG positions CSV for Study_ID={study_id}."
            )

    working_df = eeg_df[[label_column, x_column, y_column, z_column]].copy()
    working_df.columns = ["label", "x", "y", "z"]
    working_df["label_norm"] = working_df["label"].astype(str).apply(normalize_eeg_label)
    working_df = working_df[working_df["label"].notna()].copy()
    working_df = working_df[working_df["label"].astype(str).str.strip() != ""].copy()

    query_label = normalize_eeg_label(label)
    exact_match = working_df[working_df["label_norm"] == query_label]
    if not exact_match.empty:
        match_row = exact_match.iloc[0]
        return [
            float(match_row["x"]),
            float(match_row["y"]),
            float(match_row["z"]),
        ]

    coordinates_by_label = {
        row["label_norm"]: [float(row["x"]), float(row["y"]), float(row["z"])]
        for _, row in working_df.iterrows()
    }

    label_keys = list(coordinates_by_label.keys())
    for first_label in label_keys:
        for second_label in label_keys:
            if first_label == second_label:
                continue
            if query_label == f"{first_label}{second_label}":
                first_xyz = coordinates_by_label[first_label]
                second_xyz = coordinates_by_label[second_label]
                return [
                    (first_xyz[0] + second_xyz[0]) / 2.0,
                    (first_xyz[1] + second_xyz[1]) / 2.0,
                    (first_xyz[2] + second_xyz[2]) / 2.0,
                ]

    available_labels = sorted(set(working_df["label"].astype(str).tolist()))
    raise ValueError(
        f"EEG label not found for Study_ID={study_id}: {label}. "
        f"Available labels include: {available_labels[:20]}"
    )


def get_center_coordinate(row, head_model_path, preferred_file_name):
    targeting_method = normalize_text(row["targeting_method"])
    study_id = row["Study_ID"]

    if targeting_method == "eeg":
        return get_eeg_coordinate_from_label(
            row["coil_center"], head_model_path, study_id, preferred_file_name
        )
    if targeting_method in {
        "xyz",
        "coord",
        "coords",
        "coordinate",
        "coordinates",
        "vector",
    }:
        return parse_vector(row["coil_center"], "coil_center", study_id)
    raise ValueError(
        f"Unsupported targeting_method for Study_ID={study_id}: "
        f"{row['targeting_method']}. Use EEG or XYZ."
    )


def get_auto_ydir(center_coordinate, region, study_id):
    region_normalized = normalize_text(region)
    if region_normalized == "":
        raise ValueError(f"Missing region for Study_ID={study_id}.")

    x_coord, y_coord, z_coord = center_coordinate
    if "vermis" in region_normalized or "cb" in region_normalized:
        return [x_coord, y_coord, z_coord - 10.0]
    if (
        "dlpfc" in region_normalized
        or "pfc" in region_normalized
        or "temporoparietal" in region_normalized
        or "tpc" in region_normalized
    ):
        return [x_coord, y_coord + 10.0, z_coord]
    raise ValueError(
        f"Unsupported region for Study_ID={study_id}: {region}. "
        "Supported presets are PFC/DLPFC, temporoparietal/TPC, and vermis/CB."
    )


def get_coil_path(row, coil_models_root):
    coil_name = row["coil_name"]
    if pd.isna(coil_name) or str(coil_name).strip() == "":
        raise ValueError(f"Missing coil_name for Study_ID={row['Study_ID']}.")

    coil_name = str(coil_name).strip()
    matches = sorted(coil_models_root.rglob(coil_name))
    if len(matches) == 0:
        raise ValueError(
            f"Coil file not found for Study_ID={row['Study_ID']}: "
            f"{coil_name} under {coil_models_root}"
        )
    if len(matches) > 1:
        match_list = "\n".join(str(path) for path in matches)
        raise ValueError(
            f"Multiple coil files found for Study_ID={row['Study_ID']}: "
            f"{coil_name}\n{match_list}"
        )
    return str(matches[0])


def get_distance_mm(row, use_distance_from_csv, default_distance_mm):
    if use_distance_from_csv:
        if pd.isna(row["distance"]):
            raise ValueError(f"Missing distance for Study_ID={row['Study_ID']}.")
        distance_mm = float(row["distance"])
    else:
        distance_mm = float(default_distance_mm)

    if distance_mm < 0:
        raise ValueError(
            f"Distance must be >= 0 mm for Study_ID={row['Study_ID']}."
        )
    return distance_mm


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1: run SimNIBS TMS simulations from the shared study CSV. "
            "Paths/settings come from the same config.yaml used by Stage 2."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate/print planned simulations without running SimNIBS (overrides YAML).",
    )
    parser.add_argument(
        "--run-only",
        nargs="+",
        help="Optional Study_ID values to run; overrides simulation.run_only_studies.",
    )
    args = parser.parse_args(argv)

    try:
        try:
            from simnibs import sim_struct, run_simnibs
        except ImportError as exc:
            raise PipelineError(
                "SimNIBS is not importable in this interpreter. Run Stage 1 with the same "
                "SimNIBS interpreter that passes `2_TMS_efield_analysis.py --doctor`."
            ) from exc

        config = load_config(args.config)
        sim = config["simulation"]
        study_table = config["input"]["study_table"]
        output_root = config["input"]["simulation_root"]

        df = pd.read_csv(study_table, dtype={"Study_ID": str})
        df["Study_ID"] = df["Study_ID"].astype(str).str.strip()
        validate_dataframe(df, config)
        didtmax_lookup_df = load_didtmax_lookup_dataframe(sim["didtmax_lookup_csv"])

        selected_ids = args.run_only if args.run_only is not None else sim["run_only_studies"]
        if selected_ids is not None:
            selected_ids = {str(v).strip() for v in selected_ids}
            df = df[df["Study_ID"].isin(selected_ids)].copy()
            missing = selected_ids - set(df["Study_ID"])
            if missing:
                raise ValueError(
                    f"Requested Study_ID values were not found in the study CSV: {sorted(missing)}"
                )

        if df.empty:
            raise ValueError("No studies selected. Check simulation.run_only_studies/--run-only.")

        output_root.mkdir(parents=True, exist_ok=True)
        dry_run = bool(sim["dry_run"] or args.dry_run)

        print(f"Loaded {len(df)} study rows from: {study_table}")
        print(f"Head model path: {sim['head_model_path']}")
        print(f"Output root: {output_root}")
        print(f"Coil models root: {sim['coil_models_root']}")
        print(f"dI/dtmax lookup CSV: {sim['didtmax_lookup_csv']}")
        print(f"Dry run: {dry_run}")
        print("----------------------------------------")

        for _, row in df.iterrows():
            study_id = str(row["Study_ID"]).strip()
            study_name = str(row.get("Study_name", f"Study_{study_id}")).strip()

            session = sim_struct.SESSION()
            head_model_path = sim["head_model_path"]
            session.subpath = str(head_model_path)
            session.open_in_gmsh = sim["open_in_gmsh"]

            output_dir = output_root / f"{study_id}{sim['output_folder_suffix']}"
            output_dir.mkdir(parents=True, exist_ok=True)
            session.pathfem = str(output_dir)

            tms_list = sim_struct.TMSLIST()
            coil_path = get_coil_path(row, sim["coil_models_root"])
            coil_label = str(row["coil_name"]).strip()
            tms_list.fnamecoil = coil_path

            position = tms_list.add_position()
            coil_didtmax = get_coil_didtmax(row, didtmax_lookup_df)
            position.didt = float(row["coil_mt"]) * coil_didtmax

            center_coordinate = get_center_coordinate(
                row,
                head_model_path,
                sim["preferred_eeg_positions_file"],
            )
            ydir_coordinate = get_auto_ydir(
                center_coordinate, row["region"], study_id
            )

            position.centre = center_coordinate
            position.pos_ydir = ydir_coordinate
            position.distance = get_distance_mm(
                row,
                sim["use_distance_from_csv"],
                sim["default_distance_mm"],
            )

            session.add_tmslist(tms_list)

            print(
                f"[{study_id}] {study_name} | "
                f"targeting_method={row['targeting_method']} | "
                f"region={row['region']} | "
                f"coil={coil_label} | "
                f"coil_didtmax={coil_didtmax} | "
                f"distance={position.distance} mm | "
                f"head_model={head_model_path.name} | "
                f"centre={position.centre} | "
                f"ydir={position.pos_ydir} | "
                f"output={output_dir}"
            )

            if not dry_run:
                run_simnibs(session)

        print("----------------------------------------")
        if dry_run:
            print("Dry run complete. No TMS simulations were launched.")
        else:
            print("Simulation stage complete.")
            print(
                "Next: validate Stage 2 with "
                f"{sys.executable} 2_TMS_efield_analysis.py --validate-only --config {config['config_path']}"
            )
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except (PipelineError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
