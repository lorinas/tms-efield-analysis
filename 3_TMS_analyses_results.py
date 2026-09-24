#!/usr/bin/env python3
"""
3_TMS_analyses_results.py

Analyses-facing synthesis for the outputs of 2_TMS_efield_analysis.py.

This script DOES NOT:
- run SimNIBS
- recompute rho
- run permutations
- perform cluster-level permutation inference
- choose visually "best" regions

It reads the completed element-wise analysis and produces reproducible descriptive
summaries of the element-wise uncorrected and max-statistic FWER results, including
sign-separated face-connected regions that were already defined by script 2.

Interpretation:
- positive rho: higher E-field is associated with larger effect sizes
- negative rho: lower E-field is associated with larger effect sizes

Coordinate descriptions assume the standard SimNIBS/RAS-style orientation used by
the analysis mesh: x<0 left, x>0 right, y<0 posterior, y>0 anterior. These are
geometric descriptions in the analyzed head-model coordinate space, not atlas labels.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ReportError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_element_results(results_dir: Path) -> pd.DataFrame:
    """Prefer NPZ (authoritative/compact); fall back to CSV."""
    npz_path = results_dir / "element_results.npz"
    csv_path = results_dir / "element_results.csv"

    if npz_path.is_file():
        z = np.load(npz_path, allow_pickle=False)
        required = {"element_id", "coordinates_mm", "volume_mm3", "rho"}
        missing = required - set(z.files)
        if missing:
            raise ReportError(f"{npz_path} is missing arrays: {sorted(missing)}")

        coords = np.asarray(z["coordinates_mm"], dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ReportError("coordinates_mm must have shape (n_elements, 3).")

        data: dict[str, np.ndarray] = {
            "element_id": np.asarray(z["element_id"]),
            "x_mm": coords[:, 0],
            "y_mm": coords[:, 1],
            "z_mm": coords[:, 2],
            "volume_mm3": np.asarray(z["volume_mm3"], dtype=float),
            "rho": np.asarray(z["rho"], dtype=float),
        }
        optional = [
            "valid_correlation",
            "p_uncorrected",
            "signed_neglog10_p_uncorrected",
            "p_fwer_maxstat",
            "signed_neglog10_p_fwer_maxstat",
            "significant_uncorrected",
            "region_id_uncorrected",
            "significant_fwer",
            "region_id_fwer",
        ]
        for name in optional:
            if name in z.files:
                data[name] = np.asarray(z[name])
        # Backward compatibility with result folders created before the terminology change.
        if "region_id_uncorrected" not in data and "component_id_uncorrected" in z.files:
            data["region_id_uncorrected"] = np.asarray(z["component_id_uncorrected"])
        if "region_id_fwer" not in data and "component_id_fwer" in z.files:
            data["region_id_fwer"] = np.asarray(z["component_id_fwer"])
        return pd.DataFrame(data)

    if csv_path.is_file():
        frame = pd.read_csv(csv_path)
        frame = frame.rename(columns={
            "component_id_uncorrected": "region_id_uncorrected",
            "component_id_fwer": "region_id_fwer",
        })
        return frame

    raise ReportError(
        f"Could not find element_results.npz or element_results.csv in {results_dir}"
    )


def weighted_centroid(df: pd.DataFrame) -> tuple[float, float, float]:
    w = df["volume_mm3"].to_numpy(float)
    xyz = df[["x_mm", "y_mm", "z_mm"]].to_numpy(float)
    if len(df) == 0:
        return (float("nan"),) * 3
    if np.isfinite(w).all() and w.sum() > 0:
        c = np.average(xyz, axis=0, weights=w)
    else:
        c = np.mean(xyz, axis=0)
    return tuple(float(v) for v in c)


def pct(condition: np.ndarray) -> float:
    return 100.0 * float(np.count_nonzero(condition)) / len(condition) if len(condition) else float("nan")


def geometric_location(df: pd.DataFrame, dominance: float = 0.70) -> tuple[str, dict[str, float]]:
    x = df["x_mm"].to_numpy(float)
    y = df["y_mm"].to_numpy(float)

    p_left = pct(x < 0)
    p_right = pct(x > 0)
    p_post = pct(y < 0)
    p_ant = pct(y > 0)

    if p_left >= 100 * dominance:
        lr = "left-sided"
        lr_dominant = True
    elif p_right >= 100 * dominance:
        lr = "right-sided"
        lr_dominant = True
    else:
        lr = "bilateral"
        lr_dominant = False

    if p_post >= 100 * dominance:
        ap = "posterior"
        ap_dominant = True
    elif p_ant >= 100 * dominance:
        ap = "anterior"
        ap_dominant = True
    else:
        ap = "spanning anterior and posterior regions"
        ap_dominant = False

    if ap_dominant and lr_dominant:
        phrase = f"predominantly {ap} and {lr}"
    elif ap_dominant:
        phrase = f"predominantly {ap} and bilateral"
    elif lr_dominant:
        phrase = f"spanning anterior and posterior regions with a {lr} predominance"
    else:
        phrase = "bilateral and spanning anterior and posterior regions"
    stats = {
        "pct_left_x_lt_0": p_left,
        "pct_right_x_gt_0": p_right,
        "pct_posterior_y_lt_0": p_post,
        "pct_anterior_y_gt_0": p_ant,
        "median_x_mm": float(np.nanmedian(x)),
        "median_y_mm": float(np.nanmedian(y)),
        "median_z_mm": float(np.nanmedian(df["z_mm"].to_numpy(float))),
    }
    return phrase, stats


def global_sign_summary(
    df: pd.DataFrame,
    p_col: str,
    inference_label: str,
    alpha: float,
    sign_value: int,
    dominance: float,
) -> dict[str, Any]:
    rho = df["rho"].to_numpy(float)
    p = df[p_col].to_numpy(float)
    valid = np.isfinite(rho) & np.isfinite(p)
    if sign_value > 0:
        mask = valid & (p <= alpha) & (rho > 0)
        sign_name = "positive"
    else:
        mask = valid & (p <= alpha) & (rho < 0)
        sign_name = "negative"

    selected = df.loc[mask].copy()
    total_valid = int(np.count_nonzero(valid))
    total_volume = float(df.loc[valid, "volume_mm3"].sum())

    base: dict[str, Any] = {
        "inference": inference_label,
        "sign": sign_name,
        "alpha": alpha,
        "n_elements": len(selected),
        "pct_valid_elements": 100.0 * len(selected) / total_valid if total_valid else np.nan,
        "volume_mm3": float(selected["volume_mm3"].sum()),
        "volume_cm3": float(selected["volume_mm3"].sum()) / 1000.0,
        "pct_valid_gm_volume": (
            100.0 * float(selected["volume_mm3"].sum()) / total_volume if total_volume else np.nan
        ),
    }

    if selected.empty:
        base.update({
            "mean_rho": np.nan,
            "median_rho": np.nan,
            "rho_min": np.nan,
            "rho_max": np.nan,
            "min_p_value": np.nan,
            "pct_left_x_lt_0": np.nan,
            "pct_right_x_gt_0": np.nan,
            "pct_posterior_y_lt_0": np.nan,
            "pct_anterior_y_gt_0": np.nan,
            "median_x_mm": np.nan,
            "median_y_mm": np.nan,
            "median_z_mm": np.nan,
            "centroid_x_mm": np.nan,
            "centroid_y_mm": np.nan,
            "centroid_z_mm": np.nan,
            "geometric_location": "no significant elements",
        })
        return base

    phrase, geom = geometric_location(selected, dominance)
    cx, cy, cz = weighted_centroid(selected)
    base.update({
        "mean_rho": float(selected["rho"].mean()),
        "median_rho": float(selected["rho"].median()),
        "rho_min": float(selected["rho"].min()),
        "rho_max": float(selected["rho"].max()),
        "min_p_value": float(selected[p_col].min()),
        **geom,
        "centroid_x_mm": cx,
        "centroid_y_mm": cy,
        "centroid_z_mm": cz,
        "geometric_location": phrase,
    })
    return base


def best_fwer_element(df: pd.DataFrame) -> dict[str, Any] | None:
    if "p_fwer_maxstat" not in df:
        return None
    valid = np.isfinite(df["p_fwer_maxstat"]) & np.isfinite(df["rho"])
    if not np.any(valid):
        return None
    sub = df.loc[valid]
    idx = sub["p_fwer_maxstat"].idxmin()
    row = df.loc[idx]
    return {
        "element_id": int(row["element_id"]),
        "p_fwer_maxstat": float(row["p_fwer_maxstat"]),
        "p_uncorrected": float(row["p_uncorrected"]) if "p_uncorrected" in row else np.nan,
        "rho": float(row["rho"]),
        "x_mm": float(row["x_mm"]),
        "y_mm": float(row["y_mm"]),
        "z_mm": float(row["z_mm"]),
    }


def region_report(
    df: pd.DataFrame,
    region_col: str,
    inference: str,
    alpha: float,
    dominance: float,
) -> pd.DataFrame:
    """Create a complete descriptive table for every connected region."""
    if region_col not in df:
        return pd.DataFrame()

    ids = pd.to_numeric(df[region_col], errors="coerce").fillna(0).astype(int)
    rows: list[dict[str, Any]] = []

    for region_id in sorted(i for i in ids.unique() if i > 0):
        sub = df.loc[ids == region_id].copy()
        if sub.empty:
            continue

        peak_idx = sub["rho"].abs().idxmax()
        peak = df.loc[peak_idx]
        phrase, geom = geometric_location(sub, dominance)
        cx, cy, cz = weighted_centroid(sub)
        sign = "positive" if float(sub["rho"].median()) > 0 else "negative"

        p_unc_min = float(sub["p_uncorrected"].min()) if "p_uncorrected" in sub else np.nan
        p_fwer_min = float(sub["p_fwer_maxstat"].min()) if "p_fwer_maxstat" in sub else np.nan
        p_unc_peak = float(peak["p_uncorrected"]) if "p_uncorrected" in peak else np.nan
        p_fwer_peak = float(peak["p_fwer_maxstat"]) if "p_fwer_maxstat" in peak else np.nan

        n_fwer = (
            int(np.count_nonzero(sub["p_fwer_maxstat"].to_numpy(float) <= alpha))
            if "p_fwer_maxstat" in sub
            else 0
        )

        rows.append({
            "inference": inference,
            "region_id": region_id,
            "sign": sign,
            "n_elements": int(len(sub)),
            "volume_mm3": float(sub["volume_mm3"].sum()),
            "volume_cm3": float(sub["volume_mm3"].sum()) / 1000.0,
            "mean_rho": float(sub["rho"].mean()),
            "median_rho": float(sub["rho"].median()),
            "volume_weighted_mean_rho": float(np.average(sub["rho"], weights=sub["volume_mm3"])),
            "peak_element_id": int(peak["element_id"]),
            "peak_rho": float(peak["rho"]),
            "peak_p_uncorrected": p_unc_peak,
            "peak_p_fwer_maxstat": p_fwer_peak,
            "min_p_uncorrected": p_unc_min,
            "min_p_fwer_maxstat": p_fwer_min,
            "n_elements_surviving_fwer": n_fwer,
            "pct_elements_surviving_fwer": 100.0 * n_fwer / len(sub),
            "peak_x_mm": float(peak["x_mm"]),
            "peak_y_mm": float(peak["y_mm"]),
            "peak_z_mm": float(peak["z_mm"]),
            "centroid_x_mm": cx,
            "centroid_y_mm": cy,
            "centroid_z_mm": cz,
            **geom,
            "geometric_location": phrase,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Rank regions by physical volume separately for positive and negative associations.
    sign_order = pd.Categorical(out["sign"], categories=["positive", "negative"], ordered=True)
    out = (
        out.assign(_sign_order=sign_order)
        .sort_values(["_sign_order", "volume_mm3", "region_id"], ascending=[True, False, True], kind="stable")
        .drop(columns="_sign_order")
        .reset_index(drop=True)
    )

    out["rank_by_volume_within_sign"] = (
        out.groupby("sign", sort=False).cumcount() + 1
    ).astype(int)
    sign_total = out.groupby("sign")["volume_mm3"].transform("sum")
    out["pct_same_sign_significant_volume"] = np.where(
        sign_total > 0, 100.0 * out["volume_mm3"] / sign_total, np.nan
    )
    out["cumulative_pct_same_sign_volume"] = (
        out.groupby("sign", sort=False)["pct_same_sign_significant_volume"].cumsum()
    )
    return out


def select_reported_regions(
    regions: pd.DataFrame,
    min_volume_cm3: float,
) -> pd.DataFrame:
    """
    Select regions for the compact HTML report using a transparent descriptive
    minimum physical-volume threshold.

    This is NOT a statistical significance criterion. All regions remain in the
    complete CSV files; the threshold only controls which regions appear in the
    compact report.
    """
    if regions.empty:
        return regions.copy()

    min_volume_mm3 = float(min_volume_cm3) * 1000.0
    selected = regions.loc[regions["volume_mm3"] >= min_volume_mm3].copy()
    if selected.empty:
        return selected

    sign_order = pd.Categorical(
        selected["sign"],
        categories=["positive", "negative"],
        ordered=True,
    )
    selected = (
        selected.assign(_sign_order=sign_order)
        .sort_values(
            ["_sign_order", "volume_mm3", "region_id"],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop(columns="_sign_order")
        .reset_index(drop=True)
    )
    return selected

def fmt_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def fmt_count(value: int | float) -> str:
    """Swiss-style thousands separator for human-facing integer counts."""
    return f"{int(value):,}".replace(",", "'")


def compact_location(text: str) -> str:
    return str(text).replace("predominantly ", "").replace(
        "spanning anterior and posterior regions with a ", ""
    )


def html_table(headers: list[str], rows: list[list[str]], *, numeric: set[int] | None = None) -> str:
    numeric = numeric or set()
    head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body_rows = []
    for row in rows:
        cells = []
        for i, value in enumerate(row):
            cls = ' class="num"' if i in numeric else ""
            cells.append(f"<td{cls}>{escape(str(value))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>'



def read_msh_view_names(mesh_path: Path) -> list[str]:
    """
    Read post-processing view names from a Gmsh MSH2 binary/ascii file without
    loading the mesh numerically. String-tag headers remain ASCII even when the
    mesh/data payload is binary.
    """
    data = mesh_path.read_bytes()
    pattern = re.compile(
        rb"\$(?:ElementData|NodeData)\r?\n\d+\r?\n\"([^\"]+)\""
    )
    return [m.group(1).decode("utf-8", errors="replace") for m in pattern.finditer(data)]


def find_gmsh_executable(explicit: str | None = None) -> Path | None:
    """
    Locate a Gmsh executable. Prefer an explicitly supplied path, then PATH,
    then likely SimNIBS/macOS locations relative to this Python interpreter.
    """
    candidates: list[Path] = []

    if explicit:
        candidates.append(Path(explicit).expanduser())

    env_value = os.environ.get("GMSH_EXECUTABLE")
    if env_value:
        candidates.append(Path(env_value).expanduser())

    which = shutil.which("gmsh")
    if which:
        candidates.append(Path(which))

    py = Path(sys.executable).resolve()
    env_bin = py.parent
    env_root = env_bin.parent
    simnibs_root = env_root.parent

    candidates.extend([
        env_bin / "gmsh",
        env_root / "bin" / "gmsh",
        simnibs_root / "gmsh",
        simnibs_root / "bin" / "gmsh",
        simnibs_root / "Gmsh.app" / "Contents" / "MacOS" / "gmsh",
        Path.home() / "Applications" / "Gmsh.app" / "Contents" / "MacOS" / "gmsh",
        Path("/Applications/Gmsh.app/Contents/MacOS/gmsh"),
    ])

    for candidate in candidates:
        try:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate.resolve()
        except OSError:
            pass

    if simnibs_root.is_dir():
        for pattern in ("**/gmsh", "**/Gmsh.app/Contents/MacOS/gmsh"):
            try:
                for candidate in simnibs_root.glob(pattern):
                    if candidate.is_file() and os.access(candidate, os.X_OK):
                        return candidate.resolve()
            except OSError:
                pass

    return None


def _geo_quote(path: Path) -> str:
    return str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"')


def render_gmsh_figure(
    mesh_path: Path,
    field_name: str,
    output_png: Path,
    *,
    value_min: float,
    value_max: float,
    discrete: bool,
    show_scale: bool,
    width: int,
    height: int,
    gmsh_executable: str | None = None,
) -> tuple[bool, str]:
    """
    Render one field directly from the exact association_maps.msh using the Gmsh
    executable. This deliberately does not require the Python `gmsh` module.
    """
    gmsh_bin = find_gmsh_executable(gmsh_executable)
    if gmsh_bin is None:
        return (
            False,
            "Could not find a Gmsh executable. Set --gmsh-executable to the Gmsh "
            "binary bundled with SimNIBS, or set GMSH_EXECUTABLE.",
        )

    view_names = read_msh_view_names(mesh_path)
    if field_name not in view_names:
        available = ", ".join(view_names) if view_names else "(none detected)"
        return False, (
            f"Field {field_name!r} not found in {mesh_path.name}. "
            f"Detected views: {available}"
        )
    target_index = view_names.index(field_name)

    visibility = "\n".join(
        f"View[{i}].Visible = {1 if i == target_index else 0};"
        for i in range(len(view_names))
    )

    if discrete:
        palette = (
            f"View[{target_index}].ColorTable = {{ Blue, White, Red }};\n"
            f"View[{target_index}].IntervalsType = 3;\n"
            f"View[{target_index}].NbIso = 3;"
        )
    else:
        palette = (
            f"View[{target_index}].ColormapNumber = 10;\n"
            f"View[{target_index}].IntervalsType = 2;\n"
            f"View[{target_index}].NbIso = 20;"
        )

    geo = f"""
Merge "{_geo_quote(mesh_path)}";

General.Trackball = 0;
General.Orthographic = 1;
General.Axes = 0;
General.SmallAxes = 0;
Mesh.Lines = 0;
Mesh.SurfaceEdges = 0;
Mesh.VolumeEdges = 0;
Print.Width = {int(width)};
Print.Height = {int(height)};
Print.Background = 1;

{visibility}

View[{target_index}].RangeType = 2;
View[{target_index}].CustomMin = {float(value_min):.17g};
View[{target_index}].CustomMax = {float(value_max):.17g};
View[{target_index}].SaturateValues = 1;
View[{target_index}].ScaleType = 1;
View[{target_index}].ShowScale = {1 if show_scale else 0};
View[{target_index}].ShowElement = 0;
View[{target_index}].SmoothNormals = 1;
View[{target_index}].Light = 1;
{palette}

Print "{_geo_quote(output_png)}";
Exit;
"""

    output_png.parent.mkdir(parents=True, exist_ok=True)
    if output_png.exists():
        output_png.unlink()

    script_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".geo",
            prefix="tms_report_",
            dir=str(output_png.parent),
            encoding="utf-8",
            delete=False,
        ) as tmp:
            tmp.write(geo)
            script_path = Path(tmp.name)

        proc = subprocess.run(
            [str(gmsh_bin), str(script_path)],
            cwd=str(output_png.parent),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=180,
            check=False,
        )

        if proc.returncode != 0:
            tail = (proc.stdout or "").strip()[-1600:]
            return False, f"Gmsh exited with code {proc.returncode}. {tail}"

        if not output_png.is_file() or output_png.stat().st_size == 0:
            tail = (proc.stdout or "").strip()[-1600:]
            return False, (
                f"Gmsh ran but did not create {output_png.name}. "
                f"Executable: {gmsh_bin}. {tail}"
            )

        return True, f"rendered directly from association_maps.msh using {gmsh_bin}"
    except subprocess.TimeoutExpired:
        return False, f"Gmsh rendering timed out after 180 seconds ({gmsh_bin})."
    except Exception as exc:
        return False, str(exc)
    finally:
        if script_path is not None:
            try:
                script_path.unlink(missing_ok=True)
            except OSError:
                pass


def render_report_figures(
    results_dir: Path,
    df: pd.DataFrame,
    *,
    width: int,
    height: int,
    gmsh_executable: str | None = None,
) -> dict[str, dict[str, str | bool]]:
    mesh_path = results_dir / "association_maps.msh"
    if not mesh_path.is_file():
        message = f"association_maps.msh not found in {results_dir}"
        return {
            "inverted": {"ok": False, "message": message, "file": "analyses_invertedLogP_uncorrected.png"},
            "significant": {"ok": False, "message": message, "file": "analyses_significant_uncorrected.png"},
        }

    signed = df["signed_neglog10_p_uncorrected"].to_numpy(float)
    finite = signed[np.isfinite(signed)]
    max_abs = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    max_abs = max(max_abs, 1.0)

    inverted_file = results_dir / "analyses_invertedLogP_uncorrected.png"
    ok1, msg1 = render_gmsh_figure(
        mesh_path,
        "invertedLogP_uncorrected",
        inverted_file,
        value_min=-max_abs,
        value_max=max_abs,
        discrete=False,
        show_scale=True,
        width=width,
        height=height,
        gmsh_executable=gmsh_executable,
    )

    significant_file = results_dir / "analyses_significant_uncorrected.png"
    ok2, msg2 = render_gmsh_figure(
        mesh_path,
        "significant_uncorrected",
        significant_file,
        value_min=-1.0,
        value_max=1.0,
        discrete=True,
        show_scale=False,
        width=width,
        height=height,
        gmsh_executable=gmsh_executable,
    )

    return {
        "inverted": {"ok": ok1, "message": msg1, "file": inverted_file.name},
        "significant": {"ok": ok2, "message": msg2, "file": significant_file.name},
    }


def figure_html(figure_info: dict[str, dict[str, str | bool]], alpha: float) -> str:
    if not figure_info:
        return ""
    cards: list[str] = []
    inverted = figure_info.get("inverted", {})
    if inverted.get("ok"):
        cards.append(
            f'''<figure class="figure-card">
<img src="{escape(str(inverted['file']))}" alt="Uncorrected association map">
<figcaption><strong>Uncorrected association map.</strong> Positive values follow positive association; negative values follow negative association.</figcaption>
</figure>'''
        )
    else:
        cards.append(
            f'''<div class="figure-card figure-missing"><strong>Uncorrected association map not rendered.</strong><br>{escape(str(inverted.get('message', 'Unknown rendering error')))}</div>'''
        )

    significant = figure_info.get("significant", {})
    if significant.get("ok"):
        cards.append(
            f'''<figure class="figure-card">
<img src="{escape(str(significant['file']))}" alt="GM elements with uncorrected permutation p-value at or below threshold">
<figcaption><strong>GM elements with uncorrected permutation p≤{alpha:g}.</strong> −1 for negative associations, 0 for nonsignificant elements and +1 for positive associations.</figcaption>
</figure>'''
        )
    else:
        cards.append(
            f'''<div class="figure-card figure-missing"><strong>Significance map not rendered.</strong><br>{escape(str(significant.get('message', 'Unknown rendering error')))}</div>'''
        )
    return '<div class="figures">' + ''.join(cards) + '</div>'

def analysis_source_info(results_dir: Path) -> dict[str, str]:
    """Read only the head-model ID recorded by Script 2."""
    info = {"head_model_id": "NA"}

    metadata = load_json(results_dir / "run_metadata.json")
    config = metadata.get("configuration", {}) if isinstance(metadata, dict) else {}
    if isinstance(config, dict):
        mesh_cfg = config.get("mesh", {}) or {}
        if mesh_cfg.get("head_model_id") not in (None, ""):
            info["head_model_id"] = str(mesh_cfg.get("head_model_id"))

    manifest_path = results_dir / "resolved_manifest.csv"
    if manifest_path.is_file() and info["head_model_id"] == "NA":
        try:
            manifest = pd.read_csv(manifest_path)
            if not manifest.empty and "head_model_id" in manifest.columns:
                value = manifest.iloc[0]["head_model_id"]
                if pd.notna(value):
                    info["head_model_id"] = str(value)
        except Exception:
            pass

    return info


def infer_cohort(results_dir: Path, summary: dict[str, Any]) -> str:
    """
    Infer cohort conservatively from the analysis name/path using whole tokens.
    Accepted labels are exactly CUD, SZ and HC. Otherwise return 'Unknown'.
    """
    candidates: list[str] = []

    for key in ("analysis_name", "project_name"):
        value = summary.get(key)
        if value not in (None, ""):
            candidates.append(str(value))

    candidates.extend(str(part) for part in results_dir.parts)

    for candidate in candidates:
        tokens = [
            token for token in re.split(r"[^A-Za-z0-9]+", candidate.upper())
            if token
        ]
        for cohort in ("CUD", "SZ", "HC"):
            if cohort in tokens:
                return cohort

    return "Unknown"

def build_html_report(
    results_dir: Path,
    summary: dict[str, Any],
    global_df: pd.DataFrame,
    unc_regions: pd.DataFrame,
    fwer_regions: pd.DataFrame,
    major_unc_regions: pd.DataFrame,
    major_fwer_regions: pd.DataFrame,
    best_fwer: dict[str, Any] | None,
    figure_info: dict[str, dict[str, str | bool]],
    alpha: float,
    min_region_volume_cm3: float,
) -> str:
    n_studies = summary.get("n_studies", "NA")
    n_perm = summary.get("n_permutations", "NA")
    n_elements = summary.get("n_elements", "NA")

    unc = global_df.loc[global_df["inference"] == "uncorrected permutation"]
    corr = global_df.loc[global_df["inference"] == "max-statistic FWER-corrected"]
    unc_total = int(unc["n_elements"].sum()) if not unc.empty else 0
    fwer_total = int(corr["n_elements"].sum()) if not corr.empty else 0
    min_fwer = best_fwer["p_fwer_maxstat"] if best_fwer is not None else np.nan

    overview_rows = [
        ["Studies", str(n_studies)],
        ["Cohort", str(summary.get("cohort", "Unknown"))],
        ["Head model ID", str(summary.get("head_model_id", "NA"))],
        ["GM elements tested", fmt_count(n_elements) if isinstance(n_elements, int) else str(n_elements)],
        ["Permutations", fmt_count(n_perm) if isinstance(n_perm, int) else str(n_perm)],
        ["Threshold", f"p ≤ {alpha:g}"],
        ["Elements with uncorrected p≤0.05", fmt_count(unc_total)],
        ["Elements with max-statistic FWER p≤0.05", fmt_count(fwer_total)],
        ["Minimum FWER p", fmt_p(min_fwer)],
    ]
    overview_table = html_table(["Metric", "Result"], overview_rows, numeric={1})

    def region_rows(regions: pd.DataFrame) -> list[list[str]]:
        rows: list[list[str]] = []
        for _, row in regions.iterrows():
            center = f"({row['centroid_x_mm']:.1f}, {row['centroid_y_mm']:.1f}, {row['centroid_z_mm']:.1f})"
            rows.append([
                str(int(row["region_id"])),
                "+" if row["sign"] == "positive" else "−",
                fmt_count(row["n_elements"]),
                f"{row['volume_cm3']:.3f}",
                compact_location(row["geometric_location"]),
                center,
                f"{row['peak_rho']:.3f}",
                fmt_p(row["peak_p_uncorrected"]),
                fmt_p(row["min_p_fwer_maxstat"]),
            ])
        return rows

    unc_rows = region_rows(major_unc_regions)
    if unc_rows:
        unc_table = html_table(
            ["Region", "Sign", "GM elements", "Volume (cm³)", "Location", "Center (x,y,z), mm", "Peak rho", "Peak p unc.", "Best p FWER"],
            unc_rows,
            numeric={0, 2, 3, 6, 7, 8},
        )
    else:
        unc_table = f"<p>No connected uncorrected regions had volume ≥{min_region_volume_cm3:g} cm³.</p>"

    fwer_rows = region_rows(major_fwer_regions)
    if fwer_rows:
        fwer_table = html_table(
            ["Region", "Sign", "GM elements", "Volume (cm³)", "Location", "Center (x,y,z), mm", "Peak rho", "Peak p unc.", "Best p FWER"],
            fwer_rows,
            numeric={0, 2, 3, 6, 7, 8},
        )
    else:
        fwer_table = f"<p>No connected regions were present because no GM elements survived max-statistic FWER p≤{alpha:g}.</p>"

    figures = figure_html(figure_info, alpha)
    figures_section = f"<h2>Spatial maps</h2>{figures}" if figures else ""

    region_note = (
        f"<p class='note'>Table shows connected regions with volume ≥{min_region_volume_cm3:g} cm³. "
        "Volume is the summed physical volume of the GM tetrahedra; GM elements is the number of tetrahedra. "
        "The center is the volume-weighted centroid of the GM tetrahedra in each region. "
        f"All {fmt_count(len(unc_regions))} regions remain in <code>analyses_regions_uncorrected.csv</code>.</p>"
    )

    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TMS E-field–Effect Size Analysis Results</title>
<style>
:root {{ color-scheme: light; --ink:#1f2933; --muted:#5b6670; --line:#d7dde3; --panel:#f7f9fb; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; color:var(--ink); background:white; line-height:1.45; }}
main {{ max-width:1180px; margin:0 auto; padding:36px 34px 60px; }}
h1 {{ font-size:28px; margin:0 0 8px; color:#132235; }}
h2 {{ font-size:19px; margin:34px 0 12px; color:#20364d; border-bottom:1px solid var(--line); padding-bottom:7px; }}
p {{ margin:8px 0 12px; }}
.table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th {{ background:#edf2f6; color:#24384c; font-weight:600; text-align:left; padding:9px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }}
td {{ padding:8px 10px; border-bottom:1px solid #e9edf1; vertical-align:top; }}
tbody tr:last-child td {{ border-bottom:none; }}
td.num {{ font-variant-numeric:tabular-nums; white-space:nowrap; }}
.figures {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; margin-top:12px; }}
.figure-card {{ margin:0; border:1px solid var(--line); border-radius:9px; overflow:hidden; background:var(--panel); }}
.figure-card img {{ display:block; width:100%; height:auto; background:white; }}
.figure-card figcaption, .figure-missing {{ padding:11px 13px; font-size:12.5px; color:#3f4d5a; }}
.figure-missing {{ min-height:110px; }}
.note {{ color:var(--muted); font-size:12.5px; }}
code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.92em; }}
@media (max-width:820px) {{ main {{ padding:24px 16px 40px; }} .figures {{ grid-template-columns:1fr; }} }}
@media print {{ main {{ max-width:none; padding:18px; }} .table-wrap {{ overflow:visible; }} }}
</style>
</head>
<body><main>
<h1>TMS E-field–Effect Size Analysis Results</h1>

<h2>Analysis overview</h2>
{overview_table}

{figures_section}

<h2>Reported uncorrected regions (≥{min_region_volume_cm3:g} cm³)</h2>
{unc_table}
{region_note}

<h2>FWER-significant regions</h2>
{fwer_table}

<h2>Reporting note</h2>
<p>Reported p-values are element-wise permutation p-values, not cluster-level p-values.</p>
</main></body></html>'''

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synthesize Script-2 TMS E-field results into a browser-ready analyses report."
    )
    parser.add_argument(
        "--results-dir", required=True, type=Path,
        help="Completed permutations_N directory from 2_TMS_efield_analysis.py",
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Override reporting alpha (default: use summary.json, otherwise 0.05).",
    )
    parser.add_argument(
        "--min-region-volume-cm3", type=float, default=1.0,
        help=(
            "Minimum physical region volume shown in the compact HTML report, in cm³ "
            "(default: 1.0). This is a descriptive reporting filter only. "
            "Use 0.001 for a 1 mm³ threshold."
        ),
    )
    parser.add_argument(
        "--dominance-threshold", type=float, default=0.70,
        help="Fraction required to call a region predominantly left/right or anterior/posterior.",
    )
    parser.add_argument(
        "--skip-figures", "--no-figures",
        dest="skip_figures",
        action="store_true",
        help="Generate tables and HTML without screenshots or a Spatial maps section.",
    )
    parser.add_argument("--image-width", type=int, default=1100)
    parser.add_argument("--image-height", type=int, default=850)
    parser.add_argument(
        "--gmsh-executable",
        type=str,
        default=None,
        help=(
            "Optional path to the Gmsh executable. Normally auto-detected from PATH "
            "or the active SimNIBS installation."
        ),
    )
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser().resolve()
    if not results_dir.is_dir():
        raise ReportError(f"Results directory not found: {results_dir}")

    summary = load_json(results_dir / "summary.json")
    summary.update(analysis_source_info(results_dir))
    summary["cohort"] = infer_cohort(results_dir, summary)
    alpha = float(args.alpha if args.alpha is not None else summary.get("significance_alpha", 0.05))
    if not (0 < alpha < 1):
        raise ReportError("alpha must be between 0 and 1.")
    if not (0.5 < args.dominance_threshold <= 1):
        raise ReportError("--dominance-threshold must be >0.5 and <=1.")
    if args.min_region_volume_cm3 < 0:
        raise ReportError("--min-region-volume-cm3 must be >= 0.")

    df = load_element_results(results_dir)
    required = {
        "element_id", "x_mm", "y_mm", "z_mm", "volume_mm3", "rho",
        "p_uncorrected", "signed_neglog10_p_uncorrected", "p_fwer_maxstat"
    }
    missing = required - set(df.columns)
    if missing:
        raise ReportError(
            f"Results are missing required columns/arrays: {sorted(missing)}. Run Script 2 with permutations first."
        )

    valid = np.isfinite(df["rho"].to_numpy(float))
    df = df.loc[valid].copy()

    global_rows = []
    for p_col, label in [
        ("p_uncorrected", "uncorrected permutation"),
        ("p_fwer_maxstat", "max-statistic FWER-corrected"),
    ]:
        for sign_value in (1, -1):
            global_rows.append(global_sign_summary(
                df, p_col, label, alpha, sign_value, args.dominance_threshold
            ))
    global_df = pd.DataFrame(global_rows)

    unc_regions = region_report(
        df, "region_id_uncorrected", "uncorrected permutation", alpha, args.dominance_threshold
    )
    fwer_regions = region_report(
        df, "region_id_fwer", "max-statistic FWER-corrected", alpha, args.dominance_threshold
    )
    major_unc_regions = select_reported_regions(unc_regions, args.min_region_volume_cm3)
    major_fwer_regions = select_reported_regions(fwer_regions, args.min_region_volume_cm3)
    best_fwer = best_fwer_element(df)

    global_df.to_csv(results_dir / "analyses_spatial_summary.csv", index=False)
    unc_regions.to_csv(results_dir / "analyses_regions_uncorrected.csv", index=False)
    fwer_regions.to_csv(results_dir / "analyses_regions_FWER.csv", index=False)
    major_unc_regions.to_csv(results_dir / "analyses_reported_regions.csv", index=False)

    if args.skip_figures:
        figure_info = {}
    else:
        figure_info = render_report_figures(
            results_dir,
            df,
            width=args.image_width,
            height=args.image_height,
            gmsh_executable=args.gmsh_executable,
        )
        for key, item in figure_info.items():
            status = "OK" if item.get("ok") else "WARNING"
            print(f"Figure {key}: {status} - {item.get('message')}")

    report = build_html_report(
        results_dir, summary, global_df, unc_regions, fwer_regions,
        major_unc_regions, major_fwer_regions, best_fwer, figure_info,
        alpha, args.min_region_volume_cm3,
    )
    (results_dir / "analyses_results.html").write_text(report, encoding="utf-8")

    metadata = {
        "alpha": alpha,
        "dominance_threshold": args.dominance_threshold,
        "min_region_volume_cm3": args.min_region_volume_cm3,
        "n_uncorrected_regions": int(len(unc_regions)),
        "n_fwer_regions": int(len(fwer_regions)),
        "n_reported_uncorrected_regions_in_summary": int(len(major_unc_regions)),
        "n_reported_fwer_regions_in_summary": int(len(major_fwer_regions)),
        "region_center_definition": "Volume-weighted centroid of GM tetrahedron centers, weighted by volume_mm3.",
        "figure_source": "Both screenshots are rendered directly from association_maps.msh using the original working Gmsh executable workflow.",
        "head_model_id": summary.get("head_model_id"),
        "cohort": summary.get("cohort"),
        "figures": figure_info,
        "interpretation": {
            "positive_rho": "higher E-field associated with larger effect sizes",
            "negative_rho": "lower E-field associated with larger effect sizes",
        },
        "important_limitation": (
            "Connected regions are descriptive groupings of element-wise significant tetrahedra; their p-values are element-wise, not cluster-level."
        ),
    }
    (results_dir / "analyses_results_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Analyses report written to: {results_dir}")
    print("  analyses_results.html")
    print("  analyses_spatial_summary.csv")
    print("  analyses_reported_regions.csv")
    print("  analyses_regions_uncorrected.csv")
    print("  analyses_regions_FWER.csv")
    print("  analyses_results_metadata.json")
    if not args.skip_figures:
        print("  analyses_invertedLogP_uncorrected.png")
        print("  analyses_significant_uncorrected.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

