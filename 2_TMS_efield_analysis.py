#!/usr/bin/env python3
"""
TMS E-field / effect-size analysis
=================================

Stage 2 of the repository. This single script replaces the old manual sequence of:
field extraction -> CSV combination -> Spearman correlations -> permutations -> map export.

Typical use:
    python 2_TMS_efield_analysis.py --config config.yaml

Useful checks:
    python 2_TMS_efield_analysis.py --validate-only --config config.yaml
    python 2_TMS_efield_analysis.py --demo --output work/demo --n-permutations 128

The script expects Stage 1 simulation folders named <Study_ID>_study by default.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import logging
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits



# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------
class PipelineError(ValueError):
    """An actionable validation error, rather than silent data substitution."""

def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Not JSON serializable: {type(value)}')

def canonical_json(value: Any) -> str:
    return json.dumps(value, default=json_default, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def code_fingerprint() -> str:
    return sha256_file(Path(__file__).resolve())

def versions() -> dict[str, Any]:
    result: dict[str, Any] = {'python': platform.python_version(), 'platform': platform.platform()}
    for name in ('numpy', 'scipy', 'pandas', 'PyYAML', 'threadpoolctl', 'simnibs'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result

def git_state() -> dict[str, Any]:
    try:
        cwd = Path(__file__).resolve().parent
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd, stderr=subprocess.DEVNULL, text=True).strip()
        status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=cwd, stderr=subprocess.DEVNULL, text=True)
        return {'commit': sha, 'dirty': bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {'commit': None, 'dirty': None}

def read_json(path: Path) -> Any:
    return json.loads(path.read_text())

def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    try:
        with temp.open('w') as f:
            json.dump(data, f, indent=2, sort_keys=True, default=json_default, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)

def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    try:
        with temp.open('wb') as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)

@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Never auto-remove another process's lock. Stale locks need human review."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 384)
    except FileExistsError as exc:
        raise PipelineError(f'Another run may be active: {path}. Inspect its host/PID. Only remove a stale lock after confirming that run has stopped.') from exc
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump({'pid': os.getpid(), 'host': socket.gethostname(), 'created_utc': timestamp()}, f)
        yield
    finally:
        path.unlink(missing_ok=True)

# -----------------------------------------------------------------------------
# INPUTS
# -----------------------------------------------------------------------------
class StrictSafeLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys rather than silently accepting the last value."""

def _unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise PipelineError(f'Duplicate YAML key: {key!r}')
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping
StrictSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)
DEFAULTS = {'project': {'analysis_name': 'analysis', 'cache_dir': 'work/cache', 'output_dir': 'results'}, 'input': {'study_table': 'studies.csv', 'simulation_root': 'simulations', 'manifest_csv': None, 'folder_template': '{Study_ID}_study', 'mesh_glob': '*_scalar.msh'}, 'columns': {'study_id': 'Study_ID', 'study_name': 'Study_name', 'effect_size': 'effect_size', 'include': None, 'outcome': None, 'independent_unit': None}, 'selection': {'outcome': None, 'study_ids': None}, 'simulation': {'head_model_path': None, 'coil_models_root': None, 'didtmax_lookup_csv': None, 'open_in_gmsh': False, 'dry_run': True, 'run_only_studies': None, 'output_folder_suffix': '_study', 'use_distance_from_csv': True, 'default_distance_mm': 0.0, 'preferred_eeg_positions_file': 'EEG10-10_UI_Jurak_2007.csv'}, 'mesh': {'head_model_id': 'head01', 'field_name': 'magnE', 'tissue_tag': 2, 'coordinate_tolerance_mm': 1e-06}, 'statistics': {'n_permutations': 10000, 'seed': 42, 'permutation_batch_size': 64, 'element_block_size': 8192, 'working_memory_mb': 512, 'blas_threads': 1, 'checkpoint_every': 256, 'independent_studies_confirmed': False, 'effect_size_definition': 'EDIT', 'positive_effect_means': 'EDIT'}, 'clusters': {'enabled': False, 'forming_p_approx': 0.01, 'rho_threshold': None, 'statistic': 'mass'}, 'export': {'csv': True, 'mesh': True, 'significance_alpha': 0.05}}

def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()

def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise PipelineError(f'Configuration not found: {path}. Copy config.example.yaml to config.yaml first.')
    raw = yaml.load(path.read_text(), Loader=StrictSafeLoader)
    if not isinstance(raw, dict):
        raise PipelineError('Configuration must be a YAML mapping.')
    config = deepcopy(DEFAULTS)
    unknown = set(raw) - set(config)
    if unknown:
        raise PipelineError(f'Unknown configuration sections: {sorted(unknown)}')
    for section, settings in raw.items():
        if not isinstance(settings, dict):
            raise PipelineError(f'{section} must be a YAML mapping.')
        unknown = set(settings) - set(config[section])
        if unknown:
            raise PipelineError(f'Unknown keys in {section}: {sorted(unknown)}')
        config[section].update(settings)
    for section, names in {'project': ['cache_dir', 'output_dir'], 'input': ['study_table', 'simulation_root', 'manifest_csv'], 'simulation': ['head_model_path', 'coil_models_root', 'didtmax_lookup_csv']}.items():
        for name in names:
            value = config[section][name]
            if value is not None:
                config[section][name] = resolve_path(value, path.parent)
    config['config_path'] = path
    validate_config(config)
    return config

def validate_config(c: dict[str, Any]) -> None:
    for key, value in [('analysis_name', c['project']['analysis_name']), ('head_model_id', c['mesh']['head_model_id'])]:
        if not isinstance(value, str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', value):
            raise PipelineError(f'{key} must contain only letters, digits, underscores, dots or hyphens.')
    for key in ('n_permutations', 'seed', 'permutation_batch_size', 'element_block_size', 'blas_threads', 'checkpoint_every'):
        value = c['statistics'][key]
        low = 0 if key in ('n_permutations', 'seed') else 1
        if type(value) is not int or value < low:
            raise PipelineError(f'statistics.{key} must be an integer >= {low}.')
    for section, key in [('statistics', 'independent_studies_confirmed'), ('clusters', 'enabled'), ('export', 'csv'), ('export', 'mesh')]:
        if type(c[section][key]) is not bool:
            raise PipelineError(f'{section}.{key} must be true or false, not a string.')
    alpha = c['export']['significance_alpha']
    if not isinstance(alpha, (int, float)) or not np.isfinite(alpha) or not 0 < float(alpha) < 1:
        raise PipelineError('export.significance_alpha must be a number in (0,1).')
    c['export']['significance_alpha'] = float(alpha)
    if not np.isfinite(c['statistics']['working_memory_mb']) or c['statistics']['working_memory_mb'] < 16:
        raise PipelineError('working_memory_mb must be >= 16.')
    tol = c['mesh']['coordinate_tolerance_mm']
    if not np.isfinite(tol) or tol < 0:
        raise PipelineError('coordinate_tolerance_mm must be finite and nonnegative.')
    if type(c['mesh']['tissue_tag']) is not int or c['mesh']['tissue_tag'] < 1:
        raise PipelineError('tissue_tag must be a positive integer.')
    for section, key in [('mesh', 'field_name'), ('input', 'mesh_glob'), ('input', 'folder_template')]:
        if not isinstance(c[section][key], str) or not c[section][key]:
            raise PipelineError(f'{section}.{key} must be a nonempty string.')
    ids = c['selection']['study_ids']
    if ids is not None and (not isinstance(ids, list) or len(ids) != len(set(map(str, ids)))):
        raise PipelineError('selection.study_ids must be a list of unique IDs, or null.')
    cl = c['clusters']
    if cl['statistic'] not in {'mass', 'extent', 'peak', 'volume', 'volume_weighted_mass'}:
        raise PipelineError('Unknown cluster statistic.')
    if not 0 < cl['forming_p_approx'] < 1:
        raise PipelineError('forming_p_approx must be in (0,1).')
    if cl['rho_threshold'] is not None and (not 0 < cl['rho_threshold'] <= 1):
        raise PipelineError('rho_threshold must be in (0,1], or null.')
    if cl['enabled'] and c['statistics']['n_permutations'] == 0:
        raise PipelineError('Cluster inference requires n_permutations > 0.')
    sim = c['simulation']
    for key in ('open_in_gmsh', 'dry_run', 'use_distance_from_csv'):
        if type(sim[key]) is not bool:
            raise PipelineError(f'simulation.{key} must be true or false, not a string.')
    if not isinstance(sim['output_folder_suffix'], str) or not sim['output_folder_suffix']:
        raise PipelineError('simulation.output_folder_suffix must be a nonempty string.')
    if not isinstance(sim['preferred_eeg_positions_file'], str) or not sim['preferred_eeg_positions_file']:
        raise PipelineError('simulation.preferred_eeg_positions_file must be a nonempty string.')
    if not np.isfinite(sim['default_distance_mm']) or sim['default_distance_mm'] < 0:
        raise PipelineError('simulation.default_distance_mm must be finite and >= 0.')
    run_only = sim['run_only_studies']
    if run_only is not None and (not isinstance(run_only, list) or len(run_only) != len(set(map(str, run_only)))):
        raise PipelineError('simulation.run_only_studies must be a list of unique IDs, or null.')

def read_csv_strings(path: Path) -> pd.DataFrame:
    """Read comma-, semicolon-, or tab-delimited study tables without changing cell contents."""
    if not path.is_file():
        raise PipelineError(f'CSV not found: {path}')
    content = path.read_bytes()
    text = content.decode('utf-8-sig')
    header_line = next((line for line in text.splitlines() if line.strip()), '')
    if not header_line:
        raise PipelineError(f'CSV is empty: {path}')
    try:
        delimiter = csv.Sniffer().sniff(header_line, delimiters=',;\t').delimiter
    except csv.Error as exc:
        raise PipelineError(f'Could not determine CSV delimiter for {path}. Use comma, semicolon, or tab.') from exc
    header = next(csv.reader(io.StringIO(header_line), delimiter=delimiter), [])
    stripped_header = [v.strip() for v in header]
    if len(stripped_header) != len(set(stripped_header)):
        raise PipelineError(f'Duplicate CSV column names in {path}')
    frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, sep=delimiter)
    frame.attrs['source_sha256'] = hashlib.sha256(content).hexdigest()
    frame.columns = [str(v).strip() for v in frame.columns]
    if frame.columns.duplicated().any():
        raise PipelineError(f'Duplicate CSV column names in {path}')
    return frame

def load_studies(c: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = read_csv_strings(c['input']['study_table'])
    col = c['columns']
    required = [col['study_id'], col['effect_size']]
    required += [col[k] for k in ('study_name', 'include', 'independent_unit') if col[k] is not None]
    if c['selection']['outcome'] is not None:
        if col['outcome'] is None:
            raise PipelineError('An outcome selection requires columns.outcome.')
        required.append(col['outcome'])
    missing = set(required) - set(frame.columns)
    if missing:
        raise PipelineError(f'Missing study-table columns: {sorted(missing)}. See README or edit columns in YAML.')
    frame[col['study_id']] = frame[col['study_id']].str.strip()
    reason = pd.Series('', index=frame.index, dtype=str)
    if col['include'] is not None:
        values = frame[col['include']].str.strip().str.lower()
        yes, no = ({'1', 'true', 'yes', 'y'}, {'0', 'false', 'no', 'n'})
        invalid = ~values.isin(yes | no)
        if invalid.any():
            raise PipelineError(f'Unrecognized include values on CSV rows {(np.flatnonzero(invalid) + 2).tolist()}.')
        reason.loc[values.isin(no)] = 'include=false'
    outcome = c['selection']['outcome']
    if outcome is not None:
        mismatch = frame[col['outcome']].str.strip() != str(outcome)
        reason.loc[mismatch & (reason == '')] = 'different outcome'
    ids = c['selection']['study_ids']
    if ids is not None:
        requested = {str(v) for v in ids}
        available = set(frame.loc[reason == '', col['study_id']])
        if requested - available:
            raise PipelineError(f'Requested IDs absent from the included outcome: {sorted(requested - available)}')
        reason.loc[~frame[col['study_id']].isin(requested) & (reason == '')] = 'not selected'
    audit = frame.copy()
    audit['pipeline_selected'] = reason == ''
    audit['pipeline_exclusion_reason'] = reason
    selected = frame.loc[reason == ''].copy().reset_index(drop=True)
    if outcome is None and col['outcome'] is not None and (col['outcome'] in selected):
        if selected[col['outcome']].str.strip().nunique() > 1:
            raise PipelineError('Selected rows contain multiple outcome values. Choose selection.outcome; v0.1 does not silently pool different outcomes.')
    if len(selected) < 3:
        raise PipelineError('At least three selected studies are required.')
    identifiers = selected[col['study_id']]
    if (identifiers == '').any() or identifiers.duplicated().any():
        raise PipelineError('Selected Study_ID values must be nonempty and unique. IDs are strings; leading zeros are retained.')
    effect = pd.to_numeric(selected[col['effect_size']], errors='coerce').to_numpy(dtype=np.float64)
    if not np.isfinite(effect).all():
        bad = identifiers[~np.isfinite(effect)].tolist()
        raise PipelineError(f'Missing, infinite or nonnumeric effect sizes for studies: {bad}')
    if np.ptp(effect) == 0:
        raise PipelineError('Effect sizes are constant; Spearman correlation is undefined.')
    selected[col['effect_size']] = effect
    if col['independent_unit'] is not None:
        units = selected[col['independent_unit']].str.strip()
        if (units == '').any() or units.duplicated().any():
            raise PipelineError('Repeated/missing independent_unit values: v0.1 requires one independent effect per unit. It does not implement dependent-effect/block permutation models.')
    return (selected, audit)

def validate_inference(c: dict[str, Any]) -> None:
    s = c['statistics']
    for key in ('effect_size_definition', 'positive_effect_means'):
        text = s[key]
        if not isinstance(text, str) or not text.strip() or text.lstrip().upper().startswith('EDIT'):
            raise PipelineError(f'Complete statistics.{key} before running statistics.')
    if s['n_permutations'] and (not s['independent_studies_confirmed']):
        raise PipelineError('Review independence/exchangeability, then set independent_studies_confirmed: true. No unrestricted permutations will run without this explicit check.')

def resolve_meshes(c: dict[str, Any], studies: pd.DataFrame) -> pd.DataFrame:
    ids = studies[c['columns']['study_id']].tolist()
    source = c['input']['manifest_csv']
    rows = []
    if source is not None:
        manifest = read_csv_strings(source)
        required = {'Study_ID', 'head_model_id', 'mesh_path'}
        if required - set(manifest.columns):
            raise PipelineError(f'Manifest requires columns {sorted(required)}')
        manifest = manifest.loc[manifest.head_model_id == c['mesh']['head_model_id']]
        for study_id in ids:
            matches = manifest.loc[manifest.Study_ID == study_id]
            if len(matches) != 1:
                raise PipelineError(f"Expected exactly one manifest row for {study_id} / {c['mesh']['head_model_id']}; found {len(matches)}.")
            path = resolve_path(matches.iloc[0].mesh_path, source.parent)
            if not path.is_file():
                raise PipelineError(f'Mesh not found: {path}')
            rows.append((study_id, c['mesh']['head_model_id'], str(path)))
    else:
        root = c['input']['simulation_root']
        for study_id in ids:
            try:
                folder = root / c['input']['folder_template'].format(Study_ID=study_id)
            except (KeyError, ValueError) as exc:
                raise PipelineError('folder_template supports {Study_ID} only.') from exc
            if not folder.resolve().is_relative_to(root):
                raise PipelineError('Study folder escapes simulation_root; use an explicit manifest instead.')
            matches = sorted(folder.glob(c['input']['mesh_glob']))
            matches = [p for p in matches if p.is_file()]
            if len(matches) != 1:
                raise PipelineError(f'Study {study_id}: expected exactly one mesh in {folder}, found {len(matches)}. Use a more specific glob or a manifest; no file is chosen arbitrarily.')
            rows.append((study_id, c['mesh']['head_model_id'], str(matches[0].resolve())))
    result = pd.DataFrame(rows, columns=['Study_ID', 'head_model_id', 'mesh_path'])
    if result.mesh_path.duplicated().any():
        raise PipelineError('Multiple studies point to the same mesh file. Correct the manifest rather than silently reusing a simulation.')
    return result

# -----------------------------------------------------------------------------
# STATISTICS
# -----------------------------------------------------------------------------
LOG = logging.getLogger(__name__)
RHO_TOLERANCE = 100 * np.finfo(np.float64).eps

def normalized_ranks(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows are elements, columns are studies. Average ranks; centered unit L2 norm."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 3 or (not np.isfinite(x).all()):
        raise PipelineError('Fields must be finite (elements, studies), with at least 3 studies.')
    ranked = rankdata(x, method='average', axis=1)
    ranked -= ranked.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(ranked, axis=1)
    valid = norm > 0
    ranked /= np.where(valid, norm, 1.0)[:, None]
    return (ranked, valid)

def effect_ranks(effect: np.ndarray) -> np.ndarray:
    effect = np.asarray(effect, dtype=np.float64)
    if effect.ndim != 1:
        raise PipelineError('Effect sizes must be a vector.')
    ranked, valid = normalized_ranks(effect[None, :])
    if not valid[0]:
        raise PipelineError('Effect sizes are constant.')
    return ranked[0]

def build_rank_cache(cache: Path, fields: np.ndarray, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Cache ranks only for this exact extraction study set/order, not a larger subset."""
    source = read_json(cache / 'metadata.json')
    environment = versions()
    identity = digest({'extraction': source['fingerprint'], 'fields_sha256': source['artifacts']['fields.npy'], 'method': 'average-centered-unitL2-float64-v1', 'code': code_fingerprint(), 'versions': {k: environment[k] for k in ('numpy', 'scipy')}})
    cache = cache / ('ranks_' + identity[:20])
    cache.mkdir(parents=True, exist_ok=True)
    meta_path = cache / 'rank_metadata.json'
    if meta_path.is_file():
        meta = read_json(meta_path)
        if meta['identity'] != identity:
            raise PipelineError('Rank cache belongs to a different field matrix.')
        for name, expected in meta['artifacts'].items():
            if sha256_file(cache / name) != expected:
                raise PipelineError('Rank-cache integrity check failed.')
        return (np.load(cache / 'ranks.npy', mmap_mode='r'), np.load(cache / 'valid.npy', allow_pickle=False))
    LOG.info('Ranking fields once for this selected study set.')
    ranks = np.lib.format.open_memmap(cache / 'ranks.npy', mode='w+', dtype=np.float64, shape=fields.shape)
    valid = np.zeros(len(fields), dtype=bool)
    for start in range(0, len(fields), block_size):
        stop = min(len(fields), start + block_size)
        ranks[start:stop], valid[start:stop] = normalized_ranks(fields[start:stop])
    ranks.flush()
    del ranks
    np.save(cache / 'valid.npy', valid, allow_pickle=False)
    atomic_json(meta_path, {'identity': identity, 'artifacts': {name: sha256_file(cache / name) for name in ('ranks.npy', 'valid.npy')}})
    return (np.load(cache / 'ranks.npy', mmap_mode='r'), valid)

def observed_rho(ranks: np.ndarray, y: np.ndarray, valid: np.ndarray, block_size: int) -> np.ndarray:
    if ranks.shape[1] != len(y) or valid.shape != (len(ranks),):
        raise PipelineError('Rank matrix, effect vector and valid-element mask do not align.')
    result = np.empty(len(ranks), dtype=np.float64)
    for start in range(0, len(ranks), block_size):
        stop = min(len(ranks), start + block_size)
        result[start:stop] = np.clip((ranks[start:stop] @ y[:, None])[:, 0], -1.0, 1.0)
    result[~valid] = np.nan
    if not valid.any():
        raise PipelineError('All elements have constant E-fields across studies; no correlations are defined.')
    return result

def working_blocks(settings: dict[str, Any], n_studies: int, n_elements: int, clusters: bool) -> tuple[int, int, int]:
    """Conservative workspace estimate; mesh loading/OS cache/BLAS are not hard-capped."""
    budget = int(settings['working_memory_mb'] * 1024 ** 2)
    batch = 1 if clusters else min(settings['permutation_batch_size'], max(1, settings['n_permutations']))
    reserve = n_elements * (64 if clusters else 32)
    remaining = budget - reserve
    if remaining <= 0:
        raise PipelineError('working_memory_mb is too small even for spatial summary arrays. Increase it.')
    while batch > 1 and n_studies * batch * 24 + 8 * batch * 8192 > remaining:
        batch = max(1, batch // 2)
    remaining -= n_studies * batch * 24
    block = min(settings['element_block_size'], n_elements, remaining // max(1, 96 * n_studies + 32 * batch))
    if block < 1:
        raise PipelineError('working_memory_mb is too small for one element block.')
    estimate = reserve + n_studies * batch * 24 + block * (96 * n_studies + 32 * batch)
    return (int(block), int(batch), int(estimate))

# -----------------------------------------------------------------------------
# CLUSTERS
# -----------------------------------------------------------------------------
def build_adjacency(connectivity: np.ndarray) -> sparse.csr_matrix:
    """Two tetrahedra are adjacent only when they share a complete three-node face."""
    n = len(connectivity)
    face_indices = np.array(list(combinations(range(4), 3)))
    faces = np.sort(connectivity[:, face_indices].reshape(-1, 3), axis=1)
    owner = np.repeat(np.arange(n, dtype=np.int64), 4)
    order = np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))
    faces, owner = (faces[order], owner[order])
    equal = np.all(faces[1:] == faces[:-1], axis=1)
    if np.any(equal[:-1] & equal[1:]):
        raise PipelineError('Non-manifold tetrahedral mesh: a face belongs to more than two tetrahedra.')
    pair = np.flatnonzero(equal)
    left, right = (owner[pair], owner[pair + 1])
    if np.any(left == right):
        raise PipelineError('Degenerate tetrahedral topology.')
    rows, cols = (np.r_[left, right], np.r_[right, left])
    graph = sparse.csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(n, n))
    return graph

def rho_threshold(settings: dict[str, Any], n_studies: int) -> float:
    if settings['rho_threshold'] is not None:
        return float(settings['rho_threshold'])
    tcrit = student_t.isf(settings['forming_p_approx'] / 2, df=n_studies - 2)
    return float(tcrit / np.sqrt(n_studies - 2 + tcrit ** 2))

def cluster_components(rho: np.ndarray, adjacency: sparse.csr_matrix, threshold: float, statistic: str, volumes: np.ndarray):
    """Yield sign, selected indices, labels and grouped statistics (no per-cluster rescans)."""
    for sign in (1, -1):
        members = np.flatnonzero(np.isfinite(rho) & (sign * rho >= threshold - RHO_TOLERANCE))
        if not len(members):
            continue
        n, labels = connected_components(adjacency[members][:, members], directed=False)
        magnitude = np.abs(rho[members])
        if statistic == 'mass':
            stats = np.bincount(labels, weights=magnitude, minlength=n)
        elif statistic == 'extent':
            stats = np.bincount(labels, minlength=n).astype(float)
        elif statistic == 'volume':
            stats = np.bincount(labels, weights=volumes[members], minlength=n)
        elif statistic == 'volume_weighted_mass':
            stats = np.bincount(labels, weights=volumes[members] * magnitude, minlength=n)
        elif statistic == 'peak':
            stats = np.zeros(n)
            np.maximum.at(stats, labels, magnitude)
        else:
            raise PipelineError(f'Unknown cluster statistic: {statistic}')
        yield (sign, members, labels, stats)

def maximum_cluster(rho: np.ndarray, adjacency: sparse.csr_matrix, threshold: float, statistic: str, volumes: np.ndarray) -> float:
    return max((float(stats.max()) for _, _, _, stats in cluster_components(rho, adjacency, threshold, statistic, volumes)), default=0.0)

def observed_clusters(rho: np.ndarray, adjacency: sparse.csr_matrix, threshold: float, statistic: str, volumes: np.ndarray, coordinates: np.ndarray, element_ids: np.ndarray, null_max: np.ndarray) -> tuple[list[dict], dict[str, np.ndarray]]:
    rows = []
    ids = np.zeros(len(rho), dtype=np.int64)
    p_map = np.full(len(rho), np.nan)
    for sign, members, labels, stats in cluster_components(rho, adjacency, threshold, statistic, volumes):
        order = np.argsort(labels, kind='stable')
        ordered_members = members[order]
        lengths = np.bincount(labels, minlength=len(stats))
        offset = 0
        for label, length in enumerate(lengths):
            selected = ordered_members[offset:offset + length]
            offset += length
            cid = len(rows) + 1
            peak = selected[np.argmax(np.abs(rho[selected]))]
            tol = RHO_TOLERANCE * max(1.0, abs(stats[label]))
            p = (1 + np.count_nonzero(null_max >= stats[label] - tol)) / (1 + len(null_max))
            ids[selected] = cid
            p_map[selected] = p
            rows.append({'cluster_id': cid, 'sign': 'positive' if sign == 1 else 'negative', 'n_elements': int(length), 'volume_mm3': float(volumes[selected].sum()), 'statistic_type': statistic, 'statistic': float(stats[label]), 'p_cluster_fwer': float(p), 'peak_element_id': int(element_ids[peak]), 'peak_rho': float(rho[peak]), 'peak_x_mm': float(coordinates[peak, 0]), 'peak_y_mm': float(coordinates[peak, 1]), 'peak_z_mm': float(coordinates[peak, 2])})
    return (rows, {'cluster_id': ids, 'p_cluster_fwer': p_map})

# -----------------------------------------------------------------------------
# DESCRIPTIVE CONNECTED-REGION REPORTING
# -----------------------------------------------------------------------------
REGION_COLUMNS = [
    'region_id', 'sign', 'n_elements', 'volume_mm3',
    'mean_rho', 'median_rho', 'volume_weighted_mean_rho',
    'peak_element_id', 'peak_rho', 'peak_p_value', 'min_p_value',
    'peak_x_mm', 'peak_y_mm', 'peak_z_mm',
    'centroid_x_mm', 'centroid_y_mm', 'centroid_z_mm',
    'significance_alpha',
]


def subset_face_regions(connectivity: np.ndarray, members: np.ndarray) -> tuple[int, np.ndarray]:
    """Connected regions among selected tetrahedra using shared complete triangular faces."""
    members = np.asarray(members, dtype=np.int64)
    if len(members) == 0:
        return (0, np.empty(0, dtype=np.int64))
    if len(members) == 1:
        return (1, np.zeros(1, dtype=np.int64))
    selected_connectivity = np.asarray(connectivity[members], dtype=np.int64)
    face_indices = np.array(list(combinations(range(4), 3)), dtype=np.int64)
    faces = np.sort(selected_connectivity[:, face_indices].reshape(-1, 3), axis=1)
    owner = np.repeat(np.arange(len(members), dtype=np.int64), 4)
    order = np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))
    faces, owner = faces[order], owner[order]
    equal = np.all(faces[1:] == faces[:-1], axis=1)
    if np.any(equal[:-1] & equal[1:]):
        raise PipelineError('Non-manifold tetrahedral mesh within significant elements: a face belongs to more than two tetrahedra.')
    pairs = np.flatnonzero(equal)
    if len(pairs):
        left, right = owner[pairs], owner[pairs + 1]
        rows, cols = np.r_[left, right], np.r_[right, left]
        graph = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
            shape=(len(members), len(members)),
        )
    else:
        graph = sparse.csr_matrix((len(members), len(members)), dtype=np.uint8)
    return connected_components(graph, directed=False)


def significant_regions(
    rho: np.ndarray,
    p_values: np.ndarray,
    connectivity: np.ndarray,
    volumes: np.ndarray,
    coordinates: np.ndarray,
    element_ids: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Group already-significant elements; this is descriptive reporting, not cluster-level inference."""
    if rho.shape != p_values.shape or rho.shape != (len(element_ids),):
        raise PipelineError('Region reporting received inconsistent element arrays.')
    significant = np.zeros(len(rho), dtype=np.int8)
    valid = np.isfinite(rho) & np.isfinite(p_values) & (p_values <= alpha)
    significant[valid & (rho > 0)] = 1
    significant[valid & (rho < 0)] = -1

    regions: list[dict[str, Any]] = []
    for sign_value, sign_name in ((1, 'positive'), (-1, 'negative')):
        members = np.flatnonzero(significant == sign_value)
        n_regions, labels = subset_face_regions(connectivity, members)
        if not n_regions:
            continue
        order = np.argsort(labels, kind='stable')
        ordered_members = members[order]
        lengths = np.bincount(labels, minlength=n_regions)
        offset = 0
        for local_id, length in enumerate(lengths):
            selected = ordered_members[offset:offset + length]
            offset += length
            abs_rho = np.abs(rho[selected])
            peak = int(selected[int(np.nanargmax(abs_rho))])
            weights = np.asarray(volumes[selected], dtype=np.float64)
            volume = float(weights.sum())
            centroid = np.average(np.asarray(coordinates[selected], dtype=np.float64), axis=0, weights=weights)
            regions.append({
                '_members': selected,
                'sign': sign_name,
                'n_elements': int(length),
                'volume_mm3': volume,
                'mean_rho': float(np.mean(rho[selected])),
                'median_rho': float(np.median(rho[selected])),
                'volume_weighted_mean_rho': float(np.average(rho[selected], weights=weights)),
                'peak_element_id': int(element_ids[peak]),
                'peak_rho': float(rho[peak]),
                'peak_p_value': float(p_values[peak]),
                'min_p_value': float(np.min(p_values[selected])),
                'peak_x_mm': float(coordinates[peak, 0]),
                'peak_y_mm': float(coordinates[peak, 1]),
                'peak_z_mm': float(coordinates[peak, 2]),
                'centroid_x_mm': float(centroid[0]),
                'centroid_y_mm': float(centroid[1]),
                'centroid_z_mm': float(centroid[2]),
                'significance_alpha': float(alpha),
            })

    # Stable, objective numbering: largest physical region first, then stronger peak, then element ID.
    regions.sort(key=lambda row: (-row['volume_mm3'], -abs(row['peak_rho']), row['peak_element_id']))
    region_ids = np.zeros(len(rho), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for region_id, item in enumerate(regions, start=1):
        members = item.pop('_members')
        region_ids[members] = region_id
        row = {'region_id': region_id, **item}
        rows.append(row)
    return significant, region_ids, rows


def write_region_table(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows, columns=REGION_COLUMNS).to_csv(path, index=False)

# -----------------------------------------------------------------------------
# EXTRACTION
# -----------------------------------------------------------------------------
LOG = logging.getLogger(__name__)

@dataclass
class MeshData:
    element_ids: np.ndarray
    connectivity: np.ndarray
    node_coordinates: np.ndarray
    values: np.ndarray

    @property
    def vertices(self) -> np.ndarray:
        return self.node_coordinates[self.connectivity - 1]

    @property
    def coordinates(self) -> np.ndarray:
        return self.vertices.mean(axis=1)

    @property
    def volumes(self) -> np.ndarray:
        vertices = self.vertices
        return np.abs(np.linalg.det(vertices[:, 1:] - vertices[:, :1])) / 6.0

def require_simnibs():
    try:
        from simnibs.mesh_tools import mesh_io
    except ImportError as exc:
        raise PipelineError('SimNIBS is not importable in this interpreter. Use the interpreter that runs your existing batch script. Statistics on an extracted cache and the synthetic demo do not need SimNIBS.') from exc
    return mesh_io

def mesh_to_data(mesh: Any, field_name: str, tissue_tag: int) -> MeshData:
    tags = np.asarray(mesh.elm.tag1).ravel()
    types = np.asarray(mesh.elm.elm_type).ravel()
    selected = (tags == tissue_tag) & (types == 4)
    if not selected.any():
        raise PipelineError(f'No tetrahedra with tissue tag {tissue_tag}.')
    fields = [f for f in mesh.elmdata if f.field_name == field_name]
    if len(fields) != 1:
        raise PipelineError(f'Expected one ELEMENT field named {field_name!r}; found {len(fields)}. Available: {[f.field_name for f in mesh.elmdata]}')
    raw = np.asarray(fields[0].value)
    if raw.ndim == 2 and raw.shape[1] == 1:
        raw = raw[:, 0]
    if raw.ndim != 1 or raw.shape[0] != len(tags):
        raise PipelineError('Analysis requires a scalar element field; vector or node fields are not silently converted.')
    data = MeshData(np.asarray(mesh.elm.elm_number, dtype=np.int64).ravel()[selected], np.asarray(mesh.elm.node_number_list, dtype=np.int64)[selected, :4], np.asarray(mesh.nodes.node_coord, dtype=np.float64), np.asarray(raw[selected], dtype=np.float64))
    validate_mesh_data(data)
    return data

def read_scalar_mesh(path: Path, field_name: str, tissue_tag: int) -> MeshData:
    mesh_io = require_simnibs()
    mesh = mesh_io.read_msh(str(path))
    return mesh_to_data(mesh, field_name, tissue_tag)

def validate_mesh_data(data: MeshData) -> None:
    n = len(data.element_ids)
    if n == 0 or data.element_ids.shape != (n,) or len(np.unique(data.element_ids)) != n:
        raise PipelineError('Empty or duplicate reference element IDs.')
    if data.connectivity.shape != (n, 4) or data.values.shape != (n,):
        raise PipelineError('Element IDs, tetrahedral connectivity and scalar values have inconsistent shapes.')
    if data.node_coordinates.ndim != 2 or data.node_coordinates.shape[1] != 3:
        raise PipelineError('Node coordinates must have shape (n_nodes, 3).')
    if not np.isfinite(data.node_coordinates).all() or not np.isfinite(data.values).all():
        raise PipelineError('Nonfinite coordinates or field values. Missing values are not imputed or pairwise deleted.')
    if not np.issubdtype(data.connectivity.dtype, np.integer) or not np.issubdtype(data.element_ids.dtype, np.integer):
        raise PipelineError('Element and node IDs must be integers.')
    if np.any(data.element_ids < 1) or np.any(data.connectivity < 1) or np.any(data.connectivity > len(data.node_coordinates)):
        raise PipelineError('Invalid 1-based mesh node/element IDs.')
    if np.any(data.volumes <= 0):
        raise PipelineError('Degenerate tetrahedra found.')

def assert_same_mesh(reference: MeshData, candidate: MeshData, tolerance_mm: float) -> None:
    if not np.array_equal(reference.element_ids, candidate.element_ids):
        raise PipelineError('Reference element IDs/order differ. Native elements cannot be stacked by position.')
    if not np.array_equal(reference.connectivity, candidate.connectivity):
        raise PipelineError('Tetrahedral connectivity/order differs between simulations.')
    if reference.node_coordinates.shape != candidate.node_coordinates.shape or not np.allclose(reference.node_coordinates, candidate.node_coordinates, atol=tolerance_mm, rtol=0):
        raise PipelineError('Reference node coordinates differ. Use one common mesh/head model per analysis.')

def load_cache(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    meta_path = path / 'metadata.json'
    if not meta_path.is_file():
        raise PipelineError(f'Missing complete cache metadata: {meta_path}')
    meta = read_json(meta_path)
    if meta.get('schema') != 1:
        raise PipelineError('Unrecognized cache schema.')
    for name, expected in meta['artifacts'].items():
        file = path / name
        if not file.is_file() or sha256_file(file) != expected:
            raise PipelineError(f'Cache integrity check failed: {file}. Do not reuse altered/incomplete cached data.')
    arrays = {name[:-4]: np.load(path / name, mmap_mode='r', allow_pickle=False) for name in meta['artifacts'] if name.endswith('.npy')}
    return (meta, arrays)

def extract_fields(c: dict[str, Any], manifest: pd.DataFrame, reader: Callable[[Path, str, int], MeshData]=read_scalar_mesh) -> Path:
    sources = []
    for row in manifest.itertuples(index=False):
        LOG.info('Checking input fingerprint for study %s', row.Study_ID)
        path = Path(row.mesh_path)
        sources.append({'Study_ID': row.Study_ID, 'path': str(path), 'sha256': sha256_file(path)})
    identity = {'schema': 1, 'mesh': c['mesh'], 'sources': sources, 'code': code_fingerprint(), 'source_type': 'simnibs_mesh' if reader is read_scalar_mesh else 'synthetic_test_adapter'}
    fingerprint = digest(identity)
    root = c['project']['cache_dir']
    root.mkdir(parents=True, exist_ok=True)
    target = root / ('fields_' + fingerprint[:20])
    with exclusive_lock(root / (target.name + '.lock')):
        if target.is_dir():
            meta, _ = load_cache(target)
            if meta['fingerprint'] != fingerprint:
                raise PipelineError('Cache-key collision; refusing reuse.')
            LOG.info('Reusing checked E-field cache: %s', target)
            return target
        temp = Path(tempfile.mkdtemp(prefix='.extract-', dir=root))
        try:
            reference = None
            matrix = None
            for index, src in enumerate(sources):
                LOG.info('Extracting study %s (%d/%d)', src['Study_ID'], index + 1, len(sources))
                data = reader(Path(src['path']), c['mesh']['field_name'], c['mesh']['tissue_tag'])
                validate_mesh_data(data)
                if reference is None:
                    reference = data
                    matrix = np.lib.format.open_memmap(temp / 'fields.npy', mode='w+', dtype=np.float64, shape=(len(data.values), len(sources)))
                    for name, value in {'element_ids': data.element_ids, 'connectivity': data.connectivity, 'node_coordinates': data.node_coordinates, 'coordinates': data.coordinates, 'volumes_mm3': data.volumes}.items():
                        np.save(temp / (name + '.npy'), value, allow_pickle=False)
                else:
                    assert_same_mesh(reference, data, c['mesh']['coordinate_tolerance_mm'])
                matrix[:, index] = data.values
            if matrix is None:
                raise PipelineError('No meshes to extract.')
            matrix.flush()
            del matrix
            for src in sources:
                if sha256_file(Path(src['path'])) != src['sha256']:
                    raise PipelineError(f"Input changed during extraction: {src['path']}")
            artifacts = {p.name: sha256_file(p) for p in sorted(temp.glob('*.npy'))}
            meta = {'schema': 1, 'fingerprint': fingerprint, 'identity': identity, 'study_ids': [s['Study_ID'] for s in sources], 'reference_mesh': sources[0]['path'], 'reference_sha256': sources[0]['sha256'], 'shape': [len(reference.values), len(sources)], 'artifacts': artifacts, 'created_utc': timestamp(), 'versions': versions()}
            atomic_json(temp / 'metadata.json', meta)
            os.replace(temp, target)
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    return target

# -----------------------------------------------------------------------------
# PERMUTATIONS
# -----------------------------------------------------------------------------
LOG = logging.getLogger(__name__)

def accumulate_batch(ranks: np.ndarray, permuted_y: np.ndarray, rho: np.ndarray, valid: np.ndarray, block_size: int, cluster_args: dict[str, Any] | None=None):
    """Process spatial blocks but retain ONE global maximum per shuffle."""
    n_elements, batch = (len(ranks), permuted_y.shape[1])
    counts = np.zeros(n_elements, dtype=np.int64)
    maxima = np.zeros(batch, dtype=np.float64)
    global_maps = np.empty((n_elements, batch)) if cluster_args is not None else None
    for start in range(0, n_elements, block_size):
        stop = min(start + block_size, n_elements)
        scores = ranks[start:stop] @ permuted_y
        np.clip(scores, -1.0, 1.0, out=scores)
        if global_maps is not None:
            global_maps[start:stop] = scores
        np.abs(scores, out=scores)
        scores[~valid[start:stop]] = 0.0
        threshold = np.abs(rho[start:stop]) - RHO_TOLERANCE
        counts[start:stop] = np.count_nonzero(scores >= threshold[:, None], axis=1)
        maxima = np.maximum(maxima, scores.max(axis=0))
    counts[~valid] = 0
    cluster_max = np.empty(0)
    if global_maps is not None:
        global_maps[~valid] = np.nan
        cluster_max = np.array([maximum_cluster(global_maps[:, i], **cluster_args) for i in range(batch)])
    return (counts, maxima, cluster_max)

def maxstat_pvalues(rho: np.ndarray, valid: np.ndarray, null_max: np.ndarray) -> np.ndarray:
    sorted_null = np.sort(null_max)
    counts = len(null_max) - np.searchsorted(sorted_null, np.abs(rho) - RHO_TOLERANCE, side='left')
    p = (counts + 1.0) / (len(null_max) + 1.0)
    p[~valid] = np.nan
    return p

def run_permutations(ranks: np.ndarray, y: np.ndarray, rho: np.ndarray, valid: np.ndarray, *, n_permutations: int, seed: int, block_size: int, batch_size: int, checkpoint: Path, identity: str, cluster_args: dict[str, Any] | None=None, checkpoint_every: int=256, progress: Callable[[int, int], None] | None=None) -> dict[str, np.ndarray | int]:
    if n_permutations < 1:
        raise PipelineError('Permutation count must be positive.')
    rng = np.random.default_rng(seed)
    counts = np.zeros(len(ranks), dtype=np.int64)
    null_max = np.empty(n_permutations, dtype=np.float64)
    cluster_null = np.empty(n_permutations if cluster_args is not None else 0, dtype=np.float64)
    done = 0
    if checkpoint.is_file():
        with np.load(checkpoint, allow_pickle=False) as state:
            meta = json.loads(str(state['metadata'].item()))
            if meta['identity'] != identity:
                raise PipelineError('Checkpoint does not match data, effects, seed, method or software identity.')
            done = int(meta['completed'])
            if done > n_permutations:
                raise PipelineError(f'Checkpoint already contains {done} permutations, more than requested ({n_permutations}). Use a new analysis_name for a shorter independent run.')
            if state['counts'].shape != counts.shape or state['null_max'].shape != (done,):
                raise PipelineError('Checkpoint array shapes are inconsistent.')
            if state['cluster_null'].shape != ((done,) if cluster_args is not None else (0,)):
                raise PipelineError('Checkpoint cluster settings are inconsistent.')
            if not np.issubdtype(state['counts'].dtype, np.integer) or np.any(state['counts'] < 0) or np.any(state['counts'] > done) or (not np.isfinite(state['null_max']).all()) or np.any(state['null_max'] < 0) or np.any(state['null_max'] > 1):
                raise PipelineError('Checkpoint contains invalid counts or null statistics.')
            counts[:] = state['counts']
            null_max[:done] = state['null_max']
            if cluster_args is not None:
                cluster_null[:done] = state['cluster_null']
            rng.bit_generator.state = meta['rng_state']
        LOG.info('Resuming from %d completed permutations.', done)
    if cluster_args is not None:
        batch_size = 1
    while done < n_permutations:
        batch = min(batch_size, n_permutations - done)
        indices = np.column_stack([rng.permutation(len(y)) for _ in range(batch)])
        delta, maxima, cl_max = accumulate_batch(ranks, y[indices], rho, valid, block_size, cluster_args)
        counts += delta
        null_max[done:done + batch] = maxima
        if cluster_args is not None:
            cluster_null[done:done + batch] = cl_max
        previous = done
        done += batch
        meta = {'schema': 1, 'identity': identity, 'completed': done, 'rng_state': rng.bit_generator.state, 'test': 'random pairings; two-sided absolute rho; add-one p; global maximum'}
        if done == n_permutations or done // checkpoint_every != previous // checkpoint_every:
            atomic_npz(checkpoint, counts=counts, null_max=null_max[:done], cluster_null=cluster_null[:done] if cluster_args is not None else cluster_null, metadata=np.array(canonical_json(meta)))
        if done == n_permutations or done // 1000 != previous // 1000:
            LOG.info('Permutations: %d / %d', done, n_permutations)
        if progress is not None:
            progress(done, n_permutations)
    p = (counts + 1.0) / (n_permutations + 1.0)
    p[~valid] = np.nan
    return {'p_uncorrected': p, 'p_fwer_maxstat': maxstat_pvalues(rho, valid, null_max), 'null_max_abs_rho': null_max, 'null_max_cluster': cluster_null, 'completed': done}

# -----------------------------------------------------------------------------
# EXPORT
# -----------------------------------------------------------------------------
def signed_neg_log10(p: np.ndarray, rho: np.ndarray) -> np.ndarray:
    out = np.full(len(rho), np.nan)
    valid = np.isfinite(p) & (p > 0) & (p <= 1) & np.isfinite(rho)
    out[valid] = np.sign(rho[valid]) * -np.log10(p[valid])
    return out

def export_tables(output: Path, arrays: dict[str, np.ndarray], results: dict[str, np.ndarray], write_csv: bool, cluster_rows: list[dict] | None) -> None:
    """NPZ is authoritative; CSV is an optional human-readable export, written in chunks."""
    output.mkdir(parents=True, exist_ok=True)
    base = {'element_id': arrays['element_ids'], 'coordinates_mm': arrays['coordinates'], 'volume_mm3': arrays['volumes_mm3'], **results}
    np.savez_compressed(output / 'element_results.npz', **base)
    if write_csv:
        path = output / 'element_results.csv'
        for start in range(0, len(arrays['element_ids']), 50000):
            sl = slice(start, start + 50000)
            data = {'element_id': arrays['element_ids'][sl], 'x_mm': arrays['coordinates'][sl, 0], 'y_mm': arrays['coordinates'][sl, 1], 'z_mm': arrays['coordinates'][sl, 2], 'volume_mm3': arrays['volumes_mm3'][sl]}
            data.update({name: values[sl] for name, values in results.items()})
            pd.DataFrame(data).to_csv(path, index=False, mode='w' if start == 0 else 'a', header=start == 0)
    if cluster_rows is not None:
        columns = ['cluster_id', 'sign', 'n_elements', 'volume_mm3', 'statistic_type', 'statistic', 'p_cluster_fwer', 'peak_element_id', 'peak_rho', 'peak_x_mm', 'peak_y_mm', 'peak_z_mm']
        pd.DataFrame(cluster_rows, columns=columns).to_csv(output / 'cluster_results.csv', index=False)

def export_mesh(path: Path, cache_meta: dict[str, Any], arrays: dict[str, np.ndarray], results: dict[str, np.ndarray], settings: dict[str, Any]) -> None:
    """Export analysis fields on a GM-only mesh.

    The statistical analysis is performed only on gray-matter tetrahedra (``tissue_tag``).
    The reference full-head mesh is first validated against the cached extraction, then cropped
    to that same tissue before any result fields are written. This keeps scalp/skull/other tissues
    out of the visualization mesh while preserving the exact GM element ordering used in the
    analysis. No visualization colours or display thresholds are imposed here.
    """
    reference_path = Path(cache_meta['reference_mesh'])
    if not reference_path.is_file() or sha256_file(reference_path) != cache_meta['reference_sha256']:
        raise PipelineError('The original reference mesh is missing or has changed; map export stopped.')

    mesh_io = require_simnibs()
    full_mesh = mesh_io.read_msh(str(reference_path))

    # Validate the full reference mesh against the exact GM tetrahedra used for extraction.
    candidate = mesh_to_data(full_mesh, settings['field_name'], settings['tissue_tag'])
    reference = MeshData(
        arrays['element_ids'],
        arrays['connectivity'],
        arrays['node_coordinates'],
        arrays['fields'][:, 0],
    )
    assert_same_mesh(reference, candidate, settings['coordinate_tolerance_mm'])

    # Match the historical mapping workflow: write analysis values to a GM-only mesh.
    gm_mesh = full_mesh.crop_mesh(settings['tissue_tag'])
    gm_coordinates = np.asarray(gm_mesh.elements_baricenters()[:], dtype=np.float64)
    expected_coordinates = np.asarray(arrays['coordinates'], dtype=np.float64)

    if gm_coordinates.shape != expected_coordinates.shape:
        raise PipelineError(
            'GM-only export mesh has a different number of elements than the analyzed GM mask: '
            f'{gm_coordinates.shape[0]} vs {expected_coordinates.shape[0]}.'
        )
    if not np.allclose(
        gm_coordinates,
        expected_coordinates,
        atol=settings['coordinate_tolerance_mm'],
        rtol=0,
    ):
        raise PipelineError(
            'GM-only export element order/coordinates differ from the analyzed GM elements; '
            'refusing to map values by position.'
        )

    gm_mesh.elmdata = []
    gm_mesh.nodedata = []
    for name, values in results.items():
        values = np.asarray(values)
        if values.shape != (len(expected_coordinates),):
            raise PipelineError(f'Map {name} has an inconsistent GM element count.')
        gm_mesh.add_element_field(values.astype(np.float64, copy=False), name)

    mesh_io.write_msh(gm_mesh, str(path))

# -----------------------------------------------------------------------------
# VALIDATION
# -----------------------------------------------------------------------------
def compare_legacy(legacy: Path, current: Path, rho_tolerance: float=1e-10, coordinate_tolerance_mm: float=1e-06) -> dict:
    old = pd.read_csv(legacy)
    new = pd.read_csv(current)
    old_rho = [name for name in ('PEC', 'Spearman_Rho', 'rho') if name in old]
    if len(old_rho) != 1 or 'rho' not in new:
        raise PipelineError('Legacy CSV must contain one of PEC/Spearman_Rho/rho; new CSV must contain rho.')
    old_coords = ['X_Coordinate', 'Y_Coordinate', 'Z_Coordinate']
    new_coords = ['x_mm', 'y_mm', 'z_mm']
    if not set(old_coords).issubset(old) or not set(new_coords).issubset(new):
        raise PipelineError('Expected legacy X_Coordinate/Y_Coordinate/Z_Coordinate and new x_mm/y_mm/z_mm columns.')
    if len(old) != len(new):
        raise PipelineError('Different element counts: correlation vectors must not be compared without alignment.')
    if not np.allclose(old[old_coords], new[new_coords], atol=coordinate_tolerance_mm, rtol=0):
        raise PipelineError('Element coordinates/order differ. Resolve mesh alignment before comparing rho.')
    a, b = (old[old_rho[0]].to_numpy(float), new.rho.to_numpy(float))
    finite_a, finite_b = (np.isfinite(a), np.isfinite(b))
    valid = finite_a & finite_b
    max_difference = float(np.max(np.abs(a[valid] - b[valid]))) if valid.any() else None
    status_agrees = np.array_equal(finite_a, finite_b)
    return {'passed': bool(status_agrees and valid.any() and (max_difference <= rho_tolerance)), 'n_elements': len(a), 'n_paired_finite': int(valid.sum()), 'finite_status_agrees': status_agrees, 'max_abs_rho_difference': max_difference, 'rho_tolerance': rho_tolerance, 'scope': 'Rho and coordinate/order comparison only; not validation of effect-size definitions, inference or mesh topology.'}

# -----------------------------------------------------------------------------
# PIPELINE
# -----------------------------------------------------------------------------
LOG = logging.getLogger(__name__)

def checked_inputs(c: dict[str, Any]):
    studies, audit = load_studies(c)
    manifest = resolve_meshes(c, studies)
    return (studies, audit, manifest)

def validate(c: dict[str, Any]) -> dict[str, Any]:
    studies, audit, manifest = checked_inputs(c)
    return {'selected_studies': len(studies), 'excluded_rows': int((~audit.pipeline_selected).sum()), 'head_model_id': c['mesh']['head_model_id'], 'study_ids': studies[c['columns']['study_id']].tolist(), 'mesh_paths_resolved': len(manifest), 'mesh_contents_validated': False, 'note': 'Run extract to check geometry, fields and alignment; inference checks run before statistics.'}

def cached_adjacency(cache: Path, connectivity: np.ndarray):
    path, meta_path = (cache / 'adjacency.npz', cache / 'adjacency_metadata.json')
    if meta_path.is_file():
        if sha256_file(path) != read_json(meta_path)['sha256']:
            raise PipelineError('Adjacency-cache integrity check failed.')
        return sparse.load_npz(path)
    graph = build_adjacency(connectivity)
    sparse.save_npz(path, graph)
    atomic_json(meta_path, {'sha256': sha256_file(path), 'shape': graph.shape, 'definition': 'shared complete tetrahedral face'})
    return graph

def run(c: dict[str, Any], *, stage: str='run', cache_path: Path | None=None, reader: Callable[[Path, str, int], MeshData]=read_scalar_mesh) -> Path:
    if stage not in {'run', 'extract', 'stats'}:
        raise PipelineError(f'Unknown pipeline stage: {stage}')
    studies, audit = load_studies(c)
    if stage != 'extract':
        validate_inference(c)
    if stage != 'extract' and c['export']['mesh']:
        require_simnibs()
    if stage == 'stats':
        if cache_path is None:
            raise PipelineError('stats requires --cache pointing to a completed fields_* directory.')
        cache = cache_path.expanduser().resolve()
        meta, arrays = load_cache(cache)
        if meta['identity']['code'] != code_fingerprint():
            raise PipelineError('Cache was generated by a different code revision. Re-extract with the current code before analysis.')
        if meta['identity']['mesh'] != c['mesh']:
            raise PipelineError('Cached field/tissue/head-model settings differ from the current configuration.')
        manifest = pd.DataFrame([{'Study_ID': s['Study_ID'], 'head_model_id': c['mesh']['head_model_id'], 'mesh_path': s['path']} for s in meta['identity']['sources']])
    else:
        manifest = resolve_meshes(c, studies)
        cache = extract_fields(c, manifest, reader)
        meta, arrays = load_cache(cache)
    if meta['study_ids'] != studies[c['columns']['study_id']].tolist():
        raise PipelineError('Selected study IDs/order differ from this cache. Re-extract the requested study set; do not silently subset ranked data.')
    if stage == 'extract':
        return cache
    s = c['statistics']
    selected_effects = studies[c['columns']['effect_size']].to_numpy(dtype=np.float64)
    environment = versions()
    signature = {'cache': meta['fingerprint'], 'study_ids': meta['study_ids'], 'effects': selected_effects.tolist(), 'seed': s['seed'], 'clusters': c['clusters'], 'source_code': code_fingerprint(), 'statistics_versions': {k: environment[k] for k in ('numpy', 'scipy')}, 'effect_size_definition': s['effect_size_definition'], 'positive_effect_means': s['positive_effect_means'], 'inference': 'random-pairing-abs-rho-v1' if s['n_permutations'] else 'rho-only', 'independent_units': studies[c['columns']['independent_unit']].tolist() if c['columns']['independent_unit'] else None}
    identity = digest(signature)
    parent = c['project']['output_dir'] / c['project']['analysis_name'] / identity[:20]
    output = parent / (f"permutations_{s['n_permutations']}" if s['n_permutations'] else 'rho_only')
    parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(parent / '.run.lock'):
        output.mkdir(parents=True, exist_ok=True)
        provenance = {'status': 'running', 'started_utc': timestamp(), 'signature': signature, 'identity': identity, 'configuration': c, 'versions': environment, 'git': git_state(), 'cache_path': str(cache), 'study_table_sha256': studies.attrs['source_sha256'], 'is_synthetic': meta['identity'].get('source_type') != 'simnibs_mesh'}
        atomic_json(output / 'run_metadata.json', provenance)
        handler = logging.FileHandler(output / 'run.log')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logging.getLogger().addHandler(handler)
        try:
            studies.to_csv(output / 'included_studies.csv', index=False)
            audit.to_csv(output / 'study_selection_audit.csv', index=False)
            manifest.to_csv(output / 'resolved_manifest.csv', index=False)
            fields = arrays['fields']
            block, batch, estimate = working_blocks(s, fields.shape[1], fields.shape[0], c['clusters']['enabled'])
            LOG.info('Analysis shape: %d elements x %d studies. Blocks: %d elements / %d permutations.', *fields.shape, block, batch)
            with threadpool_limits(limits=s['blas_threads'], user_api='blas'):
                with exclusive_lock(cache / '.rank.lock'):
                    ranks, valid = build_rank_cache(cache, fields, block)
                    adjacency = cached_adjacency(cache, arrays['connectivity']) if c['clusters']['enabled'] else None
                y = effect_ranks(selected_effects)
                rho = observed_rho(ranks, y, valid, block)
                result = {'rho': rho, 'valid_correlation': valid.astype(np.uint8)}
                cluster_rows = None
                permutation = None
                threshold = None
                if s['n_permutations']:
                    cluster_args = None
                    if adjacency is not None:
                        threshold = rho_threshold(c['clusters'], fields.shape[1])
                        cluster_args = {'adjacency': adjacency, 'threshold': threshold, 'statistic': c['clusters']['statistic'], 'volumes': arrays['volumes_mm3']}
                        LOG.info('Cluster inference enabled: sign-separated, global map, absolute-rho threshold %.8g; one shuffle per batch.', threshold)
                    permutation = run_permutations(ranks, y, rho, valid, n_permutations=s['n_permutations'], seed=s['seed'], block_size=block, batch_size=batch, checkpoint=parent / 'permutation_checkpoint.npz', identity=identity, cluster_args=cluster_args, checkpoint_every=s['checkpoint_every'])
                    for name in ('p_uncorrected', 'p_fwer_maxstat'):
                        result[name] = permutation[name]
                        result['signed_neglog10_' + name] = signed_neg_log10(result[name], rho)
                    if adjacency is not None:
                        cluster_rows, cluster_maps = observed_clusters(rho, adjacency, threshold, c['clusters']['statistic'], arrays['volumes_mm3'], arrays['coordinates'], arrays['element_ids'], permutation['null_max_cluster'])
                        result.update(cluster_maps)
                    np.savez_compressed(output / 'null_distributions.npz', max_abs_rho=permutation['null_max_abs_rho'], max_cluster=permutation['null_max_cluster'])

            region_rows_uncorrected: list[dict[str, Any]] = []
            region_rows_fwer: list[dict[str, Any]] = []
            if permutation is not None:
                alpha = c['export']['significance_alpha']
                sig_unc, region_unc, region_rows_uncorrected = significant_regions(
                    rho, result['p_uncorrected'], arrays['connectivity'], arrays['volumes_mm3'],
                    arrays['coordinates'], arrays['element_ids'], alpha,
                )
                sig_fwer, region_fwer, region_rows_fwer = significant_regions(
                    rho, result['p_fwer_maxstat'], arrays['connectivity'], arrays['volumes_mm3'],
                    arrays['coordinates'], arrays['element_ids'], alpha,
                )
                result['significant_uncorrected'] = sig_unc
                result['region_id_uncorrected'] = region_unc
                result['significant_fwer'] = sig_fwer
                result['region_id_fwer'] = region_fwer
                write_region_table(output / 'regions_uncorrected.csv', region_rows_uncorrected)
                write_region_table(output / 'regions_FWER.csv', region_rows_fwer)

            export_tables(output, arrays, result, c['export']['csv'], cluster_rows)
            if c['export']['mesh']:
                # One GM-only visualization mesh containing all element-wise analysis fields.
                # No visualization colors are imposed; Gmsh controls display styling.
                mesh_fields: dict[str, np.ndarray] = {
                    'rho': result['rho'],
                }
                if permutation is not None:
                    mesh_fields.update({
                        'p_uncorrected': result['p_uncorrected'],
                        'invertedLogP_uncorrected': result['signed_neglog10_p_uncorrected'],
                        'p_FWER': result['p_fwer_maxstat'],
                        'invertedLogP_FWER': result['signed_neglog10_p_fwer_maxstat'],
                        'significant_uncorrected': result['significant_uncorrected'],
                        'significant_FWER': result['significant_fwer'],
                        'region_uncorrected': result['region_id_uncorrected'],
                        'region_FWER': result['region_id_fwer'],
                    })
                export_mesh(output / 'association_maps.msh', meta, arrays, mesh_fields, c['mesh'])
            summary = {'n_studies': fields.shape[1], 'n_elements': fields.shape[0], 'n_valid_elements': int(valid.sum()), 'n_constant_elements': int((~valid).sum()), 'rho_min': float(np.nanmin(rho)), 'rho_max': float(np.nanmax(rho)), 'n_permutations': s['n_permutations'], 'minimum_attainable_randomized_p': 1 / (s['n_permutations'] + 1) if s['n_permutations'] else None, 'element_block_size': block, 'permutation_batch_size': 1 if adjacency is not None else batch, 'estimated_numerical_workspace_bytes': estimate, 'cluster_rho_threshold': threshold, 'spatial_correction_scope': 'All nonconstant GM tetrahedra in this single head-model/outcome analysis; not across outcomes or head models.', 'inference_unit': 'Selected study effect, not mesh element or repeated head simulation'}
            if permutation is not None:
                alpha = c['export']['significance_alpha']
                summary['significance_alpha'] = alpha
                summary['n_elements_p_uncorrected_le_alpha'] = int(np.count_nonzero(result['p_uncorrected'] <= alpha))
                summary['n_elements_p_fwer_le_alpha'] = int(np.count_nonzero(result['p_fwer_maxstat'] <= alpha))
                summary['n_regions_uncorrected'] = len(region_rows_uncorrected)
                summary['n_regions_fwer'] = len(region_rows_fwer)
                summary['n_clusters_p_fwer_le_0_05'] = sum((row['p_cluster_fwer'] <= 0.05 for row in cluster_rows or []))
            atomic_json(output / 'summary.json', summary)
            provenance.update({'status': 'complete', 'finished_utc': timestamp(), 'summary': summary})
            atomic_json(output / 'run_metadata.json', provenance)
            LOG.info('Completed: %s', output)
        except BaseException as exc:
            provenance.update({'status': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', 'finished_utc': timestamp(), 'error': str(exc)})
            atomic_json(output / 'run_metadata.json', provenance)
            raise
        finally:
            logging.getLogger().removeHandler(handler)
            handler.close()
    return output

# -----------------------------------------------------------------------------
# DEMO
# -----------------------------------------------------------------------------
def synthetic_reader(path: Path, field_name: str, tissue_tag: int) -> MeshData:
    """Test-only adapter. Production extraction always uses SimNIBS mesh_io."""
    with np.load(path, allow_pickle=False) as archive:
        return MeshData(archive['element_ids'], archive['connectivity'], archive['nodes'], archive['values'])

def make_demo(root: Path, n_permutations: int=128, clusters: bool=False) -> Path:
    root = root.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise PipelineError(f'Demo destination is not empty: {root}. Choose a new --output folder.')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'SYNTHETIC_DATA_ONLY.txt').write_text('Software test only. These fields, tetrahedra and effects are artificial; they are not simulations or clinical results.\n')
    rng = np.random.default_rng(173)
    n_elements, n_studies = (256, 12)
    effect = np.array([-0.4, -0.2, -0.2, 0.0, 0.1, 0.3, 0.5, 0.5, 0.7, 0.8, 1.1, 1.2])
    nodes = rng.normal(size=(n_elements + 3, 3)) * 10
    connectivity = np.arange(n_elements)[:, None] + np.arange(4)[None, :] + 1
    fields = 60 + rng.normal(size=(n_elements, n_studies)) * 7
    fields[:30] = 60 + effect * 18 + rng.normal(size=(30, n_studies)) * 2
    fields[30:60] = 60 - effect * 18 + rng.normal(size=(30, n_studies)) * 2
    fields[-1] = 42.0
    rows = []
    for j in range(n_studies):
        study_id = f'demo_{j + 1:02d}'
        folder = root / 'simulations' / (study_id + '_study')
        folder.mkdir(parents=True)
        np.savez(folder / (study_id + '_synthetic.npz'), element_ids=np.arange(1, n_elements + 1), connectivity=connectivity, nodes=nodes, values=fields[:, j])
        rows.append({'Study_ID': study_id, 'Study_name': f'Artificial study {j + 1}', 'effect_size': effect[j], 'include': 'true', 'outcome': 'synthetic', 'independent_unit': f'synthetic_trial_{j + 1}'})
    pd.DataFrame(rows).to_csv(root / 'studies.csv', index=False)
    config = deepcopy(DEFAULTS)
    config['columns'].update(include='include', outcome='outcome', independent_unit='independent_unit')
    config['project'].update(analysis_name='SYNTHETIC_DEMO', cache_dir='cache', output_dir='results')
    config['input'].update(study_table='studies.csv', simulation_root='simulations', mesh_glob='*_synthetic.npz')
    config['selection']['outcome'] = 'synthetic'
    config['statistics'].update(n_permutations=n_permutations, element_block_size=53, permutation_batch_size=17, independent_studies_confirmed=True, effect_size_definition='Artificial numerical values for software testing only', positive_effect_means='Artificial positive direction; not a clinical interpretation')
    config['clusters'].update(enabled=clusters, rho_threshold=0.7)
    config['export']['mesh'] = False
    path = root / 'config.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path

def run_demo(root: Path, n_permutations: int=128, clusters: bool=False) -> Path:
    config = load_config(make_demo(root, n_permutations, clusters))
    result = run(config, reader=synthetic_reader)
    metadata = read_json(result / 'run_metadata.json')
    metadata['is_synthetic'] = True
    atomic_json(result / 'run_metadata.json', metadata)
    (result / 'SYNTHETIC_DATA_ONLY.txt').write_text('Artificial test data. No TMS was simulated. No clinical conclusions may be drawn.\n')
    return result

# -----------------------------------------------------------------------------
# USER-FACING COMMAND LINE
# -----------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 2: extract SimNIBS E-fields, correlate with effect sizes, run permutations, and export maps."
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config.yaml")
    parser.add_argument("--n-permutations", type=int, help="Override statistics.n_permutations from YAML")
    parser.add_argument("--validate-only", action="store_true", help="Check study table and simulation paths only; do not extract/analyze")
    parser.add_argument("--extract-only", action="store_true", help="Extract/cache E-fields only; do not run correlations")
    parser.add_argument("--stats-only", action="store_true", help="Run statistics from an existing --cache directory")
    parser.add_argument("--cache", type=Path, help="Existing fields_* cache directory for --stats-only")
    parser.add_argument("--doctor", action="store_true", help="Print Python/package versions and exit")
    parser.add_argument("--demo", action="store_true", help="Run a synthetic software test; no real TMS data")
    parser.add_argument("--output", type=Path, default=Path("work/demo"), help="Empty output folder for --demo")
    parser.add_argument("--clusters", action="store_true", help="Enable clusters in --demo only; real analysis uses config.yaml")
    parser.add_argument("--compare-legacy", nargs=2, metavar=("OLD_CSV", "NEW_CSV"), help="Compare old rho CSV against a new element_results.csv")
    parser.add_argument("--rho-tolerance", type=float, default=1e-10)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    try:
        if args.doctor:
            print(json.dumps({"interpreter": sys.executable, "versions": versions(),
                              "simnibs_module_present": importlib.util.find_spec("simnibs") is not None}, indent=2))
            return 0

        if args.demo:
            n = 128 if args.n_permutations is None else args.n_permutations
            if n < 1:
                raise PipelineError("Demo requires --n-permutations >= 1.")
            print(run_demo(args.output, n, args.clusters))
            return 0

        if args.compare_legacy:
            comparison = compare_legacy(Path(args.compare_legacy[0]), Path(args.compare_legacy[1]), args.rho_tolerance)
            print(json.dumps(comparison, indent=2))
            return 0 if comparison["passed"] else 1

        config = load_config(args.config)
        if args.n_permutations is not None:
            config["statistics"]["n_permutations"] = args.n_permutations
            validate_config(config)

        modes = sum((args.validate_only, args.extract_only, args.stats_only))
        if modes > 1:
            raise PipelineError("Choose only one of --validate-only, --extract-only, or --stats-only.")
        if args.clusters:
            raise PipelineError("--clusters is only for --demo. For real data set clusters.enabled in config.yaml.")

        if args.validate_only:
            print(json.dumps(validate(config), indent=2, default=json_default))
            return 0
        if args.extract_only:
            print(run(config, stage="extract"))
            return 0
        if args.stats_only:
            if args.cache is None:
                raise PipelineError("--stats-only requires --cache PATH.")
            print(run(config, stage="stats", cache_path=args.cache))
            return 0

        # Default: complete Stage 2 in one command.
        print(run(config, stage="run"))
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Rerun the same command to resume from the last completed permutation checkpoint.", file=sys.stderr)
        return 130
    except (PipelineError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())