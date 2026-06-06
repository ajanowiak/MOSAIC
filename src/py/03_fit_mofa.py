# 03_fit_mofa.py
"""
Fit MOFA+ on all preprocessed multi-omics views and extract factor scores and loadings.

Pipeline stage 3 (MOFA fitting). Reads z-scored matrices from results/preprocessing/.
Fits one model per random seed, selects the best by ELBO, drops inactive factors by ARD
threshold, and writes factor scores, loadings, R² tables, HDF5 models, and diagnostic figures.
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mofapy2.run.entry_point import entry_point

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.io import load_config, load_csv, save_csv
from utils.plotting import savefig, set_style

VIEW_NAMES = ["rna", "proteomics", "phospho"]


def make_mofa_feature_names(feature_names: dict[str, list[str]]) -> list[list[str]]:
    """Prefix feature names by view so MOFA receives globally unique identifiers."""
    return [
        [f"{view}:{name}" for name in feature_names[view]] for view in VIEW_NAMES
    ]


def build_mofa_data(
    rna: pd.DataFrame, proteomics: pd.DataFrame, phospho: pd.DataFrame
) -> list:
    """Build a fresh MOFA data matrix for one fit (mofapy2 mutates this in place)."""
    return [
        [rna.values.copy()],
        [proteomics.values.copy()],
        [phospho.values.copy()],
    ]


def parse_arguments():
    """Parse CLI arguments for the MOFA fitting script."""
    parser = argparse.ArgumentParser(
        description="Fit MOFA+ models for the MOSAIC pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config/config.yml.",
    )
    parser.add_argument(
        "--skip-fit",
        action="store_true",
        default=False,
        help="Skip MOFA fitting; reload best seed from existing HDF5 and re-export outputs.",
    )
    return parser.parse_args()


def setup_logging(log_path: Path) -> logging.Logger:
    """Configure logging to stdout and a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mosaic.fit_mofa")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def resolve_preprocessing_paths(config: dict) -> dict[str, Path]:
    """Build paths to preprocessed omics matrices."""
    results_dir = Path(config["paths"]["results"])
    preprocess_dir = results_dir / "preprocessing"
    return {
        "rna": preprocess_dir / "rna_preprocessed.csv",
        "proteomics": preprocess_dir / "proteomics_preprocessed.csv",
        "phospho": preprocess_dir / "phospho_preprocessed.csv",
    }


def load_preprocessed_data(
    input_paths: dict[str, Path], logger: logging.Logger
) -> dict[str, pd.DataFrame]:
    """Load and validate preprocessed omics matrices."""
    data = {}
    for view, path in input_paths.items():
        if not path.exists():
            raise FileNotFoundError(str(path.resolve()))
        data[view] = load_csv(str(path))
        logger.info("Loaded %s: shape %s", view, data[view].shape)

    indices = [data[view].index for view in VIEW_NAMES]
    if not all(idx.equals(indices[0]) for idx in indices[1:]):
        raise ValueError("Preprocessed matrices do not share identical sample indices.")

    return data


def extract_final_elbo(ent: entry_point) -> float:
    """Extract the final ELBO from a trained MOFA entry point."""
    model = ent.model
    for attr in ("training_stats", "train_stats"):
        stats = getattr(model, attr, None)
        if stats is None:
            continue
        for key in ("ELBO", "elbo"):
            if key in stats and len(stats[key]) > 0:
                return float(stats[key][-1])

    try:
        elbo_series = model.compute_ELBO()
        return float(elbo_series)
    except AttributeError:
        pass

    try:
        elbo_series = model.calculateELBO()
        if hasattr(elbo_series, "get"):
            return float(elbo_series["total"])
        return float(elbo_series)
    except (AttributeError, TypeError, KeyError):
        pass

    return float("nan")


class _ExpectationNode:
    """Minimal node wrapper exposing getExpectation() for HDF5-loaded arrays."""

    def __init__(self, value):
        self._value = value

    def getExpectation(self):
        return self._value


class _LoadedMofaModel:
    """Minimal in-memory model reconstructed from a saved MOFA HDF5 file."""

    def __init__(self, z_arr: np.ndarray, w_list: list[np.ndarray]):
        self.nodes = {
            "Z": _ExpectationNode(z_arr),
            "W": _ExpectationNode(w_list),
        }


class _LoadedEntryPoint:
    """Entry-point wrapper around a loaded MOFA model."""

    def __init__(self, model: _LoadedMofaModel):
        self.model = model


def _read_h5_array(h5_file: h5py.File, candidate_paths: list[str]) -> np.ndarray:
    """Read the first available dataset from a list of HDF5 paths."""
    for path in candidate_paths:
        if path in h5_file:
            return np.array(h5_file[path])
    raise KeyError(f"None of the HDF5 paths exist: {candidate_paths}")


def load_best_seed_from_elbo(out_dir: Path) -> tuple[int, float, pd.DataFrame]:
    """Select the best seed from a saved model_elbo.csv table."""
    elbo_path = out_dir / "model_elbo.csv"
    if not elbo_path.exists():
        raise FileNotFoundError(str(elbo_path.resolve()))

    elbo_df = pd.read_csv(elbo_path)
    elbo_col = "final_elbo" if "final_elbo" in elbo_df.columns else "elbo"
    best_row = elbo_df.loc[elbo_df[elbo_col].idxmax()]
    return int(best_row["seed"]), float(best_row[elbo_col]), elbo_df


def load_model_from_hdf5(
    hdf5_path: Path,
    logger: logging.Logger,
) -> _LoadedEntryPoint:
    """Load factor scores and loadings from a MOFA HDF5 model file."""
    if not hdf5_path.exists():
        raise FileNotFoundError(str(hdf5_path.resolve()))

    with h5py.File(hdf5_path, "r") as h5_file:
        z_raw = _read_h5_array(
            h5_file,
            ["expectations/Z/all", "expectations/Z/group0", "expectations/Z/group1"],
        )
        logger.info("HDF5 Z raw shape: %s", z_raw.shape)
        if z_raw.shape[0] < z_raw.shape[1]:
            z_arr = z_raw.T
        else:
            z_arr = z_raw
        logger.info("HDF5 Z array shape after orienting: %s", z_arr.shape)

        w_list = []
        for view in VIEW_NAMES:
            w_raw = _read_h5_array(
                h5_file,
                [
                    f"expectations/W/{view}",
                    f"expectations/W/{view}/all",
                    f"expectations/W/{view}/group0",
                ],
            )
            logger.info("HDF5 W/%s raw shape: %s", view, w_raw.shape)
            if w_raw.shape[0] < w_raw.shape[1]:
                w_view = w_raw.T
            else:
                w_view = w_raw
            logger.info("HDF5 W/%s array shape after orienting: %s", view, w_view.shape)
            w_list.append(w_view)

    model = _LoadedMofaModel(z_arr=z_arr, w_list=w_list)
    return _LoadedEntryPoint(model)


def fit_model_for_seed(
    seed: int,
    data: list,
    samples_names: list,
    features_names: list,
    likelihoods: list[str],
    mofa_cfg: dict,
    out_dir: Path,
    logger: logging.Logger,
) -> dict:
    """Fit a single MOFA+ model for one random seed."""
    ent = entry_point()
    ent.set_data_matrix(
        data=data,
        likelihoods=likelihoods,
        views_names=VIEW_NAMES,
        groups_names=["all"],
        samples_names=samples_names,
        features_names=features_names,
    )
    ent.set_model_options(factors=mofa_cfg["n_factors"])
    ent.set_train_options(
        iter=mofa_cfg["max_iter"],
        convergence_mode=mofa_cfg["convergence_mode"],
        seed=seed,
        verbose=False,
        outfile=str(out_dir / f"model_seed{seed}.hdf5"),
    )
    ent.build()
    ent.run()
    model_path = out_dir / f"model_seed{seed}.hdf5"
    ent.save(str(model_path))
    logger.info("Saved model %s", model_path)

    elbo = extract_final_elbo(ent)
    logger.info("Seed %d: ELBO = %.4f", seed, elbo)
    return {"seed": seed, "elbo": elbo, "ent": ent}


def extract_factor_scores(
    ent: entry_point, sample_index: pd.Index, n_factors: int
) -> pd.DataFrame:
    """Extract sample factor scores Z from the best MOFA model."""
    z_raw = ent.model.nodes["Z"].getExpectation()
    if np.ndim(z_raw) == 3:
        z_arr = z_raw[0]
    elif np.ndim(z_raw) == 2:
        z_arr = z_raw
    else:
        raise ValueError(f"Unexpected Z shape ndim={np.ndim(z_raw)}")

    if z_arr.shape != (len(sample_index), n_factors):
        raise ValueError(
            f"Z shape {z_arr.shape} != expected ({len(sample_index)}, {n_factors})"
        )

    factor_cols = [f"Factor{k + 1}" for k in range(n_factors)]
    return pd.DataFrame(z_arr, index=sample_index, columns=factor_cols)


def orient_weight_matrix(
    w_view: np.ndarray, n_features: int, n_factors: int, view_name: str
) -> np.ndarray:
    """Ensure weight matrix has shape (n_features, n_factors)."""
    if w_view.shape == (n_features, n_factors):
        return w_view
    if w_view.shape == (n_factors, n_features):
        return w_view.T
    raise ValueError(
        f"Unexpected W shape for {view_name}: {w_view.shape}; "
        f"expected ({n_features}, {n_factors}) or ({n_factors}, {n_features})"
    )


def extract_factor_loadings(
    ent: entry_point, feature_names: dict[str, list[str]], n_factors: int
) -> dict[str, pd.DataFrame]:
    """Extract per-view factor loadings W from the best MOFA model."""
    w_raw_list = ent.model.nodes["W"].getExpectation()
    factor_cols = [f"Factor{k + 1}" for k in range(n_factors)]
    loadings = {}

    for view_idx, view_name in enumerate(VIEW_NAMES):
        w_view = orient_weight_matrix(
            np.asarray(w_raw_list[view_idx]),
            len(feature_names[view_name]),
            n_factors,
            view_name,
        )
        loadings[view_name] = pd.DataFrame(
            w_view,
            index=feature_names[view_name],
            columns=factor_cols,
        )

    return loadings


def compute_r2_per_factor_per_view(
    ent: entry_point,
    data_arrays: dict[str, np.ndarray],
    z_arr: np.ndarray,
    n_factors: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Compute variance explained per factor per view using MOFA or a fallback approximation."""
    factor_cols = [f"Factor{k + 1}" for k in range(n_factors)]

    try:
        r2_raw = ent.model.calculate_variance_explained()
        logger.info(
            "calculate_variance_explained() returned type %s", type(r2_raw).__name__
        )

        if isinstance(r2_raw, dict):
            logger.info("Top-level keys: %s", list(r2_raw.keys()))
            for key, value in r2_raw.items():
                if isinstance(value, dict):
                    logger.info(
                        "  %s: nested keys %s", key, list(value.keys())
                    )
                    for sub_key, sub_val in value.items():
                        logger.info(
                            "    %s[%s] shape: %s",
                            key,
                            sub_key,
                            getattr(sub_val, "shape", type(sub_val)),
                        )
                else:
                    logger.info(
                        "  %s shape: %s", key, getattr(value, "shape", type(value))
                    )

            if "r2_per_factor" in r2_raw:
                group_key = next(iter(r2_raw["r2_per_factor"]))
                r2_arr = np.asarray(r2_raw["r2_per_factor"][group_key])
            else:
                raise ValueError("Unexpected dict structure from calculate_variance_explained()")
        elif isinstance(r2_raw, list):
            logger.info("List length: %d", len(r2_raw))
            for idx, item in enumerate(r2_raw):
                logger.info("  r2_raw[%d] shape: %s", idx, np.asarray(item).shape)
            r2_arr = np.asarray(r2_raw[0])
        else:
            r2_arr = np.asarray(r2_raw)
            logger.info("Array shape: %s", r2_arr.shape)

        if r2_arr.shape == (len(VIEW_NAMES), n_factors):
            r2_matrix = r2_arr.T
        elif r2_arr.shape == (n_factors, len(VIEW_NAMES)):
            r2_matrix = r2_arr
        else:
            raise ValueError(f"Unexpected R2 array shape: {r2_arr.shape}")

        return pd.DataFrame(r2_matrix, index=factor_cols, columns=VIEW_NAMES)

    except (AttributeError, ValueError, KeyError, TypeError) as exc:
        logger.warning(
            "calculate_variance_explained() failed (%s); using fallback approximation.",
            exc,
        )
        logger.warning(
            "Fallback R² assumes orthogonal factors and may be inaccurate."
        )

    w_raw_list = ent.model.nodes["W"].getExpectation()
    r2_matrix = np.zeros((n_factors, len(VIEW_NAMES)))

    for view_idx, view_name in enumerate(VIEW_NAMES):
        y_v = data_arrays[view_name]
        n_features = y_v.shape[1]
        w_v = orient_weight_matrix(
            np.asarray(w_raw_list[view_idx]),
            n_features,
            n_factors,
            view_name,
        )
        total_var = np.var(y_v, axis=0, ddof=1).sum()
        for k in range(n_factors):
            recon_k = np.outer(z_arr[:, k], w_v[:, k])
            r2_matrix[k, view_idx] = np.var(recon_k, axis=0, ddof=1).sum() / total_var

    return pd.DataFrame(r2_matrix, index=factor_cols, columns=VIEW_NAMES)


def drop_inactive_factors(
    factor_scores: pd.DataFrame,
    loadings: dict[str, pd.DataFrame],
    r2_df: pd.DataFrame,
    threshold: float,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, pd.Index]:
    """Drop factors whose maximum per-view R² falls below the ARD threshold."""
    n_factors = factor_scores.shape[1]
    max_r2_per_factor = r2_df.max(axis=1)
    retained_labels = max_r2_per_factor[max_r2_per_factor >= threshold].index
    logger.info(
        "Retained %d/%d factors (max per-view R² >= %s)",
        len(retained_labels),
        n_factors,
        threshold,
    )

    factor_scores = factor_scores.loc[:, retained_labels]
    loadings = {
        view: df.loc[:, retained_labels] for view, df in loadings.items()
    }
    r2_df = r2_df.loc[retained_labels, :]
    return factor_scores, loadings, r2_df, retained_labels


def plot_variance_explained_heatmap(
    r2_df: pd.DataFrame,
    threshold: float,
    n_total: int,
    output_path: Path,
    dpi: int,
) -> None:
    """Plot per-view variance explained as a heatmap for retained factors."""
    set_style()
    n_retained = len(r2_df.index)
    fig_height = n_retained * 0.45 + 1.5
    fig, ax = plt.subplots(figsize=(5, fig_height))

    try:
        import seaborn as sns

        sns.heatmap(
            r2_df,
            cmap="YlOrRd",
            vmin=0,
            annot=True,
            fmt=".1%",
            linewidths=0.5,
            square=False,
            cbar_kws={"label": "Fraction of variance explained (per view)"},
            ax=ax,
        )
    except ImportError:
        values = r2_df.values
        im = ax.imshow(values, aspect="auto", cmap="YlOrRd", vmin=0)
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                ax.text(
                    col,
                    row,
                    f"{values[row, col]:.1%}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        ax.set_xticks(np.arange(len(VIEW_NAMES)))
        ax.set_xticklabels(VIEW_NAMES)
        ax.set_yticks(np.arange(n_retained))
        ax.set_yticklabels(r2_df.index)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Fraction of variance explained (per view)")

    ax.set_title(
        "MOFA+ variance explained per factor per view\n"
        f"(ARD threshold: max per-view R² ≥ {threshold:.0%}, "
        f"{n_retained}/{n_total} factors retained)"
    )
    ax.set_xlabel("Omics view")
    ax.set_ylabel("Factor")
    savefig(fig, str(output_path), dpi=dpi)


def extract_elbo_history(ent) -> list[float]:
    """Extract ELBO history from a trained model if available."""
    if ent is None or not hasattr(ent, "model"):
        return []
    model = ent.model
    for attr in ("training_stats", "train_stats"):
        stats = getattr(model, attr, None)
        if stats is None:
            continue
        for key in ("ELBO", "elbo"):
            if key in stats:
                values = list(np.asarray(stats[key]).astype(float))
                if values:
                    return values
    return []


def plot_elbo_convergence(
    fit_results: list[dict],
    best_seed: int,
    output_path: Path,
    dpi: int,
    logger: logging.Logger,
) -> None:
    """Plot ELBO convergence curves or final ELBO bar chart per seed."""
    set_style()
    histories = {result["seed"]: extract_elbo_history(result["ent"]) for result in fit_results}
    multi_point = any(len(history) > 1 for history in histories.values())

    fig, ax = plt.subplots(figsize=(10, 4))

    if multi_point:
        for result in fit_results:
            seed = result["seed"]
            history = histories[seed]
            if len(history) <= 1:
                continue
            color = "red" if seed == best_seed else None
            label = f"seed {seed}" + (" (best)" if seed == best_seed else "")
            ax.plot(np.arange(len(history)), history, label=label, color=color)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("ELBO")
    else:
        seeds = [result["seed"] for result in fit_results]
        elbos = [result["elbo"] for result in fit_results]
        colors = ["red" if seed == best_seed else "#4C72B0" for seed in seeds]
        ax.bar([str(seed) for seed in seeds], elbos, color=colors)
        ax.set_xlabel("Seed")
        ax.set_ylabel("Final ELBO")
        logger.info("Only scalar ELBO values available; plotted bar chart.")

    ax.set_title("MOFA+ — ELBO per seed")
    ax.legend()
    savefig(fig, str(output_path), dpi=dpi)


def select_best_model(fit_results: list[dict], logger: logging.Logger) -> dict:
    """Select the fit result with the highest final ELBO."""
    valid = [result for result in fit_results if not np.isnan(result["elbo"])]
    if valid:
        best = max(valid, key=lambda item: item["elbo"])
    else:
        best = fit_results[-1]
        logger.warning("All seeds returned NaN ELBO; using last seed %d.", best["seed"])

    logger.info(
        "Best model: seed %d with ELBO = %.4f",
        best["seed"],
        best["elbo"],
    )
    return best


def main():
    """Run MOFA fitting, extraction, and output generation."""
    args = parse_arguments()
    repo_root = Path(__file__).resolve().parents[2]
    log_path = repo_root / "log" / "03_fit_mofa.log"
    logger = setup_logging(log_path)

    config = load_config(args.config)
    mofa_cfg = config["mofa"]
    figures_cfg = config["figures"]
    results_dir = repo_root / config["paths"]["results"]
    figures_dir = repo_root / config["paths"]["figures"]
    out_dir = results_dir / "mofa"
    fig_dir = figures_dir / "mofa"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    input_paths = resolve_preprocessing_paths(config)
    omics = load_preprocessed_data(input_paths, logger)

    rna = omics["rna"]
    proteomics = omics["proteomics"]
    phospho = omics["phospho"]

    samples_names = [rna.index.tolist()]
    original_feature_names = {
        "rna": rna.columns.tolist(),
        "proteomics": proteomics.columns.tolist(),
        "phospho": phospho.columns.tolist(),
    }
    features_names = make_mofa_feature_names(original_feature_names)
    likelihoods = list(mofa_cfg["likelihoods"].values())

    if args.skip_fit:
        best_seed, best_elbo, elbo_df = load_best_seed_from_elbo(out_dir)
        elbo_col = "final_elbo" if "final_elbo" in elbo_df.columns else "elbo"
        logger.info(
            "Skipping fit — loading best model from HDF5 (seed=%d)",
            best_seed,
        )
        hdf5_path = out_dir / f"model_seed{best_seed}.hdf5"
        ent_best = load_model_from_hdf5(hdf5_path, logger)
        best = {"seed": best_seed, "elbo": best_elbo, "ent": ent_best}
        fit_results = [
            {
                "seed": int(row["seed"]),
                "elbo": float(row[elbo_col]),
                "ent": ent_best if int(row["seed"]) == best_seed else None,
            }
            for _, row in elbo_df.iterrows()
        ]
    else:
        fit_results = []
        for seed in mofa_cfg["random_seeds"]:
            logger.info("Fitting MOFA model with seed %d", seed)
            fit_results.append(
                fit_model_for_seed(
                    seed=seed,
                    data=build_mofa_data(rna, proteomics, phospho),
                    samples_names=samples_names,
                    features_names=features_names,
                    likelihoods=likelihoods,
                    mofa_cfg=mofa_cfg,
                    out_dir=out_dir,
                    logger=logger,
                )
            )

        best = select_best_model(fit_results, logger)
        ent_best = best["ent"]
        elbo_df = pd.DataFrame(
            [{"seed": result["seed"], "elbo": result["elbo"]} for result in fit_results]
        ).sort_values("seed")

    n_factors = mofa_cfg["n_factors"]

    factor_scores = extract_factor_scores(ent_best, rna.index, n_factors)
    loadings = extract_factor_loadings(
        ent_best, original_feature_names, n_factors
    )

    z_arr = factor_scores.values
    data_arrays = {
        "rna": rna.values,
        "proteomics": proteomics.values,
        "phospho": phospho.values,
    }
    r2_df = compute_r2_per_factor_per_view(
        ent_best, data_arrays, z_arr, n_factors, logger
    )

    factor_scores, loadings, r2_df, _ = drop_inactive_factors(
        factor_scores,
        loadings,
        r2_df,
        mofa_cfg["ard_r2_threshold"],
        logger,
    )

    save_csv(elbo_df, str(out_dir / "model_elbo.csv"), index=False)
    save_csv(factor_scores, str(out_dir / "factor_scores.csv"), index=True)
    save_csv(loadings["rna"], str(out_dir / "factor_loadings_rna.csv"), index=True)
    save_csv(
        loadings["proteomics"],
        str(out_dir / "factor_loadings_proteomics.csv"),
        index=True,
    )
    save_csv(
        loadings["phospho"],
        str(out_dir / "factor_loadings_phospho.csv"),
        index=True,
    )
    save_csv(r2_df, str(out_dir / "r2_per_factor_per_view.csv"), index=True)

    plot_variance_explained_heatmap(
        r2_df,
        mofa_cfg["ard_r2_threshold"],
        n_factors,
        fig_dir / "variance_explained_heatmap.pdf",
        figures_cfg["dpi"],
    )
    plot_elbo_convergence(
        fit_results,
        best["seed"],
        fig_dir / "elbo_convergence.pdf",
        figures_cfg["dpi"],
        logger,
    )

    logger.info("MOFA fitting complete.")


if __name__ == "__main__":
    main()
