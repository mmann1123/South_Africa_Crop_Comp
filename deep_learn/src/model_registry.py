"""
Model Registry - Track trained models and their artifacts.

Provides a central place to register, list, and load trained models.
Each model run is logged with metadata including:
- Model name and type
- Training date/time
- Metrics (accuracy, kappa, F1)
- Artifact paths
- Hyperparameters

Usage:
    from model_registry import ModelRegistry

    # Register a new model after training
    registry = ModelRegistry()
    registry.register(
        name="CNN-BiLSTM",
        model_type="pytorch",
        metrics={"accuracy": 0.84, "kappa": 0.77},
        artifacts={
            "weights": ["models/new_model_seed_0_25epochs.pt", ...],
            "label_encoder": "models/cnn_bilstm_label_encoder.pkl",
        },
        hyperparameters={"epochs": 25, "lr": 1e-3, ...},
    )

    # List all registered models
    registry.list_models()

    # Get best model by metric
    best = registry.get_best("CNN-BiLSTM", metric="kappa")
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

REGISTRY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model_registry.json"
)


class ModelRegistry:
    """Track trained models and their artifacts."""

    def __init__(self, registry_file: str = REGISTRY_FILE):
        self.registry_file = registry_file
        self.models = self._load()

    def _load(self) -> Dict:
        """Load registry from disk."""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, "r") as f:
                return json.load(f)
        return {"models": []}

    def _save(self):
        """Save registry to disk."""
        with open(self.registry_file, "w") as f:
            json.dump(self.models, f, indent=2, default=str)

    def register(
        self,
        name: str,
        model_type: str,
        metrics: Dict[str, float],
        artifacts: Dict[str, Any],
        hyperparameters: Optional[Dict] = None,
        notes: Optional[str] = None,
    ) -> str:
        """
        Register a trained model.

        Args:
            name: Model name (e.g., "CNN-BiLSTM", "XGBoost")
            model_type: Type (e.g., "pytorch", "sklearn", "keras", "tabnet")
            metrics: Dict of metric name -> value
            artifacts: Dict of artifact type -> path(s)
            hyperparameters: Optional hyperparameters used
            notes: Optional notes about this run

        Returns:
            run_id: Unique identifier for this run
        """
        timestamp = datetime.now()
        run_id = f"{name.lower().replace(' ', '_')}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        entry = {
            "run_id": run_id,
            "name": name,
            "model_type": model_type,
            "timestamp": timestamp.isoformat(),
            "metrics": metrics,
            "artifacts": artifacts,
            "hyperparameters": hyperparameters or {},
            "notes": notes,
        }

        self.models["models"].append(entry)
        self._save()

        print(f"[Registry] Registered: {run_id}")
        print(f"  Metrics: {metrics}")
        return run_id

    def list_models(self, name: Optional[str] = None) -> List[Dict]:
        """
        List all registered models.

        Args:
            name: Filter by model name (optional)

        Returns:
            List of model entries
        """
        models = self.models["models"]
        if name:
            models = [m for m in models if m["name"] == name]

        if not models:
            print("No models registered.")
            return []

        print(f"\n{'='*70}")
        print("REGISTERED MODELS")
        print(f"{'='*70}")

        for m in models:
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in m["metrics"].items())
            print(f"\n{m['run_id']}")
            print(f"  Type: {m['model_type']}")
            print(f"  Date: {m['timestamp'][:19]}")
            print(f"  Metrics: {metrics_str}")
            if m.get("notes"):
                print(f"  Notes: {m['notes']}")

        return models

    def get_best(
        self,
        name: str,
        metric: str = "kappa",
        higher_is_better: bool = True,
    ) -> Optional[Dict]:
        """
        Get the best model run by a specific metric.

        Args:
            name: Model name to filter by
            metric: Metric to optimize
            higher_is_better: Whether higher values are better

        Returns:
            Best model entry or None
        """
        models = [m for m in self.models["models"] if m["name"] == name]
        if not models:
            return None

        models = [m for m in models if metric in m["metrics"]]
        if not models:
            return None

        if higher_is_better:
            best = max(models, key=lambda m: m["metrics"][metric])
        else:
            best = min(models, key=lambda m: m["metrics"][metric])

        return best

    def get_latest(self, name: Optional[str] = None) -> Optional[Dict]:
        """Get the most recent model run."""
        models = self.models["models"]
        if name:
            models = [m for m in models if m["name"] == name]
        if not models:
            return None
        return models[-1]

    def get_artifacts(self, run_id: str) -> Optional[Dict]:
        """Get artifact paths for a specific run."""
        for m in self.models["models"]:
            if m["run_id"] == run_id:
                return m["artifacts"]
        return None

    def check_artifacts_exist(self, run_id: str) -> Dict[str, bool]:
        """Check if all artifacts for a run still exist on disk."""
        artifacts = self.get_artifacts(run_id)
        if not artifacts:
            return {}

        status = {}
        for artifact_type, paths in artifacts.items():
            if isinstance(paths, list):
                status[artifact_type] = all(os.path.exists(p) for p in paths)
            else:
                status[artifact_type] = os.path.exists(paths)
        return status


def register_from_report(report_dir: str) -> Optional[str]:
    """
    Register a model from an existing report directory.

    Args:
        report_dir: Path to report directory containing metadata.json

    Returns:
        run_id if successful, None otherwise
    """
    metadata_path = os.path.join(report_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"No metadata.json found in {report_dir}")
        return None

    with open(metadata_path) as f:
        metadata = json.load(f)

    registry = ModelRegistry()
    run_id = registry.register(
        name=metadata.get("model_name", "Unknown"),
        model_type=metadata.get("model_type", "unknown"),
        metrics=metadata.get("metrics", {}),
        artifacts={"report_dir": report_dir},
        hyperparameters=metadata.get("hyperparameters", {}),
        notes=f"Imported from {report_dir}",
    )
    return run_id


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model Registry CLI")
    parser.add_argument("command", choices=["list", "best", "import-reports"],
                        help="Command to run")
    parser.add_argument("--name", help="Filter by model name")
    parser.add_argument("--metric", default="kappa", help="Metric for 'best' command")
    args = parser.parse_args()

    registry = ModelRegistry()

    if args.command == "list":
        registry.list_models(name=args.name)

    elif args.command == "best":
        if not args.name:
            print("--name required for 'best' command")
        else:
            best = registry.get_best(args.name, metric=args.metric)
            if best:
                print(f"\nBest {args.name} by {args.metric}:")
                print(f"  Run ID: {best['run_id']}")
                print(f"  {args.metric}: {best['metrics'].get(args.metric, 'N/A')}")
                print(f"  Artifacts: {best['artifacts']}")
            else:
                print(f"No {args.name} models found with {args.metric} metric")

    elif args.command == "import-reports":
        reports_dir = os.path.join(os.path.dirname(__file__), "reports")
        if os.path.exists(reports_dir):
            for report_name in os.listdir(reports_dir):
                report_path = os.path.join(reports_dir, report_name)
                if os.path.isdir(report_path):
                    register_from_report(report_path)
