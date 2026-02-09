"""
Standardized model reporting for South Africa crop classification.

Generates a timestamped folder per model run containing:
  - report.pdf          Multi-page PDF with all metrics and plots
  - metadata.json       Machine-readable metrics, hyperparams, split info
  - metrics.csv         Top-level metrics (single row)
  - per_class_metrics.csv  Per-class precision/recall/F1/support
  - confusion_matrix.csv   NxN confusion matrix with class labels
  - confusion_matrix.png   Heatmap (300 dpi)
  - feature_importance.png Horizontal bar chart (tree models, optional)
  - training_curves.png    Loss/accuracy over epochs (DL models, optional)
  - predictions.csv        Field-level true vs predicted labels (optional)

Usage:
    from report import ModelReport

    report = ModelReport("XGBoost Field-Level")
    report.set_hyperparameters(best_params)
    report.set_split_info(train=100, val=20, test=30, seed=42)
    report.set_metrics(y_test, y_pred, class_names)
    report.set_predictions(field_ids, y_test, y_pred, class_names)
    report.set_feature_importance(importances, feature_names)
    report.generate()
"""

import os
import re
import json
import inspect
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _numpy_to_python(obj):
    """Recursively convert numpy types to Python natives for JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): _numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_numpy_to_python(v) for v in obj]
    return obj


class ModelReport:
    """Collects model metadata, metrics, and plots, then generates a report folder."""

    def __init__(self, model_name: str, script_path: str = None):
        """
        Args:
            model_name: Human-readable name (e.g. "XGBoost Field-Level").
            script_path: Path to calling script. Auto-detected if None.
        """
        self.model_name = model_name
        self.timestamp = datetime.now()
        self._script_path = script_path or self._detect_script_path()
        self._hyperparameters = {}
        self._split_info = {}
        self._metrics = {}
        self._class_names = None
        self._confusion_matrix = None
        self._per_class_report = None
        self._training_history = None
        self._feature_importance = None
        self._feature_names = None
        self._top_n_features = 20
        self._predictions = None
        self._notes = None

    @staticmethod
    def _detect_script_path():
        """Walk the call stack to find the outermost script."""
        for frame_info in inspect.stack():
            fname = frame_info.filename
            if fname != __file__ and not fname.startswith("<"):
                return os.path.abspath(fname)
        return None

    def _sanitize(self, name: str) -> str:
        return re.sub(r"[^\w\-]", "_", name).strip("_")

    # ---- Builder methods ----

    def set_hyperparameters(self, params: dict) -> "ModelReport":
        self._hyperparameters = dict(params)
        return self

    def set_split_info(
        self,
        train: int,
        val: int = 0,
        test: int = 0,
        seed: int = None,
        split_method: str = "fid-wise",
    ) -> "ModelReport":
        self._split_info = {
            "train_count": train,
            "val_count": val,
            "test_count": test,
            "random_seed": seed,
            "split_method": split_method,
        }
        return self

    def set_metrics(self, y_true, y_pred, class_names=None, level: str = "test") -> "ModelReport":
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        self._class_names = list(class_names) if class_names is not None else None
        self._confusion_matrix = confusion_matrix(y_true, y_pred)
        report_dict = classification_report(
            y_true, y_pred,
            target_names=self._class_names,
            output_dict=True,
            zero_division=0,
        )
        self._per_class_report = {
            k: v for k, v in report_dict.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        }
        self._metrics = {
            "level": level,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_avg": report_dict.get("macro avg", {}),
            "weighted_avg": report_dict.get("weighted avg", {}),
        }
        return self

    def set_predictions(self, field_ids, y_true, y_pred, class_names=None) -> "ModelReport":
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if class_names is not None:
            class_names = list(class_names)
            # If labels are numeric, map to class names
            if np.issubdtype(y_true.dtype, np.integer):
                y_true_labels = [class_names[i] for i in y_true]
                y_pred_labels = [class_names[i] for i in y_pred]
            else:
                y_true_labels = list(y_true)
                y_pred_labels = list(y_pred)
        else:
            y_true_labels = list(y_true)
            y_pred_labels = list(y_pred)
        self._predictions = pd.DataFrame({
            "field_id": list(field_ids),
            "true_label": y_true_labels,
            "predicted_label": y_pred_labels,
        })
        return self

    def set_training_history(self, history: dict) -> "ModelReport":
        self._training_history = {k: list(v) for k, v in history.items()}
        return self

    def set_feature_importance(self, importances, feature_names, top_n: int = 20) -> "ModelReport":
        importances = np.asarray(importances)
        feature_names = list(feature_names)
        idx = np.argsort(importances)[::-1][:top_n]
        self._feature_importance = importances[idx]
        self._feature_names = [feature_names[i] for i in idx]
        self._top_n_features = top_n
        return self

    def add_notes(self, notes: str) -> "ModelReport":
        self._notes = notes
        return self

    # ---- Generation ----

    def generate(self) -> str:
        """Generate all report artifacts. Returns path to the report folder."""
        ts = self.timestamp.strftime("%Y%m%d_%H%M%S")
        folder_name = f"{self._sanitize(self.model_name)}_{ts}"

        # Use REPORTS_DIR from config if available, otherwise default
        try:
            from config import REPORTS_DIR
            base_dir = REPORTS_DIR
        except (ImportError, AttributeError):
            base_dir = os.path.join(os.path.dirname(__file__), "reports")

        report_dir = os.path.join(base_dir, folder_name)
        os.makedirs(report_dir, exist_ok=True)

        # Save CSVs
        self._save_metrics_csv(report_dir)
        self._save_per_class_csv(report_dir)
        self._save_confusion_matrix_csv(report_dir)
        if self._predictions is not None:
            self._predictions.to_csv(os.path.join(report_dir, "predictions.csv"), index=False)

        # Save PNGs and build PDF
        pdf_path = os.path.join(report_dir, "report.pdf")
        with PdfPages(pdf_path) as pdf:
            self._render_summary_page(pdf, report_dir)
            self._render_per_class_table(pdf, report_dir)
            self._render_confusion_matrix(pdf, report_dir)
            if self._training_history:
                self._render_training_curves(pdf, report_dir)
            if self._feature_importance is not None:
                self._render_feature_importance(pdf, report_dir)

        # Save JSON
        self._save_json(report_dir)

        # Auto-register in model registry
        self._register_model(report_dir)

        print(f"[Report] Saved to {report_dir}/")
        return report_dir

    def _register_model(self, report_dir: str):
        """Register this model run in the model registry."""
        try:
            from model_registry import ModelRegistry
            registry = ModelRegistry()
            registry.register(
                name=self.model_name,
                model_type=self._infer_model_type(),
                metrics=self._metrics,
                artifacts={"report_dir": report_dir},
                hyperparameters=self._hyperparameters,
            )
        except Exception as e:
            # Don't fail report generation if registry fails
            print(f"[Registry] Warning: Could not register model: {e}")

    def _infer_model_type(self) -> str:
        """Infer model type from hyperparameters or name."""
        name_lower = self.model_name.lower()
        if "xgboost" in name_lower or "xgb" in name_lower:
            return "xgboost"
        elif "cnn" in name_lower or "lstm" in name_lower:
            return "pytorch"
        elif "tabnet" in name_lower or "tabtransformer" in name_lower:
            return "tabnet"
        elif "3d" in name_lower or "patch" in name_lower:
            return "keras"
        elif "ensemble" in name_lower or "voting" in name_lower or "stacking" in name_lower:
            return "sklearn"
        return "unknown"

    # ---- CSV outputs ----

    def _save_metrics_csv(self, report_dir: str):
        if not self._metrics:
            return
        row = {"model_name": self.model_name}
        row.update({k: v for k, v in self._metrics.items() if k not in ("macro_avg", "weighted_avg", "level")})
        row["level"] = self._metrics.get("level", "test")
        pd.DataFrame([row]).to_csv(os.path.join(report_dir, "metrics.csv"), index=False)

    def _save_per_class_csv(self, report_dir: str):
        if not self._per_class_report:
            return
        rows = []
        for cls, m in self._per_class_report.items():
            rows.append({"class_name": cls, **m})
        pd.DataFrame(rows).to_csv(os.path.join(report_dir, "per_class_metrics.csv"), index=False)

    def _save_confusion_matrix_csv(self, report_dir: str):
        if self._confusion_matrix is None:
            return
        labels = self._class_names or [str(i) for i in range(len(self._confusion_matrix))]
        df = pd.DataFrame(self._confusion_matrix, index=labels, columns=labels)
        df.index.name = "true_label"
        df.to_csv(os.path.join(report_dir, "confusion_matrix.csv"))

    # ---- PDF / PNG rendering ----

    def _render_summary_page(self, pdf: PdfPages, report_dir: str):
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        lines = []
        lines.append(f"Model Report: {self.model_name}")
        lines.append(f"{'=' * 50}")
        lines.append(f"Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        if self._script_path:
            lines.append(f"Script: {self._script_path}")
        lines.append("")

        if self._split_info:
            lines.append("Data Split:")
            for k, v in self._split_info.items():
                if v is not None and v != 0:
                    lines.append(f"  {k}: {v}")
            lines.append("")

        if self._metrics:
            lines.append(f"Metrics ({self._metrics.get('level', 'test')} set):")
            lines.append(f"  Accuracy:     {self._metrics['accuracy']:.4f}")
            lines.append(f"  Cohen Kappa:  {self._metrics['cohen_kappa']:.4f}")
            lines.append(f"  F1 Weighted:  {self._metrics['f1_weighted']:.4f}")
            lines.append(f"  F1 Macro:     {self._metrics['f1_macro']:.4f}")
            lines.append("")

        if self._hyperparameters:
            lines.append("Hyperparameters:")
            for k, v in self._hyperparameters.items():
                lines.append(f"  {k}: {v}")
            lines.append("")

        if self._notes:
            lines.append("Notes:")
            lines.append(f"  {self._notes}")

        text = "\n".join(lines)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", fontfamily="monospace")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _render_per_class_table(self, pdf: PdfPages, report_dir: str):
        if not self._per_class_report:
            return
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.set_title("Per-Class Metrics", fontsize=14, fontweight="bold", pad=20)

        col_labels = ["Class", "Precision", "Recall", "F1-Score", "Support"]
        table_data = []
        for cls, m in self._per_class_report.items():
            table_data.append([
                cls,
                f"{m['precision']:.3f}",
                f"{m['recall']:.3f}",
                f"{m['f1-score']:.3f}",
                f"{int(m['support'])}",
            ])

        table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.5)

        # Style header
        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#4472C4")
            table[0, j].set_text_props(color="white", fontweight="bold")

        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            color = "#D9E2F3" if i % 2 == 0 else "white"
            for j in range(len(col_labels)):
                table[i, j].set_facecolor(color)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _render_confusion_matrix(self, pdf: PdfPages, report_dir: str):
        if self._confusion_matrix is None:
            return
        cm = self._confusion_matrix
        labels = self._class_names or [str(i) for i in range(len(cm))]
        n = len(labels)

        fig, ax = plt.subplots(figsize=(max(8, n * 1.1), max(6, n * 0.9)))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        thresh = cm.max() / 2.0
        for i in range(n):
            for j in range(n):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > thresh else "black")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        fig.tight_layout()

        # Save PNG
        fig.savefig(os.path.join(report_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
        pdf.savefig(fig)
        plt.close(fig)

    def _render_training_curves(self, pdf: PdfPages, report_dir: str):
        if not self._training_history:
            return
        history = self._training_history
        has_loss = "loss" in history
        has_acc = any(k in history for k in ("accuracy", "acc"))

        n_plots = int(has_loss) + int(has_acc)
        if n_plots == 0:
            return

        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        plot_idx = 0
        if has_loss:
            ax = axes[plot_idx]
            ax.plot(history["loss"], label="Train Loss")
            if "val_loss" in history:
                ax.plot(history["val_loss"], label="Val Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        if has_acc:
            acc_key = "accuracy" if "accuracy" in history else "acc"
            ax = axes[plot_idx]
            ax.plot(history[acc_key], label="Train Accuracy")
            val_acc_key = "val_accuracy" if "val_accuracy" in history else "val_acc"
            if val_acc_key in history:
                ax.plot(history[val_acc_key], label="Val Accuracy")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title("Training Accuracy")
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle("Training Curves", fontsize=14, fontweight="bold")
        fig.tight_layout()

        fig.savefig(os.path.join(report_dir, "training_curves.png"), dpi=300, bbox_inches="tight")
        pdf.savefig(fig)
        plt.close(fig)

    def _render_feature_importance(self, pdf: PdfPages, report_dir: str):
        if self._feature_importance is None:
            return
        fig, ax = plt.subplots(figsize=(8, max(5, len(self._feature_names) * 0.35)))
        y_pos = np.arange(len(self._feature_names))
        ax.barh(y_pos, self._feature_importance[::-1], color="#4472C4")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(self._feature_names[::-1], fontsize=9)
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {len(self._feature_names)} Feature Importances",
                     fontsize=14, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()

        fig.savefig(os.path.join(report_dir, "feature_importance.png"), dpi=300, bbox_inches="tight")
        pdf.savefig(fig)
        plt.close(fig)

    # ---- JSON output ----

    def _save_json(self, report_dir: str):
        payload = {
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat(),
            "script_path": self._script_path,
            "hyperparameters": self._hyperparameters,
            "split_info": self._split_info,
            "metrics": self._metrics,
            "class_names": self._class_names,
            "confusion_matrix": self._confusion_matrix.tolist() if self._confusion_matrix is not None else None,
            "per_class": self._per_class_report,
            "training_history": self._training_history,
            "feature_importance": {
                "feature_names": self._feature_names,
                "importance_values": self._feature_importance.tolist(),
            } if self._feature_importance is not None else None,
            "notes": self._notes,
            "pdf_path": os.path.join(report_dir, "report.pdf"),
        }
        payload = _numpy_to_python(payload)
        with open(os.path.join(report_dir, "metadata.json"), "w") as f:
            json.dump(payload, f, indent=2)
