import json
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from matplotlib.backends.backend_pdf import PdfPages

TMP_DIR = Path("/tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)


def plot_equity_curve(nav_series: List[float], output_path: Path):
    """繪製權益曲線。"""
    plt.figure()
    plt.plot(nav_series)
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("NAV")
    plt.savefig(output_path)
    plt.close()


def plot_drawdown(drawdowns: List[float], output_path: Path):
    """繪製回撤曲線。"""
    plt.figure()
    plt.plot(drawdowns)
    plt.title("Drawdown")
    plt.xlabel("Time")
    plt.ylabel("Drawdown")
    plt.savefig(output_path)
    plt.close()


def plot_return_histogram(returns: List[float], output_path: Path):
    """繪製報酬分布直方圖。"""
    plt.figure()
    plt.hist(returns, bins=50)
    plt.title("Return Histogram")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.savefig(output_path)
    plt.close()


class ReportGen:
    def __init__(
        self,
        run_id: str,
        results: Dict[str, Any],
        strategy_name: str = "Strategy",
        hyperparams: Dict[str, Any] = None,
    ):
        if hyperparams is None:
            hyperparams = {}
        self.run_id = run_id
        self.results = results
        self.strategy_name = strategy_name
        self.hyperparams = hyperparams

    def generate_json(self) -> str:
        metrics = self.results.get("metrics", {})
        cpcv_slices = self.results.get("slices", [])

        report_data = {
            "run_id": self.run_id,
            "strategy": self.strategy_name,
            "hyperparams": self.hyperparams,
            "metrics": {
                "total_return": metrics.get("total_return", 0),
                "sharpe": metrics.get("sharpe", 0),
                "sortino": metrics.get("sortino", 0),
                "max_drawdown": metrics.get("max_drawdown", 0),
                "var_95": metrics.get("var_95", 0),
            },
            "cpcv_slices": [
                {
                    "slice": s["slice_id"],
                    "sharpe": s["metrics"]["sharpe"],
                    "max_dd": s["metrics"]["max_drawdown"],
                }
                for s in cpcv_slices
            ],
        }
        return json.dumps(report_data, indent=2)

    def generate_pdf(self, output_file: Path):
        doc = SimpleDocTemplate(str(output_file), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph(f"Report for Run ID: {self.run_id}", styles["h1"]))
        story.append(Paragraph(f"Strategy: {self.strategy_name}", styles["h2"]))
        story.append(
            Paragraph(
                f"Hyperparameters: {json.dumps(self.hyperparams)}", styles["BodyText"]
            )
        )

        # Add metrics
        metrics = self.results.get("metrics", {})
        for key, value in metrics.items():
            story.append(
                Paragraph(
                    f"<b>{key.replace('_', ' ').title()}:</b> {value:.4f}",
                    styles["BodyText"],
                )
            )

        # Add charts
        chart_paths = []
        if "nav_series" in self.results:
            chart_path = TMP_DIR / f"{self.run_id}_equity_curve.png"
            plot_equity_curve(self.results["nav_series"], chart_path)
            chart_paths.append(chart_path)

        if "drawdowns" in self.results:
            chart_path = TMP_DIR / f"{self.run_id}_drawdown.png"
            plot_drawdown(self.results["drawdowns"], chart_path)
            chart_paths.append(chart_path)

        if "returns" in self.results:
            chart_path = TMP_DIR / f"{self.run_id}_return_histogram.png"
            plot_return_histogram(self.results["returns"], chart_path)
            chart_paths.append(chart_path)

        for path in chart_paths:
            story.append(Image(str(path), width=400, height=200))
            story.append(Spacer(1, 12))

        doc.build(story)

    def generate_pdf_from_figures(
        self,
        output_file: Path,
        chart_figures: List[plt.Figure],
        metrics: Dict[str, Any],
    ):
        with PdfPages(output_file) as pdf:
            # Cover page
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(
                0.5,
                0.9,
                f"Report for Run ID: {self.run_id}",
                ha="center",
                size=20,
            )
            fig.text(0.5, 0.8, f"Strategy: {self.strategy_name}", ha="center", size=16)
            fig.text(
                0.5,
                0.7,
                f"Hyperparameters: {json.dumps(self.hyperparams)}",
                ha="center",
                size=12,
            )

            # Metrics table
            metric_text = "\n".join(
                [
                    f"{key.replace('_', ' ').title()}: {value:.4f}"
                    for key, value in metrics.items()
                ]
            )
            fig.text(0.5, 0.5, metric_text, ha="center", size=12, va="top")
            pdf.savefig(fig)
            plt.close(fig)

            # Chart pages
            for fig in chart_figures:
                pdf.savefig(fig)
                plt.close(fig)

    def generate_arrow(
        self, chart_figures: List[plt.Figure], metrics: Dict[str, Any]
    ) -> bytes:
        import pyarrow as pa
        import io

        chart_data = []
        for fig in chart_figures:
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            chart_data.append(buf.read())

        schema = pa.schema([
            pa.field("run_id", pa.string()),
            pa.field("strategy_name", pa.string()),
            pa.field("hyperparams", pa.string()),
            pa.field("metrics", pa.string()),
            pa.field("charts", pa.list_(pa.binary()))
        ])

        batch = pa.RecordBatch.from_arrays([
            pa.array([self.run_id]),
            pa.array([self.strategy_name]),
            pa.array([json.dumps(self.hyperparams)]),
            pa.array([json.dumps(metrics)]),
            pa.array([chart_data], type=pa.list_(pa.binary()))
        ], schema=schema)

        sink = io.BytesIO()
        with pa.ipc.new_stream(sink, schema) as writer:
            writer.write_batch(batch)

        return sink.getvalue()
