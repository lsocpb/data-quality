"""Results view - analysis history and details"""

import tkinter as tk
from tkinter import ttk
from datetime import datetime


class ResultsView(ttk.Frame):
    """Tab for viewing analysis results history"""

    def __init__(self, parent):
        super().__init__(parent)
        self.results_history = []
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # --- Summary Panel ---
        summary_frame = ttk.LabelFrame(self, text="Analysis History", padding=10)
        summary_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        summary_frame.columnconfigure(0, weight=1)

        summary_label = ttk.Label(
            summary_frame,
            text="Recent analysis results appear below. Click to view details."
        )
        summary_label.pack(fill="both", expand=True)

        # --- Results Treeview ---
        tree_frame = ttk.LabelFrame(self, text="Results", padding=10)
        tree_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        # Create treeview
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.tree = ttk.Treeview(
            tree_frame,
            columns=("query_user", "metric", "k", "predicted", "confidence", "timing"),
            height=12,
            yscrollcommand=scrollbar.set
        )
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.config(command=self.tree.yview)

        # Configure columns
        self.tree.column("#0", width=0)
        self.tree.column("query_user", anchor="w", width=100)
        self.tree.column("metric", anchor="center", width=80)
        self.tree.column("k", anchor="center", width=40)
        self.tree.column("predicted", anchor="w", width=100)
        self.tree.column("confidence", anchor="center", width=100)
        self.tree.column("timing", anchor="center", width=80)

        # Configure headings
        self.tree.heading("#0", text="")
        self.tree.heading("query_user", text="Query User")
        self.tree.heading("metric", text="Metric")
        self.tree.heading("k", text="k")
        self.tree.heading("predicted", text="Predicted")
        self.tree.heading("confidence", text="Confidence")
        self.tree.heading("timing", text="Timing (ms)")

        # Bind selection
        self.tree.bind("<Double-1>", self._on_result_selected)

        # --- Detail Panel ---
        detail_frame = ttk.LabelFrame(self, text="Details", padding=10)
        detail_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        detail_frame.columnconfigure(0, weight=1)

        scrollbar_detail = ttk.Scrollbar(detail_frame)
        scrollbar_detail.grid(row=0, column=1, sticky="ns")

        self.detail_text = tk.Text(
            detail_frame,
            height=8,
            width=80,
            yscrollcommand=scrollbar_detail.set,
            font=("Courier", 9),
            state="disabled"
        )
        self.detail_text.grid(row=0, column=0, sticky="nsew")
        scrollbar_detail.config(command=self.detail_text.yview)

    def add_result(self, result):
        """Add new analysis result to history"""
        self.results_history.insert(0, result)
        self.results_history = self.results_history[:50]  # Keep last 50

        # Update treeview
        item_text = f"{len(self.results_history)}"
        self.tree.insert(
            "",
            "end",
            values=(
                result.user_id,
                result.metric,
                result.k,
                result.predicted_user_id,
                f"{result.confidence_score:.6f}",
                f"{sum(result.timing_ms.values()):.1f}"
            )
        )

    def _on_result_selected(self, event):
        """Display details when result is double-clicked"""
        selection = self.tree.selection()
        if not selection:
            return

        # Get index from treeview
        item_index = list(self.tree.get_children()).index(selection[0])
        if item_index < len(self.results_history):
            result = self.results_history[item_index]
            self._display_detail(result)

    def _display_detail(self, result):
        """Display detailed information about result"""
        self.detail_text.config(state="normal")
        self.detail_text.delete("1.0", "end")

        detail_text = f"""Query: {result.user_id} sample {result.sample_number}
Metric: {result.metric}, k={result.k}
Predicted: {result.predicted_user_id} (confidence: {result.confidence_score:.6f})
Raw Events: {result.raw_event_count}
Features: {len(result.features)} keys
Total Time: {sum(result.timing_ms.values()):.1f}ms

Top 3 Neighbors:
"""
        self.detail_text.insert("end", detail_text)

        for idx, (user, distance, sample) in enumerate(result.neighbors[:3], 1):
            self.detail_text.insert(
                "end",
                f"  {idx}. {user} (sample {sample}): {distance:.6f}\n"
            )

        self.detail_text.config(state="disabled")

