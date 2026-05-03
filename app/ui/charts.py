"""Analysis Charts tab - visualizations of LOO validation results"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from collections import defaultdict


class ChartsView(ttk.Frame):
    """Tab for displaying analysis charts"""

    def __init__(self, parent, loo_view_ref):
        super().__init__(parent)
        self.loo_view = loo_view_ref
        self.canvas_widgets = []
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Create main container with scrollbar
        main_container = ttk.Frame(self)
        main_container.grid(row=0, column=0, sticky="nsew")
        main_container.columnconfigure(0, weight=1)
        main_container.rowconfigure(0, weight=1)

        # Create canvas with scrollbar
        canvas = tk.Canvas(main_container, bg="white")
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        canvas.configure(yscrollcommand=scrollbar.set)

        # Create scrollable frame
        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.columnconfigure(0, weight=1)
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        def _on_frame_configure(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        self.scrollable_frame.bind("<Configure>", _on_frame_configure)

        # Initial message
        self.message_var = tk.StringVar(value="Run Leave-One-Out Validation first to see charts")
        self.message_label = ttk.Label(
            self.scrollable_frame,
            textvariable=self.message_var,
            font=("Arial", 10),
            foreground="gray"
        )
        self.message_label.pack(padx=20, pady=20)

    def refresh_charts(self):
        """Refresh charts based on LOO results"""
        if not self.loo_view or not self.loo_view.loo_results:
            self.message_var.set("No validation results available. Run LOO validation first.")
            return

        # Clear existing charts
        for widget in self.canvas_widgets:
            widget.destroy()
        self.canvas_widgets.clear()

        # Hide message
        self.message_label.pack_forget()

        results = self.loo_view.loo_results

        # Create charts
        self._create_accuracy_chart(results)
        self._create_f1_chart(results)
        self._create_metric_comparison(results)

    def _create_accuracy_chart(self, results):
        """Create accuracy vs k chart"""
        fig = Figure(figsize=(10, 4), dpi=80)
        ax = fig.add_subplot(111)

        # Group by metric
        metrics_data = defaultdict(lambda: {"k_values": [], "accuracy": []})

        for (k, metric), metrics_dict in sorted(results.items()):
            metrics_data[metric]["k_values"].append(k)
            metrics_data[metric]["accuracy"].append(metrics_dict['accuracy'])

        # Plot lines
        colors = {"euclidean": "blue", "bray_curtis": "green", "chebyshev": "red"}
        for metric in sorted(metrics_data.keys()):
            data = metrics_data[metric]
            k_vals = sorted(set(data["k_values"]))
            acc_vals = []
            for k in k_vals:
                idx = data["k_values"].index(k)
                acc_vals.append(data["accuracy"][idx])
            
            ax.plot(k_vals, acc_vals, marker='o', label=metric, color=colors.get(metric, "gray"), linewidth=2)

        ax.set_xlabel("k (Number of Neighbors)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Leave-One-Out Validation: Accuracy vs k")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        canvas_widget = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        self.canvas_widgets.append(canvas_widget.get_tk_widget())

    def _create_f1_chart(self, results):
        """Create F1-Score vs k chart"""
        fig = Figure(figsize=(10, 4), dpi=80)
        ax = fig.add_subplot(111)

        # Group by metric
        metrics_data = defaultdict(lambda: {"k_values": [], "f1": []})

        for (k, metric), metrics_dict in sorted(results.items()):
            metrics_data[metric]["k_values"].append(k)
            metrics_data[metric]["f1"].append(metrics_dict['f1'])

        # Plot lines
        colors = {"euclidean": "blue", "bray_curtis": "green", "chebyshev": "red"}
        for metric in sorted(metrics_data.keys()):
            data = metrics_data[metric]
            k_vals = sorted(set(data["k_values"]))
            f1_vals = []
            for k in k_vals:
                idx = data["k_values"].index(k)
                f1_vals.append(data["f1"][idx])
            
            ax.plot(k_vals, f1_vals, marker='s', label=metric, color=colors.get(metric, "gray"), linewidth=2)

        ax.set_xlabel("k (Number of Neighbors)")
        ax.set_ylabel("F1-Score")
        ax.set_title("Leave-One-Out Validation: F1-Score vs k")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        canvas_widget = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        self.canvas_widgets.append(canvas_widget.get_tk_widget())

    def _create_metric_comparison(self, results):
        """Create comparison chart (accuracy bar chart for each metric)"""
        fig = Figure(figsize=(10, 4), dpi=80)
        ax = fig.add_subplot(111)

        # Group by metric
        metrics_data = defaultdict(lambda: {"accuracy": [], "f1": [], "precision": []})

        for (k, metric), metrics_dict in sorted(results.items()):
            metrics_data[metric]["accuracy"].append(metrics_dict['accuracy'])
            metrics_data[metric]["f1"].append(metrics_dict['f1'])
            metrics_data[metric]["precision"].append(metrics_dict['precision'])

        # Calculate averages
        metric_names = sorted(metrics_data.keys())
        avg_accuracy = [sum(metrics_data[m]["accuracy"]) / len(metrics_data[m]["accuracy"]) for m in metric_names]
        avg_f1 = [sum(metrics_data[m]["f1"]) / len(metrics_data[m]["f1"]) for m in metric_names]
        avg_precision = [sum(metrics_data[m]["precision"]) / len(metrics_data[m]["precision"]) for m in metric_names]

        # Create bar chart
        x = range(len(metric_names))
        width = 0.25

        ax.bar([i - width for i in x], avg_accuracy, width, label='Accuracy', color='skyblue')
        ax.bar([i for i in x], avg_f1, width, label='F1-Score', color='lightgreen')
        ax.bar([i + width for i in x], avg_precision, width, label='Precision', color='salmon')

        ax.set_ylabel('Score')
        ax.set_title('Average Performance by Metric (across all k values)')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])

        canvas_widget = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        self.canvas_widgets.append(canvas_widget.get_tk_widget())
