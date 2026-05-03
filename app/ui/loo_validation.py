"""Leave-One-Out Validation tab"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading


class LOOValidationView(ttk.Frame):
    """Tab for Leave-One-Out cross-validation testing"""

    def __init__(self, parent, pipeline_service, main_window_ref=None):
        super().__init__(parent)
        self.pipeline = pipeline_service
        self.main_window = main_window_ref
        self.loo_results = None
        self.columnconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)

        # --- Control Panel ---
        control_frame = ttk.LabelFrame(self, text="Leave-One-Out Configuration", padding=10)
        control_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        control_frame.columnconfigure(1, weight=1)

        # k values selection
        ttk.Label(control_frame, text="k Values (comma-separated):").grid(row=0, column=0, sticky="w", padx=5)
        self.k_var = tk.StringVar(value="1,2,3")  # Changed from 3,5,7,10 - optimal for small dataset
        self.k_entry = ttk.Entry(control_frame, textvariable=self.k_var, width=30)
        self.k_entry.grid(row=0, column=1, sticky="ew", padx=5)

        # Metrics selection
        ttk.Label(control_frame, text="Metrics:").grid(row=1, column=0, sticky="w", padx=5)
        
        metrics_frame = ttk.Frame(control_frame)
        metrics_frame.grid(row=1, column=1, sticky="w", padx=5)
        
        self.euclidean_var = tk.BooleanVar(value=True)
        self.bray_curtis_var = tk.BooleanVar(value=True)
        self.chebyshev_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(metrics_frame, text="Euclidean", variable=self.euclidean_var).pack(side="left", padx=5)
        ttk.Checkbutton(metrics_frame, text="Bray-Curtis", variable=self.bray_curtis_var).pack(side="left", padx=5)
        ttk.Checkbutton(metrics_frame, text="Chebyshev", variable=self.chebyshev_var).pack(side="left", padx=5)

        # Run button
        self.run_btn = ttk.Button(
            control_frame,
            text="Run Leave-One-Out Validation",
            command=self._run_validation
        )
        self.run_btn.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=10)

        # --- Progress Panel ---
        progress_frame = ttk.LabelFrame(self, text="Progress", padding=10)
        progress_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        progress_frame.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        self.status_label.pack(fill="x", padx=5, pady=5)

        # --- Results Panel ---
        results_frame = ttk.LabelFrame(self, text="Results", padding=10)
        results_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)

        # Results treeview
        scrollbar = ttk.Scrollbar(results_frame)
        scrollbar.grid(row=1, column=1, sticky="ns")

        self.tree = ttk.Treeview(
            results_frame,
            columns=("k", "metric", "accuracy", "precision", "recall", "f1", "tp", "fp", "fn"),
            height=12,
            yscrollcommand=scrollbar.set
        )
        self.tree.grid(row=1, column=0, sticky="nsew")
        scrollbar.config(command=self.tree.yview)

        # Configure columns
        self.tree.column("#0", width=0)
        self.tree.column("k", anchor="center", width=40)
        self.tree.column("metric", anchor="w", width=100)
        self.tree.column("accuracy", anchor="center", width=80)
        self.tree.column("precision", anchor="center", width=80)
        self.tree.column("recall", anchor="center", width=80)
        self.tree.column("f1", anchor="center", width=80)
        self.tree.column("tp", anchor="center", width=60)
        self.tree.column("fp", anchor="center", width=60)
        self.tree.column("fn", anchor="center", width=60)

        # Configure headings
        self.tree.heading("#0", text="")
        self.tree.heading("k", text="k")
        self.tree.heading("metric", text="Metric")
        self.tree.heading("accuracy", text="Accuracy")
        self.tree.heading("precision", text="Precision")
        self.tree.heading("recall", text="Recall")
        self.tree.heading("f1", text="F1-Score")
        self.tree.heading("tp", text="TP")
        self.tree.heading("fp", text="FP")
        self.tree.heading("fn", text="FN")

        # Summary label
        ttk.Label(results_frame, text="Summary statistics:").grid(row=0, column=0, sticky="w", pady=5)

    def _run_validation(self):
        """Run LOO validation in background"""
        # Parse k values
        try:
            k_str = self.k_var.get().strip()
            k_values = [int(k.strip()) for k in k_str.split(",")]
        except ValueError:
            messagebox.showerror("Input Error", "Invalid k values format. Use comma-separated integers.")
            return

        # Get selected metrics
        metrics = []
        if self.euclidean_var.get():
            metrics.append("euclidean")
        if self.bray_curtis_var.get():
            metrics.append("bray_curtis")
        if self.chebyshev_var.get():
            metrics.append("chebyshev")

        if not metrics:
            messagebox.showwarning("Selection Error", "Please select at least one metric")
            return

        # Disable button
        self.run_btn.config(state="disabled")
        self.status_var.set("Running LOO validation...")

        # Run in background
        thread = threading.Thread(
            target=self._validation_worker,
            args=(k_values, metrics),
            daemon=True
        )
        thread.start()

    def _validation_worker(self, k_values, metrics):
        """Background worker for LOO validation"""
        try:
            self.status_var.set("Validation in progress...")
            
            results = self.pipeline.leave_one_out_validation(
                k_values=k_values,
                metrics=metrics
            )

            # Update UI in main thread
            self.after(0, self._display_results, results)

        except Exception as e:
            self.after(0, self._show_error, str(e))
        finally:
            self.after(0, self._enable_controls)

    def _display_results(self, results):
        """Display LOO results in treeview"""
        self.loo_results = results

        # Clear treeview
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Insert results
        for (k, metric), metrics_dict in sorted(results.items()):
            self.tree.insert(
                "",
                "end",
                values=(
                    k,
                    metric,
                    f"{metrics_dict['accuracy']:.4f}",
                    f"{metrics_dict['precision']:.4f}",
                    f"{metrics_dict['recall']:.4f}",
                    f"{metrics_dict['f1']:.4f}",
                    metrics_dict['tp'],
                    metrics_dict['fp'],
                    metrics_dict['fn']
                )
            )

        # Calculate and display summary
        avg_accuracy = sum(r['accuracy'] for r in results.values()) / len(results)
        avg_f1 = sum(r['f1'] for r in results.values()) / len(results)
        self.status_var.set(f"✓ COMPLETE! Avg Accuracy: {avg_accuracy:.4f}, Avg F1: {avg_f1:.4f}")
        
        # Signal charts to refresh
        if self.main_window and hasattr(self.main_window, 'charts_view'):
            self.after(100, self.main_window.charts_view.refresh_charts)

    def _enable_controls(self):
        """Re-enable controls"""
        self.run_btn.config(state="normal")

    def _show_error(self, error_msg: str):
        """Show error message"""
        self.status_var.set(f"Error: {error_msg}")
        messagebox.showerror("Validation Error", error_msg)
