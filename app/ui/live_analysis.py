"""Live Analysis view - interactive analysis tab"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from app.services.models import AnalysisResult


class LiveAnalysisView(ttk.Frame):
    """Tab for running live keystroke analysis"""

    def __init__(self, parent, pipeline_service, db_service, main_window):
        super().__init__(parent)
        self.pipeline = pipeline_service
        self.db = db_service
        self.main_window = main_window
        self.columnconfigure(1, weight=1)

        # Load users
        self.users = sorted([u for u in db_service.load_users() if u])  # Filter empty string
        
        # --- Control Panel ---
        control_frame = ttk.LabelFrame(self, text="Analysis Controls", padding=10)
        control_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        control_frame.columnconfigure(1, weight=1)

        # Mode selection (Identification / Verification)
        ttk.Label(control_frame, text="Mode:").grid(row=0, column=0, sticky="w", padx=5)
        self.mode_var = tk.StringVar(value="identification")
        mode_frame = ttk.Frame(control_frame)
        mode_frame.grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Radiobutton(mode_frame, text="Identification", variable=self.mode_var, value="identification",
                       command=self._on_mode_changed).pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text="Verification", variable=self.mode_var, value="verification",
                       command=self._on_mode_changed).pack(side="left", padx=5)

        # User selection
        ttk.Label(control_frame, text="User:").grid(row=0, column=2, sticky="w", padx=5)
        self.user_var = tk.StringVar()
        self.user_combo = ttk.Combobox(
            control_frame,
            textvariable=self.user_var,
            values=self.users,
            state="readonly",
            width=15
        )
        self.user_combo.grid(row=0, column=3, sticky="ew", padx=5)
        self.user_combo.bind("<<ComboboxSelected>>", self._on_user_selected)

        # Sample selection
        ttk.Label(control_frame, text="Sample:").grid(row=1, column=0, sticky="w", padx=5)
        self.sample_var = tk.StringVar()
        self.sample_combo = ttk.Combobox(
            control_frame,
            textvariable=self.sample_var,
            state="readonly",
            width=10
        )
        self.sample_combo.grid(row=1, column=1, sticky="ew", padx=5)
        self.sample_combo.bind("<<ComboboxSelected>>", self._on_sample_selected)

        # Claimed User (only for Verification mode)
        self.claimed_user_label = ttk.Label(control_frame, text="Claimed User:")
        self.claimed_user_label.grid(row=1, column=2, sticky="w", padx=5)
        self.claimed_user_var = tk.StringVar()
        self.claimed_user_combo = ttk.Combobox(
            control_frame,
            textvariable=self.claimed_user_var,
            values=self.users,
            state="readonly",
            width=15
        )
        self.claimed_user_combo.grid(row=1, column=3, sticky="ew", padx=5)
        
        # Threshold (only for Verification mode)
        self.threshold_label = ttk.Label(control_frame, text="Threshold:")
        self.threshold_label.grid(row=2, column=0, sticky="w", padx=5)
        self.threshold_var = tk.DoubleVar(value=150.0)
        self.threshold_spinbox = ttk.Spinbox(
            control_frame,
            from_=0.0,
            to=500.0,
            textvariable=self.threshold_var,
            width=10
        )
        self.threshold_spinbox.grid(row=2, column=1, sticky="ew", padx=5)

        # Metric selection
        ttk.Label(control_frame, text="Metric:").grid(row=2, column=2, sticky="w", padx=5)
        self.metric_var = tk.StringVar(value="bray_curtis")
        self.metric_combo = ttk.Combobox(
            control_frame,
            textvariable=self.metric_var,
            values=["euclidean", "bray_curtis", "chebyshev"],
            state="readonly",
            width=15
        )
        self.metric_combo.grid(row=2, column=3, sticky="ew", padx=5)

        # k selection
        ttk.Label(control_frame, text="k (neighbors):").grid(row=3, column=0, sticky="w", padx=5)
        self.k_var = tk.IntVar(value=3)
        self.k_spinbox = ttk.Spinbox(
            control_frame,
            from_=1,
            to=20,
            textvariable=self.k_var,
            width=10
        )
        self.k_spinbox.grid(row=3, column=1, sticky="ew", padx=5)

        # Analyze button
        self.analyze_btn = ttk.Button(
            control_frame,
            text="Analyze",
            command=self._run_analysis
        )
        self.analyze_btn.grid(row=3, column=2, columnspan=2, sticky="ew", padx=5, pady=10)

        # Hide verification controls initially
        self._update_mode_visibility()

        # --- Results Panel ---
        results_frame = ttk.LabelFrame(self, text="Analysis Results", padding=10)
        results_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Results text widget with scrollbar
        scrollbar = ttk.Scrollbar(results_frame)
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        self.results_text = tk.Text(
            results_frame,
            height=20,
            width=60,
            yscrollcommand=scrollbar.set,
            font=("Courier", 9),
            state="disabled"
        )
        self.results_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.config(command=self.results_text.yview)

        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(self, textvariable=self.status_var)
        status_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=5)

    def _on_user_selected(self, event=None):
        """Update sample list when user is selected"""
        user = self.user_var.get()
        if user:
            samples = self.db.load_samples_for_user(user)
            self.sample_combo.config(values=samples, state="readonly")
            if samples:
                self.sample_var.set(samples[0])
        else:
            self.sample_combo.config(values=[], state="readonly")
            self.sample_var.set("")

    def _on_sample_selected(self, event=None):
        """Sample selected - enable analyze button"""
        pass

    def _on_mode_changed(self):
        """Handle mode change (Identification/Verification)"""
        self._update_mode_visibility()

    def _update_mode_visibility(self):
        """Show/hide verification-specific controls"""
        mode = self.mode_var.get()
        if mode == "verification":
            # Show verification controls
            self.claimed_user_label.grid()
            self.claimed_user_combo.grid()
            self.threshold_label.grid()
            self.threshold_spinbox.grid()
        else:
            # Hide verification controls
            self.claimed_user_label.grid_remove()
            self.claimed_user_combo.grid_remove()
            self.threshold_label.grid_remove()
            self.threshold_spinbox.grid_remove()

    def _run_analysis(self):
        """Run keystroke analysis in background thread"""
        user = self.user_var.get()
        sample_str = self.sample_var.get()
        metric = self.metric_var.get()
        k = self.k_var.get()
        mode = self.mode_var.get()

        if not user or not sample_str:
            messagebox.showwarning("Input Error", "Please select both user and sample")
            return

        try:
            sample = int(sample_str)
        except ValueError:
            messagebox.showerror("Input Error", "Invalid sample number")
            return

        # Verification mode specific validation
        if mode == "verification":
            claimed_user = self.claimed_user_var.get()
            if not claimed_user:
                messagebox.showwarning("Input Error", "Please select claimed user for verification")
                return
        else:
            claimed_user = None

        # Disable button and show status
        self.analyze_btn.config(state="disabled")
        self.status_var.set("Running analysis...")
        self._clear_results()

        # Run analysis in background thread
        thread = threading.Thread(
            target=self._analysis_worker,
            args=(user, sample, metric, k, mode, claimed_user),
            daemon=True
        )
        thread.start()

    def _analysis_worker(self, user, sample, metric, k, mode, claimed_user=None):
        """Background worker for analysis"""
        try:
            if mode == "verification":
                # Run verification analysis
                threshold = self.threshold_var.get()
                result = self.pipeline.analyze_verification(
                    user_id=user,
                    sample_number=sample,
                    claimed_user_id=claimed_user,
                    threshold=threshold,
                    metric=metric,
                    k=k
                )
            else:
                # Run identification analysis
                result = self.pipeline.analyze_identification(user, sample, metric, k=k)
            
            # Update UI in main thread
            self.after(0, self._display_results, result, mode)
            
            # Notify main window
            self.after(0, self.main_window.set_current_analysis, result)
            
        except Exception as e:
            self.after(0, self._show_error, str(e))
        finally:
            self.after(0, self._enable_controls)

    def _display_results(self, result, mode="identification"):
        """Display analysis results in text widget"""
        self._clear_results()
        
        self.results_text.config(state="normal")
        
        if mode == "verification":
            # Display verification results
            text = f"""VERIFICATION RESULTS
{'='*50}

Query Sample: {result.user_id} (Sample {result.sample_number})
Claimed User: {result.claimed_user_id}
Metric: {result.metric.upper()}
k (neighbors): {result.k}
Threshold: {result.threshold:.1f}

Raw Events: {result.raw_event_count}
Features Extracted: {len(result.features)} keys

VERIFICATION: {'✅ PASSED' if result.verification_match else '❌ REJECTED'}
Confidence Score: {result.confidence_score:.6f}
Explanation: {result.explanation}

TOP NEIGHBORS ({len(result.neighbors)} returned):
{'-'*50}
"""
        else:
            # Display identification results
            text = f"""ANALYSIS RESULTS
{'='*50}

Query Sample: {result.user_id} (Sample {result.sample_number})
Metric: {result.metric.upper()}
k (neighbors): {result.k}

Raw Events: {result.raw_event_count}
Features Extracted: {len(result.features)} keys
Prediction Timing: {result.timing_ms.get('extract_features', 0):.1f}ms extraction + {result.timing_ms.get('knn_analysis', 0):.1f}ms KNN = {sum(result.timing_ms.values()):.1f}ms total

PREDICTED USER: {result.predicted_user_id}
Confidence Score: {result.confidence_score:.6f}

TOP NEIGHBORS ({len(result.neighbors)} returned):
{'-'*50}
"""
        
        self.results_text.insert("end", text)
        
        for idx, (user, distance, sample) in enumerate(result.neighbors, 1):
            neighbor_text = f"{idx:2d}. {user:15s} (sample {sample:2d}): distance = {distance:.6f}\n"
            self.results_text.insert("end", neighbor_text)
        
        self.results_text.config(state="disabled")
        self.status_var.set(f"Analysis complete - {len(result.neighbors)} neighbors found")

    def _clear_results(self):
        """Clear results text widget"""
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.config(state="disabled")

    def _enable_controls(self):
        """Re-enable controls after analysis"""
        self.analyze_btn.config(state="normal")
        self.status_var.set("Ready")

    def _show_error(self, error_msg: str):
        """Show error message"""
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", f"ERROR:\n{error_msg}")
        self.results_text.config(state="disabled")
        self.status_var.set(f"Error: {error_msg}")
        messagebox.showerror("Analysis Error", error_msg)

