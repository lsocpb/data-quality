"""Main Tkinter window with tab navigation"""

import tkinter as tk
from tkinter import ttk
from pathlib import Path
import sys


class MainWindow(tk.Tk):
    """Main application window with three tabs"""

    def __init__(self, engine):
        super().__init__()

        self.title("Keystroke Dynamics Analysis")
        self.geometry("1100x750")
        
        # Initialize services
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from app.services.database_service import DatabaseService
        from app.services.pipeline_service import PipelineService

        self.db_service = DatabaseService(engine)
        self.pipeline_service = PipelineService(self.db_service)
        
        # Store current analysis for syncing between tabs
        self.current_analysis = None

        # Create notebook (tab container)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Import view modules
        from app.ui.live_analysis import LiveAnalysisView
        from app.ui.results import ResultsView
        from app.ui.registration_demo import RegistrationDemoView
        from app.ui.loo_validation import LOOValidationView
        from app.ui.charts import ChartsView

        # Create tab instances
        self.live_analysis_view = LiveAnalysisView(
            self.notebook,
            self.pipeline_service,
            self.db_service,
            self
        )
        self.notebook.add(self.live_analysis_view, text="Live Analysis")

        self.results_view = ResultsView(self.notebook)
        self.notebook.add(self.results_view, text="Results")

        self.registration_demo_view = RegistrationDemoView(
            self.notebook,
            self.db_service
        )
        self.notebook.add(self.registration_demo_view, text="Registration Demo")

        self.charts_view = ChartsView(self.notebook, None)  # Placeholder - set later
        
        self.loo_validation_view = LOOValidationView(
            self.notebook,
            self.pipeline_service,
            main_window_ref=self  # Pass main window reference so LOO can signal charts
        )
        loo_tab_id = self.notebook.add(self.loo_validation_view, text="LOO Validation")
        
        # Now update charts_view with proper reference
        self.charts_view.loo_view = self.loo_validation_view
        self.notebook.add(self.charts_view, text="Analysis Charts")

    def set_current_analysis(self, result):
        """Called by Live Analysis when analysis completes"""
        self.current_analysis = result
        self.results_view.add_result(result)
        # Auto-refresh charts if LOO results available
        if hasattr(self, 'charts_view') and self.loo_validation_view.loo_results:
            self.after(100, self.charts_view.refresh_charts)
