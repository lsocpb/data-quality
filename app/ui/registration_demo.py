"""Registration Demo view - showing registration samples"""

import tkinter as tk
from tkinter import ttk


class RegistrationDemoView(ttk.Frame):
    """Tab for viewing user registration samples"""

    def __init__(self, parent, db_service):
        super().__init__(parent)
        self.db = db_service
        self.columnconfigure(1, weight=1)

        # Load users
        self.users = sorted([u for u in db_service.load_users() if u])

        # --- User Selection ---
        control_frame = ttk.LabelFrame(self, text="Select User", padding=10)
        control_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        control_frame.columnconfigure(1, weight=1)

        ttk.Label(control_frame, text="User:").grid(row=0, column=0, sticky="w", padx=5)
        self.user_var = tk.StringVar()
        self.user_combo = ttk.Combobox(
            control_frame,
            textvariable=self.user_var,
            values=self.users,
            state="readonly",
            width=20
        )
        self.user_combo.grid(row=0, column=1, sticky="ew", padx=5)
        self.user_combo.bind("<<ComboboxSelected>>", self._on_user_selected)

        # --- Samples Info ---
        info_frame = ttk.LabelFrame(self, text="Registration Samples", padding=10)
        info_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        scrollbar = ttk.Scrollbar(info_frame)
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.info_text = tk.Text(
            info_frame,
            height=20,
            width=70,
            yscrollcommand=scrollbar.set,
            font=("Courier", 9),
            state="disabled"
        )
        self.info_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.config(command=self.info_text.yview)

    def _on_user_selected(self, event=None):
        """Load and display user samples"""
        user = self.user_var.get()
        if not user:
            return

        # Get samples for user
        samples = self.db.load_samples_for_user(user)
        
        self.info_text.config(state="normal")
        self.info_text.delete("1.0", "end")

        info = f"""User: {user}
Registration Samples: {len(samples)}
{'='*60}

"""
        self.info_text.insert("end", info)

        # Show info for each sample
        for sample_num in samples:
            events = self.db.get_sample_events(user, sample_num)
            
            # Events DataFrame contains KeyPressed column with press/release indicators
            # Count total events (each row = one keystroke)
            total_events = len(events)
            
            # Count press and release (if KeyPressed column has these values)
            if not events.empty and 'KeyPressed' in events.columns:
                key_presses = len(events[events['KeyPressed'].notna()])
                # For releases, check if column has release data
                key_releases = len(events)  # All events include release timing
            else:
                key_presses = 0
                key_releases = 0
            
            sample_info = f"""Sample {sample_num}:
  Total Events: {total_events}
  Key Presses: {key_presses}
  Key Releases: {key_releases}

"""
            self.info_text.insert("end", sample_info)

        self.info_text.config(state="disabled")

