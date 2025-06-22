"""
User Preferences Dialog for Pattern Detection Settings
Implements custom user controls as specified in UX design
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
import os

class PreferencesDialog:
    def __init__(self, parent, alert_system):
        self.parent = parent
        self.alert_system = alert_system
        self.dialog = None
        
    def show(self):
        """Show preferences dialog"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Pattern Detection Preferences")
        self.dialog.geometry("500x600")
        self.dialog.resizable(False, False)
        
        # Center the dialog
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup preferences UI"""
        # Main container with padding
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for tabbed interface
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Alert Settings Tab
        self.create_alert_settings_tab(notebook)
        
        # Pattern Settings Tab
        self.create_pattern_settings_tab(notebook)
        
        # Trading Settings Tab
        self.create_trading_settings_tab(notebook)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Apply", command=self.apply_settings).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_defaults).pack(side=tk.LEFT)
        
    def create_alert_settings_tab(self, notebook):
        """Create alert settings tab"""
        frame = ttk.Frame(notebook, padding="15")
        notebook.add(frame, text="Alert Settings")
        
        # Confidence threshold
        conf_frame = ttk.LabelFrame(frame, text="Confidence Threshold", padding="10")
        conf_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(conf_frame, text="Minimum confidence for alerts:").pack(anchor=tk.W)
        
        self.confidence_var = tk.DoubleVar(value=self.alert_system.notification_settings['min_confidence'])
        self.confidence_scale = ttk.Scale(conf_frame, from_=50.0, to=95.0, 
                                         variable=self.confidence_var, orient=tk.HORIZONTAL)
        self.confidence_scale.pack(fill=tk.X, pady=5)
        
        self.confidence_label = ttk.Label(conf_frame, text="")
        self.confidence_label.pack(anchor=tk.W)
        
        def update_confidence_label(*args):
            self.confidence_label.config(text=f"Current: {self.confidence_var.get():.1f}%")
        
        self.confidence_var.trace('w', update_confidence_label)
        update_confidence_label()
        
        # Notification methods
        notif_frame = ttk.LabelFrame(frame, text="Notification Methods", padding="10")
        notif_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.popup_var = tk.BooleanVar(value=self.alert_system.notification_settings['popup_enabled'])
        ttk.Checkbutton(notif_frame, text="Show popup alerts", variable=self.popup_var).pack(anchor=tk.W)
        
        self.sound_var = tk.BooleanVar(value=self.alert_system.notification_settings['sound_enabled'])
        ttk.Checkbutton(notif_frame, text="Play notification sound", variable=self.sound_var).pack(anchor=tk.W)
        
        self.email_var = tk.BooleanVar(value=self.alert_system.notification_settings['email_enabled'])
        ttk.Checkbutton(notif_frame, text="Send email notifications", variable=self.email_var).pack(anchor=tk.W)
        
        # Alert frequency
        freq_frame = ttk.LabelFrame(frame, text="Alert Frequency", padding="10")
        freq_frame.pack(fill=tk.X)
        
        ttk.Label(freq_frame, text="Maximum alerts per hour:").pack(anchor=tk.W)
        self.alert_limit_var = tk.IntVar(value=10)
        alert_limit_spin = ttk.Spinbox(freq_frame, from_=1, to=50, textvariable=self.alert_limit_var, width=10)
        alert_limit_spin.pack(anchor=tk.W, pady=5)
        
    def create_pattern_settings_tab(self, notebook):
        """Create pattern monitoring settings tab"""
        frame = ttk.Frame(notebook, padding="15")
        notebook.add(frame, text="Pattern Settings")
        
        # Pattern types to monitor
        patterns_frame = ttk.LabelFrame(frame, text="Monitor These Patterns", padding="10")
        patterns_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.pattern_vars = {}
        enabled_patterns = self.alert_system.notification_settings['enabled_patterns']
        
        pattern_options = [
            ('head_shoulders', 'Head and Shoulders'),
            ('double_top', 'Double Top'),
            ('double_bottom', 'Double Bottom'),
            ('triangle', 'Triangle Patterns'),
            ('wedge', 'Wedge Patterns'),
            ('flag', 'Flag Patterns'),
            ('pennant', 'Pennant Patterns'),
            ('cup_handle', 'Cup and Handle'),
            ('support_resistance', 'Support/Resistance Levels')
        ]
        
        for pattern_key, pattern_name in pattern_options:
            var = tk.BooleanVar(value=pattern_key in enabled_patterns)
            self.pattern_vars[pattern_key] = var
            ttk.Checkbutton(patterns_frame, text=pattern_name, variable=var).pack(anchor=tk.W)
            
        # Timeframe settings
        timeframe_frame = ttk.LabelFrame(frame, text="Analysis Timeframes", padding="10")
        timeframe_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(timeframe_frame, text="Primary timeframe for analysis:").pack(anchor=tk.W)
        self.primary_timeframe_var = tk.StringVar(value="5min")
        timeframe_combo = ttk.Combobox(timeframe_frame, textvariable=self.primary_timeframe_var,
                                      values=["1min", "5min", "15min", "30min", "1hour", "4hour", "daily"],
                                      state="readonly", width=15)
        timeframe_combo.pack(anchor=tk.W, pady=5)
        
        # Multi-timeframe analysis
        self.multi_timeframe_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(timeframe_frame, text="Enable multi-timeframe confirmation", 
                       variable=self.multi_timeframe_var).pack(anchor=tk.W, pady=5)
        
    def create_trading_settings_tab(self, notebook):
        """Create trading preferences tab"""
        frame = ttk.Frame(notebook, padding="15")
        notebook.add(frame, text="Trading Settings")
        
        # Preferred symbols
        symbols_frame = ttk.LabelFrame(frame, text="Preferred Trading Symbols", padding="10")
        symbols_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(symbols_frame, text="Monitor these symbols (comma-separated):").pack(anchor=tk.W)
        self.symbols_var = tk.StringVar(value="AAPL, MSFT, TSLA, GOOGL, AMZN")
        symbols_entry = ttk.Entry(symbols_frame, textvariable=self.symbols_var, width=50)
        symbols_entry.pack(fill=tk.X, pady=5)
        
        # Risk management
        risk_frame = ttk.LabelFrame(frame, text="Risk Management", padding="10")
        risk_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(risk_frame, text="Default stop-loss percentage:").pack(anchor=tk.W)
        self.stop_loss_var = tk.DoubleVar(value=2.0)
        stop_loss_spin = ttk.Spinbox(risk_frame, from_=0.5, to=10.0, increment=0.1,
                                    textvariable=self.stop_loss_var, width=10)
        stop_loss_spin.pack(anchor=tk.W, pady=2)
        
        ttk.Label(risk_frame, text="Default take-profit percentage:").pack(anchor=tk.W, pady=(10, 0))
        self.take_profit_var = tk.DoubleVar(value=4.0)
        take_profit_spin = ttk.Spinbox(risk_frame, from_=1.0, to=20.0, increment=0.1,
                                      textvariable=self.take_profit_var, width=10)
        take_profit_spin.pack(anchor=tk.W, pady=2)
        
        # Trading session
        session_frame = ttk.LabelFrame(frame, text="Trading Session", padding="10")
        session_frame.pack(fill=tk.X)
        
        ttk.Label(session_frame, text="Active trading hours:").pack(anchor=tk.W)
        
        hours_frame = ttk.Frame(session_frame)
        hours_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(hours_frame, text="From:").pack(side=tk.LEFT)
        self.start_hour_var = tk.StringVar(value="09:30")
        start_time_combo = ttk.Combobox(hours_frame, textvariable=self.start_hour_var,
                                       values=[f"{h:02d}:00" for h in range(24)] + 
                                              [f"{h:02d}:30" for h in range(24)],
                                       width=8, state="readonly")
        start_time_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(hours_frame, text="To:").pack(side=tk.LEFT, padx=(20, 0))
        self.end_hour_var = tk.StringVar(value="16:00")
        end_time_combo = ttk.Combobox(hours_frame, textvariable=self.end_hour_var,
                                     values=[f"{h:02d}:00" for h in range(24)] + 
                                            [f"{h:02d}:30" for h in range(24)],
                                     width=8, state="readonly")
        end_time_combo.pack(side=tk.LEFT, padx=5)
        
    def apply_settings(self):
        """Apply all settings"""
        try:
            # Update alert system settings
            new_settings = {
                'min_confidence': self.confidence_var.get(),
                'popup_enabled': self.popup_var.get(),
                'sound_enabled': self.sound_var.get(),
                'email_enabled': self.email_var.get(),
                'enabled_patterns': [pattern for pattern, var in self.pattern_vars.items() if var.get()]
            }
            
            self.alert_system.update_settings(new_settings)
            
            # Save additional preferences
            additional_prefs = {
                'primary_timeframe': self.primary_timeframe_var.get(),
                'multi_timeframe_enabled': self.multi_timeframe_var.get(),
                'preferred_symbols': [s.strip() for s in self.symbols_var.get().split(',')],
                'stop_loss_percent': self.stop_loss_var.get(),
                'take_profit_percent': self.take_profit_var.get(),
                'trading_start_time': self.start_hour_var.get(),
                'trading_end_time': self.end_hour_var.get(),
                'max_alerts_per_hour': self.alert_limit_var.get()
            }
            
            self.save_additional_preferences(additional_prefs)
            
            messagebox.showinfo("Settings Applied", "Preferences have been saved successfully!")
            self.dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
            
    def reset_defaults(self):
        """Reset all settings to defaults"""
        if messagebox.askyesno("Reset Settings", "Reset all preferences to default values?"):
            # Reset alert settings
            self.confidence_var.set(80.0)
            self.popup_var.set(True)
            self.sound_var.set(True)
            self.email_var.set(False)
            self.alert_limit_var.set(10)
            
            # Reset pattern settings
            default_patterns = ['head_shoulders', 'double_top', 'double_bottom', 'triangle']
            for pattern, var in self.pattern_vars.items():
                var.set(pattern in default_patterns)
                
            self.primary_timeframe_var.set("5min")
            self.multi_timeframe_var.set(True)
            
            # Reset trading settings
            self.symbols_var.set("AAPL, MSFT, TSLA, GOOGL, AMZN")
            self.stop_loss_var.set(2.0)
            self.take_profit_var.set(4.0)
            self.start_hour_var.set("09:30")
            self.end_hour_var.set("16:00")
            
    def save_additional_preferences(self, prefs):
        """Save additional preferences to file"""
        try:
            os.makedirs('data', exist_ok=True)
            with open('data/user_preferences.json', 'w') as f:
                json.dump(prefs, f, indent=2)
        except Exception as e:
            print(f"Error saving preferences: {e}")
            
    def load_additional_preferences(self):
        """Load additional preferences from file"""
        try:
            if os.path.exists('data/user_preferences.json'):
                with open('data/user_preferences.json', 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading preferences: {e}")
        return {}