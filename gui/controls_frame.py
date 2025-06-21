"""
Controls frame for data fetching and pattern detection
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
from datetime import datetime

class ControlsFrame(ttk.Frame):
    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup control UI components"""
        # Configure frame
        self.configure(width=320)
        self.pack_propagate(False)
        
        # Create scrollable frame
        canvas = tk.Canvas(self, width=300)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Title
        title_label = ttk.Label(scrollable_frame, text="Control Panel", font=("Arial", 14, "bold"))
        title_label.pack(pady=(10,20))
        
        # Create sections
        self.create_data_section(scrollable_frame)
        self.create_pattern_section(scrollable_frame)
        self.create_realtime_section(scrollable_frame)
        self.create_training_section(scrollable_frame)
        self.create_info_section(scrollable_frame)
        
    def create_data_section(self, parent):
        """Create data fetching controls"""
        data_frame = ttk.LabelFrame(parent, text="Data Fetching", padding=15)
        data_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Stock symbol selection
        ttk.Label(data_frame, text="Stock Symbol:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.symbol_var = tk.StringVar(value="AAPL")
        symbol_combo = ttk.Combobox(
            data_frame,
            textvariable=self.symbol_var,
            values=[
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "NVDA",
                "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA",
                "SPY", "QQQ", "IWM", "EFA", "VTI", "BND", "GLD", "SLV",
                "XOM", "CVX", "COP", "PG", "JNJ", "UNH", "HD", "WMT"
            ],
            state="readonly",
            width=18
        )
        symbol_combo.pack(fill=tk.X, pady=(5,10))
        
        # Timeframe selection
        ttk.Label(data_frame, text="Timeframe:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.timeframe_var = tk.StringVar(value="5min")
        timeframe_combo = ttk.Combobox(
            data_frame,
            textvariable=self.timeframe_var,
            values=["1min", "5min", "15min", "30min", "60min", "daily"],
            state="readonly",
            width=18
        )
        timeframe_combo.pack(fill=tk.X, pady=(5,10))
        
        # Output size selection
        ttk.Label(data_frame, text="Data Amount:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.outputsize_var = tk.StringVar(value="compact")
        outputsize_frame = ttk.Frame(data_frame)
        outputsize_frame.pack(fill=tk.X, pady=(5,15))
        
        ttk.Radiobutton(outputsize_frame, text="Compact (100 points)", 
                       variable=self.outputsize_var, value="compact").pack(anchor=tk.W)
        ttk.Radiobutton(outputsize_frame, text="Full (All available)", 
                       variable=self.outputsize_var, value="full").pack(anchor=tk.W)
        
        # Fetch button
        self.fetch_button = ttk.Button(
            data_frame,
            text="Fetch Data",
            command=self.fetch_data,
            style="Accent.TButton"
        )
        self.fetch_button.pack(fill=tk.X, pady=5)
        
        # Quick data info
        self.data_info_var = tk.StringVar(value="No data loaded")
        info_label = ttk.Label(data_frame, textvariable=self.data_info_var, 
                              font=("Arial", 9), foreground="gray")
        info_label.pack(anchor=tk.W, pady=(5,0))
        
    def create_pattern_section(self, parent):
        """Create pattern detection controls"""
        pattern_frame = ttk.LabelFrame(parent, text="Pattern Detection", padding=15)
        pattern_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Pattern types to detect
        ttk.Label(pattern_frame, text="Patterns to Detect:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        patterns_grid = ttk.Frame(pattern_frame)
        patterns_grid.pack(fill=tk.X, pady=(5,10))
        
        self.head_shoulders_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(patterns_grid, text="Head & Shoulders", 
                       variable=self.head_shoulders_var).grid(row=0, column=0, sticky=tk.W)
        
        self.double_top_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(patterns_grid, text="Double Top/Bottom", 
                       variable=self.double_top_var).grid(row=1, column=0, sticky=tk.W)
        
        self.triangle_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(patterns_grid, text="Triangle Patterns", 
                       variable=self.triangle_var).grid(row=2, column=0, sticky=tk.W)
        
        self.support_resistance_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(patterns_grid, text="Support/Resistance", 
                       variable=self.support_resistance_var).grid(row=3, column=0, sticky=tk.W)
        
        # Detection sensitivity
        ttk.Label(pattern_frame, text="Detection Sensitivity:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(15,5))
        
        sensitivity_frame = ttk.Frame(pattern_frame)
        sensitivity_frame.pack(fill=tk.X, pady=(0,10))
        
        self.sensitivity_var = tk.DoubleVar(value=0.015)
        sensitivity_scale = ttk.Scale(
            sensitivity_frame,
            from_=0.005,
            to=0.050,
            variable=self.sensitivity_var,
            orient=tk.HORIZONTAL
        )
        sensitivity_scale.pack(fill=tk.X)
        
        self.sensitivity_label = ttk.Label(sensitivity_frame, text="1.5%", font=("Arial", 9))
        self.sensitivity_label.pack(anchor=tk.W)
        
        def update_sensitivity_label(*args):
            self.sensitivity_label.config(text=f"{self.sensitivity_var.get():.1%}")
        
        self.sensitivity_var.trace("w", update_sensitivity_label)
        
        # Auto-detect option
        self.auto_detect_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(pattern_frame, text="Auto-detect after data fetch", 
                       variable=self.auto_detect_var).pack(anchor=tk.W, pady=(5,10))
        
        # Detect button
        self.detect_button = ttk.Button(
            pattern_frame,
            text="Detect Patterns",
            command=self.detect_patterns,
            style="Accent.TButton"
        )
        self.detect_button.pack(fill=tk.X, pady=5)
        
        # Pattern count display
        self.pattern_count_var = tk.StringVar(value="No patterns detected")
        count_label = ttk.Label(pattern_frame, textvariable=self.pattern_count_var,
                               font=("Arial", 9), foreground="gray")
        count_label.pack(anchor=tk.W, pady=(5,0))
        
    def create_realtime_section(self, parent):
        """Create real-time monitoring controls"""
        realtime_frame = ttk.LabelFrame(parent, text="⏱️ Real-time Monitoring", padding=15)
        realtime_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Status indicator
        self.realtime_status_var = tk.StringVar(value="● Inactive")
        status_label = ttk.Label(realtime_frame, textvariable=self.realtime_status_var,
                                font=("Arial", 10, "bold"))
        status_label.pack(anchor=tk.W, pady=(0,10))
        
        # Update interval
        interval_frame = ttk.Frame(realtime_frame)
        interval_frame.pack(fill=tk.X, pady=(0,10))
        
        ttk.Label(interval_frame, text="Update Interval:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        
        self.interval_var = tk.IntVar(value=60)
        interval_spin = ttk.Spinbox(
            interval_frame,
            from_=10,
            to=300,
            textvariable=self.interval_var,
            width=8,
            increment=10
        )
        interval_spin.pack(side=tk.RIGHT)
        
        ttk.Label(interval_frame, text="seconds").pack(side=tk.RIGHT, padx=(0,5))
        
        # Real-time controls
        button_frame = ttk.Frame(realtime_frame)
        button_frame.pack(fill=tk.X, pady=(0,10))
        
        self.start_realtime_button = ttk.Button(
            button_frame,
            text="▶️ Start",
            command=self.start_realtime,
            width=8
        )
        self.start_realtime_button.pack(side=tk.LEFT, padx=(0,5))
        
        self.stop_realtime_button = ttk.Button(
            button_frame,
            text="⏸️ Stop",
            command=self.stop_realtime,
            state=tk.DISABLED,
            width=8
        )
        self.stop_realtime_button.pack(side=tk.RIGHT)
        
        # Alert settings
        alert_frame = ttk.LabelFrame(realtime_frame, text="Alert Settings", padding=10)
        alert_frame.pack(fill=tk.X, pady=(10,0))
        
        self.popup_alerts_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(alert_frame, text="🔔 Popup Alerts", 
                       variable=self.popup_alerts_var).pack(anchor=tk.W)
        
        self.sound_alerts_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(alert_frame, text="🔊 Sound Alerts", 
                       variable=self.sound_alerts_var).pack(anchor=tk.W)
        
        # Update real-time configuration
        def update_realtime_config():
            config = {
                'real_time_alerts': self.popup_alerts_var.get(),
                'sound_alerts': self.sound_alerts_var.get()
            }
            self.main_window.config.update(config)
        
        self.popup_alerts_var.trace("w", lambda *args: update_realtime_config())
        self.sound_alerts_var.trace("w", lambda *args: update_realtime_config())
        
    def create_training_section(self, parent):
        """Create model training controls"""
        training_frame = ttk.LabelFrame(parent, text="🤖 AI Model Training", padding=15)
        training_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Training status
        self.training_status_var = tk.StringVar(value="✅ Model Ready")
        status_label = ttk.Label(training_frame, textvariable=self.training_status_var, 
                                font=("Arial", 10))
        status_label.pack(anchor=tk.W, pady=(0,10))
        
        # Progress bar
        self.training_progress = ttk.Progressbar(training_frame, mode='indeterminate')
        self.training_progress.pack(fill=tk.X, pady=(0,10))
        
        # Training buttons
        button_frame = ttk.Frame(training_frame)
        button_frame.pack(fill=tk.X, pady=(0,10))
        
        self.retrain_button = ttk.Button(
            button_frame,
            text="🔄 Retrain",
            command=self.retrain_model,
            width=10
        )
        self.retrain_button.pack(side=tk.LEFT, padx=(0,5))
        
        self.validate_button = ttk.Button(
            button_frame,
            text="✅ Validate",
            command=self.validate_model,
            width=10
        )
        self.validate_button.pack(side=tk.RIGHT)
        
        # Model metrics
        metrics_frame = ttk.LabelFrame(training_frame, text="Model Performance", padding=8)
        metrics_frame.pack(fill=tk.X)
        
        # Accuracy
        acc_frame = ttk.Frame(metrics_frame)
        acc_frame.pack(fill=tk.X, pady=2)
        ttk.Label(acc_frame, text="Accuracy:", font=("Arial", 9)).pack(side=tk.LEFT)
        self.accuracy_var = tk.StringVar(value="85.2%")
        ttk.Label(acc_frame, textvariable=self.accuracy_var, 
                 foreground="green", font=("Arial", 9, "bold")).pack(side=tk.RIGHT)
        
        # Confidence
        conf_frame = ttk.Frame(metrics_frame)
        conf_frame.pack(fill=tk.X, pady=2)
        ttk.Label(conf_frame, text="Avg Confidence:", font=("Arial", 9)).pack(side=tk.LEFT)
        self.confidence_var = tk.StringVar(value="78.5%")
        ttk.Label(conf_frame, textvariable=self.confidence_var, 
                 foreground="blue", font=("Arial", 9, "bold")).pack(side=tk.RIGHT)
        
    def create_info_section(self, parent):
        """Create information and quick actions section"""
        info_frame = ttk.LabelFrame(parent, text="ℹ️ Quick Info & Actions", padding=15)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Quick stats
        stats_frame = ttk.Frame(info_frame)
        stats_frame.pack(fill=tk.X, pady=(0,10))
        
        # Current symbol
        symbol_frame = ttk.Frame(stats_frame)
        symbol_frame.pack(fill=tk.X, pady=2)
        ttk.Label(symbol_frame, text="Current Pair:", font=("Arial", 9)).pack(side=tk.LEFT)
        self.current_symbol_var = tk.StringVar(value="None")
        ttk.Label(symbol_frame, textvariable=self.current_symbol_var, 
                 font=("Arial", 9, "bold")).pack(side=tk.RIGHT)
        
        # Data points
        data_frame = ttk.Frame(stats_frame)
        data_frame.pack(fill=tk.X, pady=2)
        ttk.Label(data_frame, text="Data Points:", font=("Arial", 9)).pack(side=tk.LEFT)
        self.data_points_var = tk.StringVar(value="0")
        ttk.Label(data_frame, textvariable=self.data_points_var, 
                 font=("Arial", 9, "bold")).pack(side=tk.RIGHT)
        
        # Last update
        update_frame = ttk.Frame(stats_frame)
        update_frame.pack(fill=tk.X, pady=2)
        ttk.Label(update_frame, text="Last Update:", font=("Arial", 9)).pack(side=tk.LEFT)
        self.last_update_var = tk.StringVar(value="Never")
        ttk.Label(update_frame, textvariable=self.last_update_var, 
                 font=("Arial", 9, "bold")).pack(side=tk.RIGHT)
        
        # Quick action buttons
        actions_frame = ttk.Frame(info_frame)
        actions_frame.pack(fill=tk.X, pady=(10,0))
        
        # Row 1
        row1 = ttk.Frame(actions_frame)
        row1.pack(fill=tk.X, pady=(0,5))
        
        ttk.Button(row1, text="🔄 Refresh", command=self.refresh_data, 
                  width=12).pack(side=tk.LEFT, padx=(0,2))
        ttk.Button(row1, text="🗑️ Clear", command=self.clear_data, 
                  width=12).pack(side=tk.RIGHT, padx=(2,0))
        
        # Row 2
        row2 = ttk.Frame(actions_frame)
        row2.pack(fill=tk.X)
        
        ttk.Button(row2, text="💾 Save", command=self.save_results, 
                  width=12).pack(side=tk.LEFT, padx=(0,2))
        ttk.Button(row2, text="📋 Export", command=self.export_chart, 
                  width=12).pack(side=tk.RIGHT, padx=(2,0))
        
        # Initialize info display
        self.update_info_display()
    
    def update_info_display(self):
        """Update the information display"""
        try:
            # Update current symbol
            if hasattr(self.main_window, 'current_symbol'):
                self.current_symbol_var.set(self.main_window.current_symbol)
            else:
                self.current_symbol_var.set("None")
            
            # Update data points
            if self.main_window.current_data is not None:
                self.data_points_var.set(str(len(self.main_window.current_data)))
                self.data_info_var.set(f"Loaded: {len(self.main_window.current_data)} points")
            else:
                self.data_points_var.set("0")
                self.data_info_var.set("No data loaded")
            
            # Update pattern count
            pattern_count = len(self.main_window.detected_patterns)
            if pattern_count > 0:
                high_conf = sum(1 for p in self.main_window.detected_patterns if p.get('confidence', 0) >= 0.8)
                self.pattern_count_var.set(f"{pattern_count} patterns ({high_conf} high confidence)")
            else:
                self.pattern_count_var.set("No patterns detected")
            
            # Update last update time
            self.last_update_var.set(datetime.now().strftime('%H:%M:%S'))
            
            # Schedule next update
            self.after(5000, self.update_info_display)  # Update every 5 seconds
            
        except Exception as e:
            self.main_window.logger.error(f"Error updating info display: {str(e)}")
        
    def fetch_data(self):
        """Fetch forex data"""
        symbol = self.symbol_var.get()
        timeframe = self.timeframe_var.get()
        outputsize = self.outputsize_var.get()
        
        if not symbol:
            messagebox.showwarning("No Symbol", "Please select a currency pair.")
            return
        
        # Disable button during fetch
        self.fetch_button.config(state=tk.DISABLED, text="📊 Fetching...")
        
        def fetch_worker():
            try:
                success = self.main_window.fetch_data(symbol, timeframe, outputsize)
                
                if success and self.auto_detect_var.get():
                    # Auto-detect patterns after successful data fetch
                    self.main_window.root.after(1000, self.detect_patterns)
                    
            except Exception as e:
                self.main_window.logger.error(f"Error in fetch worker: {str(e)}")
            finally:
                # Re-enable button
                self.fetch_button.config(state=tk.NORMAL, text="📈 Fetch Data")
                
        # Run in separate thread
        thread = threading.Thread(target=fetch_worker, daemon=True)
        thread.start()
        
    def detect_patterns(self):
        """Detect patterns in current data"""
        if self.main_window.current_data is None:
            messagebox.showwarning("No Data", "No data available for pattern detection. Please fetch data first.")
            return
            
        # Disable button during detection
        self.detect_button.config(state=tk.DISABLED, text="🔍 Analyzing...")
        
        def detect_worker():
            try:
                # Configure detection parameters
                pattern_config = {
                    'head_shoulders': self.head_shoulders_var.get(),
                    'double_patterns': self.double_top_var.get(),
                    'triangles': self.triangle_var.get(),
                    'support_resistance': self.support_resistance_var.get(),
                    'sensitivity': self.sensitivity_var.get()
                }
                
                self.main_window.pattern_detector.configure(pattern_config)
                self.main_window.detect_patterns()
                
            except Exception as e:
                self.main_window.logger.error(f"Error in pattern detection worker: {str(e)}")
                self.main_window.root.after(0, lambda: messagebox.showerror("Detection Error", f"Pattern detection failed: {str(e)}"))
            finally:
                # Re-enable button
                self.detect_button.config(state=tk.NORMAL, text="🔍 Detect Patterns")
                
        thread = threading.Thread(target=detect_worker, daemon=True)
        thread.start()
        
    def start_realtime(self):
        """Start real-time monitoring"""
        symbol = self.symbol_var.get()
        interval = self.interval_var.get()
        
        if not symbol:
            messagebox.showwarning("No Symbol", "Please select a currency pair for real-time monitoring.")
            return
        
        try:
            self.main_window.start_real_time(symbol, interval)
            
            # Update UI
            self.realtime_status_var.set("🟢 Active")
            self.start_realtime_button.config(state=tk.DISABLED)
            self.stop_realtime_button.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Real-time Error", f"Failed to start real-time monitoring: {str(e)}")
        
    def stop_realtime(self):
        """Stop real-time monitoring"""
        try:
            self.main_window.stop_real_time()
            
            # Update UI
            self.realtime_status_var.set("🔴 Inactive")
            self.start_realtime_button.config(state=tk.NORMAL)
            self.stop_realtime_button.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Real-time Error", f"Failed to stop real-time monitoring: {str(e)}")
        
    def retrain_model(self):
        """Retrain the pattern detection model"""
        if self.main_window.current_data is None:
            messagebox.showwarning("No Data", "No data available for model training. Please fetch data first.")
            return
            
        # Disable button and start progress
        self.retrain_button.config(state=tk.DISABLED, text="🔄 Training...")
        self.training_progress.start()
        self.training_status_var.set("🔄 Training in progress...")
        
        def train_worker():
            try:
                # Train model with current data
                accuracy = self.main_window.pattern_detector.retrain_model(
                    self.main_window.current_data
                )
                
                # Update UI in main thread
                self.main_window.root.after(0, lambda: self._update_training_success(accuracy))
                
            except Exception as e:
                error_msg = f"Model training failed: {str(e)}"
                self.main_window.logger.error(error_msg, exc_info=True)
                self.main_window.root.after(0, lambda: self._update_training_error(error_msg))
                
        thread = threading.Thread(target=train_worker, daemon=True)
        thread.start()
    
    def _update_training_success(self, accuracy):
        """Update UI after successful training"""
        self.accuracy_var.set(f"{accuracy:.1%}")
        self.training_status_var.set("✅ Training completed successfully")
        self.retrain_button.config(state=tk.NORMAL, text="🔄 Retrain")
        self.training_progress.stop()
        
        messagebox.showinfo("Training Complete", f"Model training completed successfully!\nNew accuracy: {accuracy:.1%}")
    
    def _update_training_error(self, error_msg):
        """Update UI after training error"""
        self.training_status_var.set("❌ Training failed")
        self.retrain_button.config(state=tk.NORMAL, text="🔄 Retrain")
        self.training_progress.stop()
        
        messagebox.showerror("Training Failed", error_msg)
        
    def validate_model(self):
        """Validate current model performance"""
        if self.main_window.current_data is None:
            messagebox.showwarning("No Data", "No data available for model validation. Please fetch data first.")
            return
            
        def validate_worker():
            try:
                self.validate_button.config(state=tk.DISABLED, text="✅ Validating...")
                
                metrics = self.main_window.pattern_detector.validate_model(
                    self.main_window.current_data
                )
                
                # Update UI
                self.accuracy_var.set(f"{metrics.get('accuracy', 0):.1%}")
                self.confidence_var.set(f"{metrics.get('precision', 0):.1%}")
                
                # Show validation results
                result_text = f"Model Validation Results:\n\n"
                result_text += f"Accuracy: {metrics.get('accuracy', 0):.1%}\n"
                result_text += f"Precision: {metrics.get('precision', 0):.3f}\n"
                result_text += f"Recall: {metrics.get('recall', 0):.3f}\n"
                result_text += f"F1-Score: {metrics.get('f1_score', 0):.3f}\n"
                result_text += f"Patterns Detected: {metrics.get('patterns_detected', 0)}"
                
                messagebox.showinfo("Validation Results", result_text)
                
            except Exception as e:
                error_msg = f"Model validation failed: {str(e)}"
                self.main_window.logger.error(error_msg, exc_info=True)
                messagebox.showerror("Validation Error", error_msg)
            finally:
                self.validate_button.config(state=tk.NORMAL, text="✅ Validate")
                
        thread = threading.Thread(target=validate_worker, daemon=True)
        thread.start()
    
    def refresh_data(self):
        """Refresh current data"""
        if hasattr(self.main_window, 'current_symbol') and self.main_window.current_symbol:
            self.symbol_var.set(self.main_window.current_symbol)
            self.fetch_data()
        else:
            messagebox.showinfo("No Data", "No current data to refresh. Please fetch data first.")
    
    def clear_data(self):
        """Clear all data"""
        if messagebox.askyesno("Clear Data", "Are you sure you want to clear all data and patterns?"):
            self.main_window.clear_all_data()
            self.update_info_display()
    
    def save_results(self):
        """Save pattern results"""
        self.main_window.save_results_json()
    
    def export_chart(self):
        """Export current chart"""
        self.main_window.export_chart()
