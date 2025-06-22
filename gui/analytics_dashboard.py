"""
Pattern History & Analytics Dashboard
Tracks pattern performance and provides actionable insights
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict, Any

class AnalyticsDashboard:
    def __init__(self, parent, alert_system):
        self.parent = parent
        self.alert_system = alert_system
        self.dashboard_window = None
        
    def show(self):
        """Show analytics dashboard"""
        self.dashboard_window = tk.Toplevel(self.parent)
        self.dashboard_window.title("Pattern Analytics Dashboard")
        self.dashboard_window.geometry("1000x700")
        
        self.setup_ui()
        self.load_analytics_data()
        
    def setup_ui(self):
        """Setup dashboard UI"""
        # Main container
        main_frame = ttk.Frame(self.dashboard_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for different analytics views
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Pattern History Tab
        self.create_history_tab(notebook)
        
        # Performance Analytics Tab
        self.create_performance_tab(notebook)
        
        # Symbol Analytics Tab
        self.create_symbol_tab(notebook)
        
        # Export Tab
        self.create_export_tab(notebook)
        
    def create_history_tab(self, notebook):
        """Create pattern history tab"""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="Pattern History")
        
        # Filter controls
        filter_frame = ttk.LabelFrame(frame, text="Filters", padding="10")
        filter_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Date range filter
        date_frame = ttk.Frame(filter_frame)
        date_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(date_frame, text="Date Range:").pack(side=tk.LEFT)
        
        self.start_date_var = tk.StringVar(value=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"))
        start_date_entry = ttk.Entry(date_frame, textvariable=self.start_date_var, width=12)
        start_date_entry.pack(side=tk.LEFT, padx=(10, 5))
        
        ttk.Label(date_frame, text="to").pack(side=tk.LEFT, padx=5)
        
        self.end_date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        end_date_entry = ttk.Entry(date_frame, textvariable=self.end_date_var, width=12)
        end_date_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(date_frame, text="Apply Filter", command=self.apply_filters).pack(side=tk.LEFT, padx=(20, 0))
        
        # Pattern type filter
        pattern_frame = ttk.Frame(filter_frame)
        pattern_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(pattern_frame, text="Pattern Type:").pack(side=tk.LEFT)
        self.pattern_filter_var = tk.StringVar(value="All")
        pattern_filter_combo = ttk.Combobox(pattern_frame, textvariable=self.pattern_filter_var,
                                          values=["All", "Head & Shoulders", "Double Top", "Double Bottom", 
                                                 "Triangle", "Wedge", "Flag", "Support/Resistance"],
                                          state="readonly", width=20)
        pattern_filter_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # History table
        table_frame = ttk.LabelFrame(frame, text="Pattern Detection History", padding="10")
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for history table
        columns = ("Date", "Symbol", "Pattern", "Confidence", "Action", "Outcome")
        self.history_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=15)
        
        # Configure columns
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=120)
            
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.history_tree.xview)
        self.history_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        self.history_tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Bind double-click to show pattern details
        self.history_tree.bind("<Double-1>", self.show_pattern_details)
        
    def create_performance_tab(self, notebook):
        """Create performance analytics tab"""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="Performance Analytics")
        
        # Summary statistics
        stats_frame = ttk.LabelFrame(frame, text="Performance Summary", padding="10")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create statistics display
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X)
        
        # Row 1: Total patterns, accuracy, best pattern
        ttk.Label(stats_grid, text="Total Patterns Detected:").grid(row=0, column=0, sticky="w", padx=(0, 20))
        self.total_patterns_label = ttk.Label(stats_grid, text="0", font=("Arial", 10, "bold"))
        self.total_patterns_label.grid(row=0, column=1, sticky="w")
        
        ttk.Label(stats_grid, text="Overall Accuracy:").grid(row=0, column=2, sticky="w", padx=(40, 20))
        self.accuracy_label = ttk.Label(stats_grid, text="0%", font=("Arial", 10, "bold"))
        self.accuracy_label.grid(row=0, column=3, sticky="w")
        
        # Row 2: Best performing pattern, average confidence
        ttk.Label(stats_grid, text="Best Pattern Type:").grid(row=1, column=0, sticky="w", padx=(0, 20), pady=(10, 0))
        self.best_pattern_label = ttk.Label(stats_grid, text="N/A", font=("Arial", 10, "bold"))
        self.best_pattern_label.grid(row=1, column=1, sticky="w", pady=(10, 0))
        
        ttk.Label(stats_grid, text="Avg Confidence:").grid(row=1, column=2, sticky="w", padx=(40, 20), pady=(10, 0))
        self.avg_confidence_label = ttk.Label(stats_grid, text="0%", font=("Arial", 10, "bold"))
        self.avg_confidence_label.grid(row=1, column=3, sticky="w", pady=(10, 0))
        
        # Charts frame
        charts_frame = ttk.Frame(frame)
        charts_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left chart: Pattern accuracy by type
        left_chart_frame = ttk.LabelFrame(charts_frame, text="Accuracy by Pattern Type", padding="5")
        left_chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Right chart: Confidence distribution
        right_chart_frame = ttk.LabelFrame(charts_frame, text="Confidence Score Distribution", padding="5")
        right_chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Create placeholder charts
        self.create_accuracy_chart(left_chart_frame)
        self.create_confidence_chart(right_chart_frame)
        
    def create_symbol_tab(self, notebook):
        """Create symbol-specific analytics tab"""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="Symbol Analytics")
        
        # Symbol selector
        selector_frame = ttk.Frame(frame)
        selector_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(selector_frame, text="Select Symbol:").pack(side=tk.LEFT)
        self.symbol_var = tk.StringVar(value="AAPL")
        symbol_combo = ttk.Combobox(selector_frame, textvariable=self.symbol_var,
                                   values=["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "META", "NVDA"],
                                   state="readonly", width=15)
        symbol_combo.pack(side=tk.LEFT, padx=(10, 0))
        symbol_combo.bind("<<ComboboxSelected>>", self.update_symbol_analytics)
        
        # Symbol statistics
        symbol_stats_frame = ttk.LabelFrame(frame, text="Symbol Statistics", padding="10")
        symbol_stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create symbol stats grid
        symbol_grid = ttk.Frame(symbol_stats_frame)
        symbol_grid.pack(fill=tk.X)
        
        ttk.Label(symbol_grid, text="Patterns Detected:").grid(row=0, column=0, sticky="w", padx=(0, 20))
        self.symbol_patterns_label = ttk.Label(symbol_grid, text="0", font=("Arial", 10, "bold"))
        self.symbol_patterns_label.grid(row=0, column=1, sticky="w")
        
        ttk.Label(symbol_grid, text="Success Rate:").grid(row=0, column=2, sticky="w", padx=(40, 20))
        self.symbol_success_label = ttk.Label(symbol_grid, text="0%", font=("Arial", 10, "bold"))
        self.symbol_success_label.grid(row=0, column=3, sticky="w")
        
        ttk.Label(symbol_grid, text="Most Common Pattern:").grid(row=1, column=0, sticky="w", padx=(0, 20), pady=(10, 0))
        self.symbol_common_pattern_label = ttk.Label(symbol_grid, text="N/A", font=("Arial", 10, "bold"))
        self.symbol_common_pattern_label.grid(row=1, column=1, sticky="w", pady=(10, 0))
        
        # Symbol chart
        symbol_chart_frame = ttk.LabelFrame(frame, text="Pattern Timeline", padding="5")
        symbol_chart_frame.pack(fill=tk.BOTH, expand=True)
        
        self.create_symbol_timeline_chart(symbol_chart_frame)
        
    def create_export_tab(self, notebook):
        """Create export and reporting tab"""
        frame = ttk.Frame(notebook, padding="20")
        notebook.add(frame, text="Export & Reports")
        
        # Export options
        export_frame = ttk.LabelFrame(frame, text="Export Options", padding="15")
        export_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Export format selection
        format_frame = ttk.Frame(export_frame)
        format_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(format_frame, text="Export Format:").pack(side=tk.LEFT)
        self.export_format_var = tk.StringVar(value="CSV")
        format_combo = ttk.Combobox(format_frame, textvariable=self.export_format_var,
                                   values=["CSV", "JSON", "PDF Report"],
                                   state="readonly", width=15)
        format_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Export buttons
        button_frame = ttk.Frame(export_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Export Pattern History", 
                  command=self.export_pattern_history).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Export Analytics Report", 
                  command=self.export_analytics).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Generate PDF Summary", 
                  command=self.generate_pdf_report).pack(side=tk.LEFT)
        
        # Recent exports
        recent_frame = ttk.LabelFrame(frame, text="Recent Exports", padding="15")
        recent_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create listbox for recent exports
        self.recent_exports_listbox = tk.Listbox(recent_frame, height=10)
        self.recent_exports_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Load recent exports
        self.load_recent_exports()
        
    def create_accuracy_chart(self, parent):
        """Create pattern accuracy chart"""
        fig, ax = plt.subplots(figsize=(5, 4))
        
        # Sample data - replace with real analytics
        patterns = ['Head & Shoulders', 'Double Top', 'Double Bottom', 'Triangle']
        accuracy = [78, 82, 75, 68]
        
        bars = ax.bar(patterns, accuracy, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Pattern Type Accuracy')
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracy):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{acc}%', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_confidence_chart(self, parent):
        """Create confidence distribution chart"""
        fig, ax = plt.subplots(figsize=(5, 4))
        
        # Sample data - replace with real analytics
        confidence_ranges = ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
        counts = [5, 12, 25, 18, 8]
        
        ax.pie(counts, labels=confidence_ranges, autopct='%1.1f%%', startangle=90)
        ax.set_title('Confidence Score Distribution')
        
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_symbol_timeline_chart(self, parent):
        """Create symbol-specific timeline chart"""
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Sample timeline data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        pattern_counts = np.random.poisson(2, 30)
        
        ax.plot(dates, pattern_counts, marker='o', linewidth=2, markersize=4)
        ax.set_xlabel('Date')
        ax.set_ylabel('Patterns Detected')
        ax.set_title('Pattern Detection Timeline')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def load_analytics_data(self):
        """Load and display analytics data"""
        # Load pattern history from alert system
        alerts = self.alert_system.get_alerts_history()
        
        # Update statistics
        self.update_statistics(alerts)
        
        # Populate history table
        self.populate_history_table(alerts)
        
    def update_statistics(self, alerts):
        """Update summary statistics"""
        if not alerts:
            return
            
        total_patterns = len(alerts)
        avg_confidence = sum(alert.confidence for alert in alerts) / total_patterns
        
        # Count pattern types
        pattern_counts = {}
        for alert in alerts:
            pattern_type = alert.pattern_type.replace('_', ' ').title()
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            
        best_pattern = max(pattern_counts.items(), key=lambda x: x[1])[0] if pattern_counts else "N/A"
        
        # Update labels
        self.total_patterns_label.config(text=str(total_patterns))
        self.accuracy_label.config(text="85%")  # Placeholder - implement real tracking
        self.best_pattern_label.config(text=best_pattern)
        self.avg_confidence_label.config(text=f"{avg_confidence:.1f}%")
        
    def populate_history_table(self, alerts):
        """Populate the history table with alert data"""
        # Clear existing items
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
            
        # Add alerts to table
        for alert in reversed(alerts[-50:]):  # Show last 50 alerts
            self.history_tree.insert("", "end", values=(
                alert.timestamp.strftime("%Y-%m-%d %H:%M"),
                alert.symbol,
                alert.pattern_type.replace('_', ' ').title(),
                f"{alert.confidence:.1f}%",
                alert.suggested_action,
                "Pending"  # Placeholder for outcome tracking
            ))
            
    def apply_filters(self):
        """Apply date and pattern filters"""
        # Implement filtering logic
        self.load_analytics_data()
        
    def update_symbol_analytics(self, event=None):
        """Update symbol-specific analytics"""
        selected_symbol = self.symbol_var.get()
        
        # Filter alerts for selected symbol
        alerts = [alert for alert in self.alert_system.get_alerts_history() 
                 if alert.symbol == selected_symbol]
                 
        if alerts:
            symbol_patterns = len(alerts)
            symbol_success = 85  # Placeholder
            
            # Find most common pattern for this symbol
            pattern_counts = {}
            for alert in alerts:
                pattern_type = alert.pattern_type.replace('_', ' ').title()
                pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
                
            most_common = max(pattern_counts.items(), key=lambda x: x[1])[0] if pattern_counts else "N/A"
            
            self.symbol_patterns_label.config(text=str(symbol_patterns))
            self.symbol_success_label.config(text=f"{symbol_success}%")
            self.symbol_common_pattern_label.config(text=most_common)
        else:
            self.symbol_patterns_label.config(text="0")
            self.symbol_success_label.config(text="N/A")
            self.symbol_common_pattern_label.config(text="N/A")
            
    def show_pattern_details(self, event):
        """Show detailed pattern information"""
        selection = self.history_tree.selection()
        if selection:
            item = self.history_tree.item(selection[0])
            values = item['values']
            
            detail_window = tk.Toplevel(self.dashboard_window)
            detail_window.title("Pattern Details")
            detail_window.geometry("400x300")
            
            # Display pattern information
            info_frame = ttk.Frame(detail_window, padding="20")
            info_frame.pack(fill=tk.BOTH, expand=True)
            
            ttk.Label(info_frame, text="Pattern Details", font=("Arial", 14, "bold")).pack(pady=(0, 15))
            
            details = [
                ("Date:", values[0]),
                ("Symbol:", values[1]),
                ("Pattern Type:", values[2]),
                ("Confidence:", values[3]),
                ("Suggested Action:", values[4]),
                ("Outcome:", values[5])
            ]
            
            for label, value in details:
                row_frame = ttk.Frame(info_frame)
                row_frame.pack(fill=tk.X, pady=2)
                ttk.Label(row_frame, text=label, font=("Arial", 10, "bold")).pack(side=tk.LEFT)
                ttk.Label(row_frame, text=str(value)).pack(side=tk.LEFT, padx=(10, 0))
                
    def export_pattern_history(self):
        """Export pattern history"""
        try:
            format_type = self.export_format_var.get()
            alerts = self.alert_system.get_alerts_history()
            
            if format_type == "CSV":
                self.export_to_csv(alerts)
            elif format_type == "JSON":
                self.export_to_json(alerts)
                
            messagebox.showinfo("Export Complete", f"Pattern history exported as {format_type}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {str(e)}")
            
    def export_to_csv(self, alerts):
        """Export alerts to CSV format"""
        import csv
        
        filename = f"pattern_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join('exports', filename)
        
        os.makedirs('exports', exist_ok=True)
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Date', 'Symbol', 'Pattern Type', 'Confidence', 'Suggested Action'])
            
            for alert in alerts:
                writer.writerow([
                    alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    alert.symbol,
                    alert.pattern_type.replace('_', ' ').title(),
                    f"{alert.confidence:.1f}%",
                    alert.suggested_action
                ])
                
    def export_to_json(self, alerts):
        """Export alerts to JSON format"""
        filename = f"pattern_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join('exports', filename)
        
        os.makedirs('exports', exist_ok=True)
        
        export_data = []
        for alert in alerts:
            export_data.append({
                'timestamp': alert.timestamp.isoformat(),
                'symbol': alert.symbol,
                'pattern_type': alert.pattern_type,
                'confidence': alert.confidence,
                'suggested_action': alert.suggested_action,
                'pattern_data': alert.pattern_data
            })
            
        with open(filepath, 'w') as jsonfile:
            json.dump(export_data, jsonfile, indent=2)
            
    def export_analytics(self):
        """Export analytics summary"""
        messagebox.showinfo("Export Analytics", "Analytics export feature coming soon!")
        
    def generate_pdf_report(self):
        """Generate comprehensive PDF report"""
        messagebox.showinfo("PDF Report", "PDF report generation feature coming soon!")
        
    def load_recent_exports(self):
        """Load list of recent exports"""
        if os.path.exists('exports'):
            exports = os.listdir('exports')
            exports.sort(reverse=True)
            
            for export_file in exports[:10]:  # Show last 10 exports
                self.recent_exports_listbox.insert(tk.END, export_file)