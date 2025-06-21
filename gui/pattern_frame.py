"""
Pattern detection results display frame
"""

import tkinter as tk
from tkinter import ttk
from datetime import datetime

class PatternFrame(ttk.Frame):
    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        self.patterns = []
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup pattern results UI"""
        # Title
        title_label = ttk.Label(self, text="Detected Patterns", font=("Arial", 12, "bold"))
        title_label.pack(pady=5)
        
        # Pattern summary
        summary_frame = ttk.Frame(self)
        summary_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(summary_frame, text="Total Patterns:").pack(side=tk.LEFT)
        self.pattern_count_var = tk.StringVar(value="0")
        ttk.Label(summary_frame, textvariable=self.pattern_count_var, 
                 font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(summary_frame, text="Last Updated:").pack(side=tk.RIGHT, padx=10)
        self.last_updated_var = tk.StringVar(value="Never")
        ttk.Label(summary_frame, textvariable=self.last_updated_var).pack(side=tk.RIGHT)
        
        # Pattern list with scrollbar
        list_frame = ttk.Frame(self)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Treeview for pattern list
        columns = ("Type", "Time", "Confidence", "Status")
        self.pattern_tree = ttk.Treeview(list_frame, columns=columns, show="headings")
        
        # Configure columns
        self.pattern_tree.heading("Type", text="Pattern Type")
        self.pattern_tree.heading("Time", text="Detection Time")
        self.pattern_tree.heading("Confidence", text="Confidence")
        self.pattern_tree.heading("Status", text="Status")
        
        self.pattern_tree.column("Type", width=120)
        self.pattern_tree.column("Time", width=100)
        self.pattern_tree.column("Confidence", width=80)
        self.pattern_tree.column("Status", width=80)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.pattern_tree.yview)
        self.pattern_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack components
        self.pattern_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind selection event
        self.pattern_tree.bind("<<TreeviewSelect>>", self.on_pattern_select)
        
        # Pattern details frame
        details_frame = ttk.LabelFrame(self, text="Pattern Details", padding=5)
        details_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Details text widget
        self.details_text = tk.Text(details_frame, height=6, wrap=tk.WORD)
        details_scrollbar = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, 
                                        command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scrollbar.set)
        
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Action buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Export Patterns", 
                  command=self.export_patterns).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Clear All", 
                  command=self.clear_patterns).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Refresh", 
                  command=self.refresh_patterns).pack(side=tk.RIGHT, padx=2)
        
    def update_patterns(self, patterns):
        """Update pattern list with new detections"""
        self.patterns = patterns
        
        # Clear existing items
        for item in self.pattern_tree.get_children():
            self.pattern_tree.delete(item)
            
        # Add new patterns
        for i, pattern in enumerate(patterns):
            pattern_type = pattern.get('type', 'Unknown')
            detection_time = pattern.get('timestamp', datetime.now().strftime('%H:%M:%S'))
            confidence = f"{pattern.get('confidence', 0):.1%}"
            status = pattern.get('status', 'Active')
            
            # Color coding based on confidence
            confidence_val = pattern.get('confidence', 0)
            if confidence_val >= 0.8:
                tags = ("high_confidence",)
            elif confidence_val >= 0.6:
                tags = ("medium_confidence",)
            else:
                tags = ("low_confidence",)
                
            self.pattern_tree.insert("", tk.END, iid=str(i),
                                   values=(pattern_type, detection_time, confidence, status),
                                   tags=tags)
        
        # Configure tags
        self.pattern_tree.tag_configure("high_confidence", foreground="green")
        self.pattern_tree.tag_configure("medium_confidence", foreground="orange")
        self.pattern_tree.tag_configure("low_confidence", foreground="red")
        
        # Update summary
        self.pattern_count_var.set(str(len(patterns)))
        self.last_updated_var.set(datetime.now().strftime('%H:%M:%S'))
        
        # Select first pattern if available
        if patterns:
            self.pattern_tree.selection_set("0")
            self.on_pattern_select()
            
    def on_pattern_select(self, event=None):
        """Handle pattern selection"""
        selection = self.pattern_tree.selection()
        if not selection:
            self.details_text.delete(1.0, tk.END)
            return
            
        # Get selected pattern
        item_id = selection[0]
        pattern_index = int(item_id)
        
        if pattern_index < len(self.patterns):
            pattern = self.patterns[pattern_index]
            self.show_pattern_details(pattern)
            
    def show_pattern_details(self, pattern):
        """Display detailed information about selected pattern"""
        self.details_text.delete(1.0, tk.END)
        
        details = f"Pattern Type: {pattern.get('type', 'Unknown')}\n"
        details += f"Detection Time: {pattern.get('timestamp', 'Unknown')}\n"
        details += f"Confidence Score: {pattern.get('confidence', 0):.1%}\n"
        details += f"Status: {pattern.get('status', 'Active')}\n\n"
        
        # Additional details based on pattern type
        if 'Head and Shoulders' in pattern.get('type', ''):
            details += "Key Points:\n"
            if 'points' in pattern:
                for i, point in enumerate(pattern['points']):
                    point_name = ['Left Shoulder', 'Head', 'Right Shoulder'][i] if i < 3 else f'Point {i+1}'
                    details += f"  {point_name}: {point.get('price', 0):.5f} at {point.get('time', 'Unknown')}\n"
                    
            if 'neckline' in pattern:
                details += f"\nNeckline Level: {pattern['neckline']:.5f}\n"
                
        elif 'Double' in pattern.get('type', ''):
            details += "Key Levels:\n"
            if 'points' in pattern:
                for i, point in enumerate(pattern['points'][:2]):
                    details += f"  Peak {i+1}: {point.get('price', 0):.5f} at {point.get('time', 'Unknown')}\n"
                    
        elif 'Triangle' in pattern.get('type', ''):
            details += "Triangle Properties:\n"
            if 'upper_trendline' in pattern:
                details += f"  Upper Trendline Slope: {pattern['upper_trendline'].get('slope', 0):.6f}\n"
            if 'lower_trendline' in pattern:
                details += f"  Lower Trendline Slope: {pattern['lower_trendline'].get('slope', 0):.6f}\n"
                
        elif 'Support' in pattern.get('type', '') or 'Resistance' in pattern.get('type', ''):
            details += f"Level: {pattern.get('level', 0):.5f}\n"
            details += f"Strength: {pattern.get('strength', 0)}/10\n"
            details += f"Touch Count: {pattern.get('touch_count', 0)}\n"
            
        # Trading implications
        details += f"\nTrading Signal: {pattern.get('signal', 'Neutral')}\n"
        details += f"Target Price: {pattern.get('target_price', 'N/A')}\n"
        details += f"Stop Loss: {pattern.get('stop_loss', 'N/A')}\n"
        
        # Risk assessment
        risk_level = pattern.get('risk_level', 'Medium')
        details += f"Risk Level: {risk_level}\n"
        
        self.details_text.insert(1.0, details)
        
    def clear_patterns(self):
        """Clear all patterns"""
        self.patterns = []
        
        # Clear treeview
        for item in self.pattern_tree.get_children():
            self.pattern_tree.delete(item)
            
        # Clear details
        self.details_text.delete(1.0, tk.END)
        
        # Update summary
        self.pattern_count_var.set("0")
        self.last_updated_var.set("Never")
        
        # Clear chart overlays
        self.main_window.chart_frame.clear_pattern_overlays()
        
    def refresh_patterns(self):
        """Refresh pattern detection"""
        if self.main_window.current_data is not None:
            self.main_window.detect_patterns()
        else:
            tk.messagebox.showwarning("Warning", "No data available to refresh patterns.")
            
    def export_patterns(self):
        """Export detected patterns to file"""
        if not self.patterns:
            tk.messagebox.showwarning("Warning", "No patterns to export.")
            return
            
        from tkinter import filedialog
        
        filename = filedialog.asksaveasfilename(
            title="Export Patterns",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                if filename.endswith('.json'):
                    import json
                    with open(filename, 'w') as f:
                        json.dump(self.patterns, f, indent=2, default=str)
                else:
                    with open(filename, 'w') as f:
                        f.write("Forex Pattern Detection Results\n")
                        f.write("=" * 40 + "\n\n")
                        
                        for i, pattern in enumerate(self.patterns, 1):
                            f.write(f"Pattern {i}:\n")
                            f.write(f"  Type: {pattern.get('type', 'Unknown')}\n")
                            f.write(f"  Time: {pattern.get('timestamp', 'Unknown')}\n")
                            f.write(f"  Confidence: {pattern.get('confidence', 0):.1%}\n")
                            f.write(f"  Signal: {pattern.get('signal', 'Neutral')}\n")
                            f.write("\n")
                            
                tk.messagebox.showinfo("Success", f"Patterns exported to {filename}")
                
            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to export patterns: {str(e)}")
