"""
Chart visualization frame using matplotlib
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.dates as mdates
import mplfinance as mpf
import pandas as pd
import numpy as np
from datetime import datetime
import os

class ChartFrame(ttk.Frame):
    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        self.current_data = None
        self.pattern_overlays = []
        self.canvas = None
        self.toolbar = None
        self.main_ax = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup chart UI components"""
        # Chart options frame
        options_frame = ttk.Frame(self)
        options_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # Chart type selection
        ttk.Label(options_frame, text="Chart Type:").pack(side=tk.LEFT, padx=5)
        self.chart_type_var = tk.StringVar(value="candlestick")
        chart_type_combo = ttk.Combobox(
            options_frame, 
            textvariable=self.chart_type_var,
            values=["candlestick", "line", "ohlc"],
            state="readonly",
            width=12
        )
        chart_type_combo.pack(side=tk.LEFT, padx=5)
        chart_type_combo.bind("<<ComboboxSelected>>", self.on_chart_type_change)
        
        # Volume display checkbox
        self.show_volume_var = tk.BooleanVar(value=True)
        volume_check = ttk.Checkbutton(
            options_frame,
            text="Volume",
            variable=self.show_volume_var,
            command=self.on_volume_toggle
        )
        volume_check.pack(side=tk.LEFT, padx=10)
        
        # Technical indicators
        self.show_ma_var = tk.BooleanVar(value=True)
        ma_check = ttk.Checkbutton(
            options_frame,
            text="Moving Avg",
            variable=self.show_ma_var,
            command=self.on_ma_toggle
        )
        ma_check.pack(side=tk.LEFT, padx=10)
        
        # Bollinger Bands
        self.show_bb_var = tk.BooleanVar(value=False)
        bb_check = ttk.Checkbutton(
            options_frame,
            text="Bollinger Bands",
            variable=self.show_bb_var,
            command=self.on_bb_toggle
        )
        bb_check.pack(side=tk.LEFT, padx=10)
        
        # Chart refresh button
        refresh_button = ttk.Button(
            options_frame,
            text="🔄 Refresh",
            command=self.refresh_chart,
            width=10
        )
        refresh_button.pack(side=tk.RIGHT, padx=5)
        
        # Chart container
        self.chart_container = ttk.Frame(self)
        self.chart_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initialize empty chart
        self.create_empty_chart()
        
    def create_empty_chart(self):
        """Create an empty chart placeholder"""
        plt.style.use('default')  # Use default matplotlib style
        
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
        ax.set_facecolor('white')
        
        # Empty state message
        ax.text(0.5, 0.5, 
                'Forex Chart Pattern Recognition System\n\n' +
                '📈 No Data Available\n\n' +
                'Steps to get started:\n' +
                '1. Select a currency pair (e.g., EUR/USD)\n' + 
                '2. Choose timeframe and data size\n' +
                '3. Click "Fetch Data" to load forex data\n' +
                '4. Click "Detect Patterns" to analyze patterns',
                horizontalalignment='center', 
                verticalalignment='center',
                transform=ax.transAxes, 
                fontsize=14, 
                alpha=0.7,
                bbox=dict(boxstyle="round,pad=1", facecolor="lightblue", alpha=0.3))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1) 
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Forex Chart - Ready for Data', fontsize=16, pad=20)
        
        plt.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(fig, self.chart_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.chart_container)
        self.toolbar.update()
        
    def update_chart(self, data):
        """Update chart with new data"""
        if data is None or data.empty:
            self.main_window.logger.warning("No data provided to update chart")
            return
            
        self.current_data = data
        
        try:
            # Clear previous chart
            for widget in self.chart_container.winfo_children():
                widget.destroy()
                
            # Create new chart based on type
            chart_type = self.chart_type_var.get()
            
            if chart_type == "candlestick":
                self.create_candlestick_chart(data)
            elif chart_type == "line":
                self.create_line_chart(data)
            else:  # ohlc
                self.create_ohlc_chart(data)
                
            self.main_window.logger.info(f"Chart updated successfully with {len(data)} data points")
                
        except Exception as e:
            error_msg = f"Error updating chart: {str(e)}"
            self.main_window.logger.error(error_msg, exc_info=True)
            messagebox.showerror("Chart Error", error_msg)
            
    def create_candlestick_chart(self, data):
        """Create candlestick chart"""
        try:
            # Prepare data for mplfinance
            plot_data = data.copy()
            plot_data.index = pd.to_datetime(plot_data.index)
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close']
            for col in required_cols:
                if col not in plot_data.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Additional plots
            addplot = []
            
            # Moving average
            if self.show_ma_var.get():
                if 'sma_20' in data.columns:
                    ma20 = data['sma_20'].dropna()
                    if not ma20.empty:
                        addplot.append(mpf.make_addplot(ma20, color='blue', width=1.5, alpha=0.7))
                else:
                    # Calculate MA if not available
                    ma20 = data['close'].rolling(window=20).mean()
                    addplot.append(mpf.make_addplot(ma20, color='blue', width=1.5, alpha=0.7))
            
            # Bollinger Bands
            if self.show_bb_var.get():
                if all(col in data.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                    addplot.append(mpf.make_addplot(data['bb_upper'], color='gray', width=1, alpha=0.5))
                    addplot.append(mpf.make_addplot(data['bb_lower'], color='gray', width=1, alpha=0.5))
                    addplot.append(mpf.make_addplot(data['bb_middle'], color='orange', width=1, alpha=0.7))
            
            # Volume
            show_volume = self.show_volume_var.get() and 'volume' in data.columns
            
            # Create chart
            style = mpf.make_mpf_style(
                base_mpl_style='default',
                marketcolors=mpf.make_marketcolors(
                    up='g', down='r',
                    edge='inherit',
                    wick={'up':'green', 'down':'red'},
                    volume='in'
                ),
                gridstyle='-',
                gridcolor='lightgray',
                facecolor='white'
            )
            
            fig, axes = mpf.plot(
                plot_data,
                type='candle',
                style=style,
                addplot=addplot,
                volume=show_volume,
                figsize=(14, 10),
                returnfig=True,
                panel_ratios=(3, 1) if show_volume else (1,),
                datetime_format='%m/%d %H:%M',
                xrotation=45,
                title=f"Forex Chart - {getattr(self.main_window, 'current_symbol', 'Unknown')}",
                ylabel='Price',
                ylabel_lower='Volume' if show_volume else None
            )
            
            # Add to tkinter
            self.canvas = FigureCanvasTkAgg(fig, self.chart_container)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Navigation toolbar
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.chart_container)
            self.toolbar.update()
            
            # Store axes for pattern overlays
            if isinstance(axes, list):
                self.main_ax = axes[0]  # Main price chart
            else:
                self.main_ax = axes
                
        except Exception as e:
            self.main_window.logger.error(f"Error creating candlestick chart: {str(e)}")
            raise
        
    def create_line_chart(self, data):
        """Create line chart"""
        try:
            fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
            ax.set_facecolor('white')
            
            # Plot close price
            ax.plot(data.index, data['close'], linewidth=2, color='blue', label='Close Price')
            
            # Moving average
            if self.show_ma_var.get():
                if 'sma_20' in data.columns:
                    ma20 = data['sma_20'].dropna()
                    if not ma20.empty:
                        ax.plot(data.index, ma20, linewidth=1.5, color='red', alpha=0.7, label='MA(20)')
                else:
                    ma20 = data['close'].rolling(window=20).mean()
                    ax.plot(data.index, ma20, linewidth=1.5, color='red', alpha=0.7, label='MA(20)')
                    
            # Bollinger Bands
            if self.show_bb_var.get() and all(col in data.columns for col in ['bb_upper', 'bb_lower']):
                ax.plot(data.index, data['bb_upper'], color='gray', alpha=0.5, linestyle='--', label='BB Upper')
                ax.plot(data.index, data['bb_lower'], color='gray', alpha=0.5, linestyle='--', label='BB Lower')
                ax.fill_between(data.index, data['bb_upper'], data['bb_lower'], alpha=0.1, color='gray')
            
            symbol = getattr(self.main_window, 'current_symbol', 'Unknown')
            ax.set_title(f'Forex Line Chart - {symbol}', fontsize=14, pad=20)
            ax.set_ylabel('Price', fontsize=12)
            ax.set_xlabel('Time', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Format x-axis
            if len(data) > 0:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Add to tkinter
            self.canvas = FigureCanvasTkAgg(fig, self.chart_container)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.chart_container)
            self.toolbar.update()
            
            self.main_ax = ax
            
        except Exception as e:
            self.main_window.logger.error(f"Error creating line chart: {str(e)}")
            raise
        
    def create_ohlc_chart(self, data):
        """Create OHLC bar chart"""
        try:
            # Similar to candlestick but with bars
            plot_data = data.copy()
            plot_data.index = pd.to_datetime(plot_data.index)
            
            addplot = []
            if self.show_ma_var.get():
                if 'sma_20' in data.columns:
                    ma20 = data['sma_20'].dropna()
                    if not ma20.empty:
                        addplot.append(mpf.make_addplot(ma20, color='blue', width=1.5))
                else:
                    ma20 = data['close'].rolling(window=20).mean()
                    addplot.append(mpf.make_addplot(ma20, color='blue', width=1.5))
            
            if self.show_bb_var.get() and all(col in data.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                addplot.append(mpf.make_addplot(data['bb_upper'], color='gray', width=1, alpha=0.5))
                addplot.append(mpf.make_addplot(data['bb_lower'], color='gray', width=1, alpha=0.5))
                addplot.append(mpf.make_addplot(data['bb_middle'], color='orange', width=1, alpha=0.7))
            
            show_volume = self.show_volume_var.get() and 'volume' in data.columns
            
            style = mpf.make_mpf_style(
                base_mpl_style='default',
                marketcolors=mpf.make_marketcolors(up='g', down='r'),
                gridstyle='-',
                gridcolor='lightgray',
                facecolor='white'
            )
            
            fig, axes = mpf.plot(
                plot_data,
                type='ohlc',
                style=style,
                addplot=addplot,
                volume=show_volume,
                figsize=(14, 10),
                returnfig=True,
                panel_ratios=(3, 1) if show_volume else (1,),
                title=f"OHLC Chart - {getattr(self.main_window, 'current_symbol', 'Unknown')}"
            )
            
            self.canvas = FigureCanvasTkAgg(fig, self.chart_container)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.chart_container)
            self.toolbar.update()
            
            if isinstance(axes, list):
                self.main_ax = axes[0]
            else:
                self.main_ax = axes
                
        except Exception as e:
            self.main_window.logger.error(f"Error creating OHLC chart: {str(e)}")
            raise
    
    def add_pattern_overlays(self, patterns):
        """Add pattern detection overlays to chart"""
        if not hasattr(self, 'main_ax') or self.main_ax is None or not patterns:
            return
            
        try:
            self.clear_pattern_overlays()
            
            for pattern in patterns:
                self.draw_pattern_overlay(pattern)
                
            self.canvas.draw()
            self.main_window.logger.info(f"Added {len(patterns)} pattern overlays to chart")
            
        except Exception as e:
            self.main_window.logger.error(f"Error adding pattern overlays: {str(e)}")
                
    def draw_pattern_overlay(self, pattern):
        """Draw individual pattern overlay"""
        try:
            pattern_type = pattern.get('type', '')
            
            if 'Head and Shoulders' in pattern_type:
                self.draw_head_shoulders(pattern)
            elif 'Double Top' in pattern_type or 'Double Bottom' in pattern_type:
                self.draw_double_pattern(pattern)
            elif 'Triangle' in pattern_type:
                self.draw_triangle(pattern)
            elif 'Support' in pattern_type or 'Resistance' in pattern_type:
                self.draw_support_resistance(pattern)
                
        except Exception as e:
            self.main_window.logger.error(f"Error drawing pattern overlay: {str(e)}")
            
    def draw_head_shoulders(self, pattern):
        """Draw head and shoulders pattern"""
        if 'points' not in pattern or len(pattern['points']) < 3:
            return
            
        points = pattern['points']
        x_coords = [pd.to_datetime(point['time']) for point in points]
        y_coords = [point['price'] for point in points]
        
        # Draw pattern lines connecting the points
        self.main_ax.plot(x_coords, y_coords, 'r-', linewidth=3, alpha=0.8, label='Head & Shoulders')
        
        # Mark key points
        colors = ['orange', 'red', 'orange']  # Left shoulder, head, right shoulder
        labels = ['LS', 'H', 'RS']
        
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            self.main_ax.plot(x, y, 'o', color=colors[i], markersize=10, markeredgecolor='black', markeredgewidth=1)
            self.main_ax.annotate(labels[i], (x, y), xytext=(0, 15), textcoords='offset points', 
                                ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Draw neckline if available
        if 'neckline' in pattern:
            neckline_y = pattern['neckline']
            xlim = self.main_ax.get_xlim()
            self.main_ax.axhline(y=neckline_y, color='purple', linestyle='--', 
                               linewidth=2, alpha=0.7, label='Neckline')
        
        # Add pattern label
        mid_x = x_coords[1]  # Use head position
        mid_y = max(y_coords) * 1.02
        confidence = pattern.get('confidence', 0)
        
        self.main_ax.text(mid_x, mid_y, f'H&S\n{confidence:.0%}', 
                         fontsize=10, ha='center', va='bottom',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7, edgecolor='black'))
                        
    def draw_double_pattern(self, pattern):
        """Draw double top/bottom pattern"""
        if 'points' not in pattern or len(pattern['points']) < 2:
            return
            
        points = pattern['points'][:2]  # Only first two peaks/valleys
        
        x_coords = [pd.to_datetime(point['time']) for point in points]
        y_coords = [point['price'] for point in points]
        
        # Draw points
        is_double_top = 'Top' in pattern['type']
        color = 'green' if not is_double_top else 'red'
        
        for x, y in zip(x_coords, y_coords):
            self.main_ax.plot(x, y, 'o', color=color, markersize=10, 
                            markeredgecolor='black', markeredgewidth=1)
        
        # Connect the points
        self.main_ax.plot(x_coords, y_coords, color=color, linestyle='--', 
                         linewidth=2, alpha=0.8)
        
        # Add support/resistance line
        level_key = 'resistance_level' if is_double_top else 'support_level'
        if level_key in pattern:
            level_y = pattern[level_key]
            self.main_ax.axhline(y=level_y, color=color, linestyle='-', 
                               linewidth=2, alpha=0.6)
        
        # Add label
        mid_x = x_coords[0] + (x_coords[1] - x_coords[0]) / 2
        mid_y = (y_coords[0] + y_coords[1]) / 2
        label = 'DT' if is_double_top else 'DB'
        confidence = pattern.get('confidence', 0)
        
        offset_y = -20 if is_double_top else 20
        
        self.main_ax.text(mid_x, mid_y, f'{label}\n{confidence:.0%}', 
                         fontsize=10, ha='center', va='center',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7, edgecolor='black'))
    
    def draw_triangle(self, pattern):
        """Draw triangle pattern"""
        if 'upper_trendline' in pattern and 'lower_trendline' in pattern:
            upper_line = pattern['upper_trendline']
            lower_line = pattern['lower_trendline']
            
            # Convert timestamps to datetime objects
            upper_x = [pd.to_datetime(pd.Timestamp.fromtimestamp(t)) for t in upper_line['x_coords']]
            lower_x = [pd.to_datetime(pd.Timestamp.fromtimestamp(t)) for t in lower_line['x_coords']]
            
            # Draw trendlines
            self.main_ax.plot(upper_x, upper_line['y_coords'], 'b-', linewidth=2, alpha=0.8, label='Upper Trendline')
            self.main_ax.plot(lower_x, lower_line['y_coords'], 'b-', linewidth=2, alpha=0.8, label='Lower Trendline')
            
            # Fill triangle area
            if len(upper_x) >= 2 and len(lower_x) >= 2:
                # Create polygon points
                all_x = upper_x + lower_x[::-1]
                all_y = upper_line['y_coords'] + lower_line['y_coords'][::-1]
                
                self.main_ax.fill(all_x, all_y, color='blue', alpha=0.1)
            
            # Add label
            if upper_x and lower_x:
                label_x = min(max(upper_x), max(lower_x))
                label_y = (upper_line['y_coords'][0] + lower_line['y_coords'][0]) / 2
                confidence = pattern.get('confidence', 0)
                
                triangle_type = pattern['type'].replace(' Triangle', '')
                
                self.main_ax.text(label_x, label_y, f'{triangle_type}\n{confidence:.0%}', 
                                fontsize=10, ha='center', va='center',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.7, edgecolor='black'))
    
    def draw_support_resistance(self, pattern):
        """Draw support/resistance levels"""
        level = pattern.get('level', 0)
        if level <= 0:
            return
            
        is_support = 'Support' in pattern['type']
        color = 'green' if is_support else 'red'
        
        # Draw horizontal line across visible area
        xlim = self.main_ax.get_xlim()
        self.main_ax.axhline(y=level, color=color, linestyle='-', 
                           linewidth=3, alpha=0.8)
        
        # Add strength indicator (line thickness and style)
        strength = pattern.get('strength', 1)
        alpha = min(0.9, 0.4 + (strength / 10) * 0.5)
        
        # Add label
        x_pos = xlim[0] + (xlim[1] - xlim[0]) * 0.02  # 2% from left edge
        confidence = pattern.get('confidence', 0)
        
        label_text = f"{'Support' if is_support else 'Resistance'}\n"
        label_text += f"Level: {level:.5f}\n"
        label_text += f"Strength: {strength}/10\n"
        label_text += f"Confidence: {confidence:.0%}"
        
        self.main_ax.text(x_pos, level, label_text, 
                         fontsize=9, ha='left', va='center',
                         bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8, edgecolor='black'))
                        
    def clear_pattern_overlays(self):
        """Clear all pattern overlays from chart"""
        if not hasattr(self, 'main_ax') or self.main_ax is None:
            return
            
        try:
            # Get current lines and remove pattern-related ones
            lines_to_remove = []
            for line in self.main_ax.lines:
                # Check if this is a pattern line by color or other properties
                color = line.get_color()
                if color in ['red', 'green', 'blue', 'orange', 'purple'] and line.get_linewidth() >= 2:
                    lines_to_remove.append(line)
            
            for line in lines_to_remove:
                line.remove()
            
            # Remove pattern-related text annotations
            texts_to_remove = []
            for text in self.main_ax.texts:
                text_content = text.get_text()
                if any(pattern in text_content for pattern in ['H&S', 'DT', 'DB', 'Triangle', 'Support', 'Resistance', 'Ascending', 'Descending', 'Symmetrical']):
                    texts_to_remove.append(text)
            
            for text in texts_to_remove:
                text.remove()
            
            # Remove pattern-related collections (filled areas)
            collections_to_remove = []
            for collection in self.main_ax.collections:
                collections_to_remove.append(collection)
            
            for collection in collections_to_remove:
                collection.remove()
            
            # Redraw canvas
            if self.canvas:
                self.canvas.draw()
                
        except Exception as e:
            self.main_window.logger.error(f"Error clearing pattern overlays: {str(e)}")
    
    def clear_chart(self):
        """Clear chart and show empty state"""
        try:
            self.current_data = None
            self.pattern_overlays = []
            
            # Clear the container and recreate empty chart
            for widget in self.chart_container.winfo_children():
                widget.destroy()
                
            self.create_empty_chart()
            
        except Exception as e:
            self.main_window.logger.error(f"Error clearing chart: {str(e)}")
    
    def refresh_chart(self):
        """Refresh the current chart"""
        if self.current_data is not None:
            self.update_chart(self.current_data)
        else:
            self.main_window.update_status("No data to refresh")
    
    def export_chart(self, filename):
        """Export current chart as image"""
        if not self.canvas:
            raise ValueError("No chart available to export")
            
        try:
            # Get file extension to determine format
            _, ext = os.path.splitext(filename.lower())
            
            if ext == '.pdf':
                self.canvas.print_pdf(filename, dpi=300, bbox_inches='tight')
            elif ext == '.svg':
                self.canvas.print_svg(filename, dpi=300, bbox_inches='tight')
            elif ext in ['.png', '.jpg', '.jpeg']:
                self.canvas.print_png(filename, dpi=300, bbox_inches='tight')
            else:
                # Default to PNG
                self.canvas.print_png(filename, dpi=300, bbox_inches='tight')
                
            self.main_window.logger.info(f"Chart exported successfully to {filename}")
            
        except Exception as e:
            self.main_window.logger.error(f"Error exporting chart: {str(e)}")
            raise
                    
    def on_chart_type_change(self, event=None):
        """Handle chart type change"""
        if self.current_data is not None:
            self.update_chart(self.current_data)
            self.main_window.update_status(f"Chart type changed to {self.chart_type_var.get()}")
            
    def on_volume_toggle(self):
        """Handle volume display toggle"""
        if self.current_data is not None:
            self.update_chart(self.current_data)
            status = "enabled" if self.show_volume_var.get() else "disabled"
            self.main_window.update_status(f"Volume display {status}")
            
    def on_ma_toggle(self):
        """Handle moving average toggle"""
        if self.current_data is not None:
            self.update_chart(self.current_data)
            status = "enabled" if self.show_ma_var.get() else "disabled"
            self.main_window.update_status(f"Moving average {status}")
            
    def on_bb_toggle(self):
        """Handle Bollinger Bands toggle"""
        if self.current_data is not None:
            self.update_chart(self.current_data)
            status = "enabled" if self.show_bb_var.get() else "disabled"
            self.main_window.update_status(f"Bollinger Bands {status}")
