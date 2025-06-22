"""
Intelligent Alert System for Pattern Detection
Non-intrusive notifications with confidence scoring
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from datetime import datetime
from typing import List, Dict, Callable
import json
import os

class PatternAlert:
    def __init__(self, pattern_type: str, symbol: str, confidence: float, 
                 timestamp: datetime, suggested_action: str, pattern_data: Dict):
        self.pattern_type = pattern_type
        self.symbol = symbol
        self.confidence = confidence
        self.timestamp = timestamp
        self.suggested_action = suggested_action
        self.pattern_data = pattern_data
        self.id = f"{symbol}_{pattern_type}_{int(timestamp.timestamp())}"

class AlertSystem:
    def __init__(self, main_window):
        self.main_window = main_window
        self.alerts_history = []
        self.alert_callbacks = []
        self.notification_settings = {
            'min_confidence': 80.0,
            'enabled_patterns': ['head_shoulders', 'double_top', 'double_bottom', 'triangle'],
            'sound_enabled': True,
            'popup_enabled': True,
            'email_enabled': False
        }
        self.load_settings()
        
    def add_alert_callback(self, callback: Callable):
        """Add callback function for new alerts"""
        self.alert_callbacks.append(callback)
        
    def create_pattern_alert(self, pattern_type: str, symbol: str, confidence: float, 
                           suggested_action: str, pattern_data: Dict) -> PatternAlert:
        """Create a new pattern alert"""
        alert = PatternAlert(
            pattern_type=pattern_type,
            symbol=symbol,
            confidence=confidence,
            timestamp=datetime.now(),
            suggested_action=suggested_action,
            pattern_data=pattern_data
        )
        
        # Check if alert meets criteria
        if self.should_trigger_alert(alert):
            self.trigger_alert(alert)
            
        return alert
        
    def should_trigger_alert(self, alert: PatternAlert) -> bool:
        """Check if alert meets user criteria"""
        return (alert.confidence >= self.notification_settings['min_confidence'] and
                alert.pattern_type in self.notification_settings['enabled_patterns'])
                
    def trigger_alert(self, alert: PatternAlert):
        """Trigger alert notifications"""
        # Add to history
        self.alerts_history.append(alert)
        
        # Save to file
        self.save_alert_history()
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Alert callback error: {e}")
                
        # Show popup if enabled
        if self.notification_settings['popup_enabled']:
            self.show_alert_popup(alert)
            
        # Play sound if enabled
        if self.notification_settings['sound_enabled']:
            self.play_alert_sound()
            
    def show_alert_popup(self, alert: PatternAlert):
        """Show non-intrusive alert popup"""
        def show_popup():
            popup = tk.Toplevel(self.main_window.root)
            popup.title("Pattern Alert")
            popup.geometry("400x250")
            popup.resizable(False, False)
            
            # Position at top-right corner
            popup.geometry("+{}+{}".format(
                self.main_window.root.winfo_x() + self.main_window.root.winfo_width() - 420,
                self.main_window.root.winfo_y() + 50
            ))
            
            # Main frame
            main_frame = ttk.Frame(popup, padding="15")
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Alert icon and title
            title_frame = ttk.Frame(main_frame)
            title_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(title_frame, text="⚡", font=("Arial", 24)).pack(side=tk.LEFT)
            ttk.Label(title_frame, text="Pattern Detected!", 
                     font=("Arial", 14, "bold")).pack(side=tk.LEFT, padx=(10, 0))
            
            # Pattern details
            details_frame = ttk.LabelFrame(main_frame, text="Alert Details", padding="10")
            details_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(details_frame, text=f"Pattern: {alert.pattern_type.replace('_', ' ').title()}",
                     font=("Arial", 10, "bold")).pack(anchor=tk.W)
            ttk.Label(details_frame, text=f"Symbol: {alert.symbol}").pack(anchor=tk.W)
            ttk.Label(details_frame, text=f"Confidence: {alert.confidence:.1f}%").pack(anchor=tk.W)
            ttk.Label(details_frame, text=f"Suggestion: {alert.suggested_action}",
                     font=("Arial", 10, "italic")).pack(anchor=tk.W)
            
            # Action buttons
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill=tk.X)
            
            ttk.Button(button_frame, text="View Pattern", 
                      command=lambda: self.view_pattern_details(alert, popup)).pack(side=tk.LEFT, padx=(0, 10))
            ttk.Button(button_frame, text="Dismiss", 
                      command=popup.destroy).pack(side=tk.LEFT)
            
            # Auto-dismiss after 10 seconds
            popup.after(10000, popup.destroy)
            
        # Run in main thread
        self.main_window.root.after(0, show_popup)
        
    def view_pattern_details(self, alert: PatternAlert, popup: tk.Toplevel):
        """Show detailed pattern analysis"""
        popup.destroy()
        
        # Create detailed analysis window
        detail_window = tk.Toplevel(self.main_window.root)
        detail_window.title(f"Pattern Analysis: {alert.pattern_type.replace('_', ' ').title()}")
        detail_window.geometry("600x500")
        
        # Main frame with scrolling
        main_frame = ttk.Frame(detail_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Pattern summary
        summary_frame = ttk.LabelFrame(main_frame, text="Pattern Summary", padding="15")
        summary_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(summary_frame, text=f"Pattern Type: {alert.pattern_type.replace('_', ' ').title()}",
                 font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=2)
        ttk.Label(summary_frame, text=f"Symbol: {alert.symbol}").pack(anchor=tk.W, pady=2)
        ttk.Label(summary_frame, text=f"Detection Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}").pack(anchor=tk.W, pady=2)
        ttk.Label(summary_frame, text=f"Confidence Score: {alert.confidence:.1f}%").pack(anchor=tk.W, pady=2)
        
        # Analysis details
        analysis_frame = ttk.LabelFrame(main_frame, text="AI Analysis", padding="15")
        analysis_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        analysis_text = tk.Text(analysis_frame, wrap=tk.WORD, height=10)
        analysis_text.pack(fill=tk.BOTH, expand=True)
        
        # Generate analysis text based on pattern type
        analysis_content = self.generate_pattern_analysis(alert)
        analysis_text.insert(tk.END, analysis_content)
        analysis_text.config(state=tk.DISABLED)
        
        # Action buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X)
        
        ttk.Button(action_frame, text="Close", command=detail_window.destroy).pack(side=tk.RIGHT)
        
    def generate_pattern_analysis(self, alert: PatternAlert) -> str:
        """Generate detailed pattern analysis text"""
        pattern_info = {
            'head_shoulders': {
                'description': 'A reversal pattern that signals a potential trend change from bullish to bearish.',
                'reliability': 'High (75-85% success rate)',
                'action': 'Consider selling or shorting after neckline break with volume confirmation.'
            },
            'double_top': {
                'description': 'A bearish reversal pattern showing two peaks at similar price levels.',
                'reliability': 'Moderate to High (70-80% success rate)',
                'action': 'Look for a break below support level for short entry confirmation.'
            },
            'double_bottom': {
                'description': 'A bullish reversal pattern showing two troughs at similar price levels.',
                'reliability': 'Moderate to High (70-80% success rate)',
                'action': 'Look for a break above resistance level for long entry confirmation.'
            },
            'triangle': {
                'description': 'A continuation pattern that typically resolves in the direction of the previous trend.',
                'reliability': 'Moderate (60-70% success rate)',
                'action': 'Wait for breakout with increased volume to confirm direction.'
            }
        }
        
        pattern_key = alert.pattern_type
        info = pattern_info.get(pattern_key, {
            'description': 'Technical pattern detected by AI analysis.',
            'reliability': 'Variable',
            'action': 'Monitor for confirmation signals.'
        })
        
        return f"""Pattern Description:
{info['description']}

Historical Reliability:
{info['reliability']}

Recommended Action:
{info['action']}

AI Confidence Explanation:
The {alert.confidence:.1f}% confidence score is based on:
• Pattern shape matching (geometric accuracy)
• Volume confirmation
• Market context analysis
• Historical pattern performance

Risk Considerations:
• Always wait for confirmation signals
• Consider overall market conditions
• Use proper position sizing
• Set stop-loss levels appropriately

This analysis is for educational purposes and should not be considered as financial advice."""
        
    def play_alert_sound(self):
        """Play alert notification sound"""
        try:
            # Simple beep for now - can be enhanced with custom sounds
            print('\a')  # System beep
        except:
            pass
            
    def get_alerts_history(self, limit: int = 50) -> List[PatternAlert]:
        """Get recent alerts history"""
        return self.alerts_history[-limit:]
        
    def save_alert_history(self):
        """Save alerts to file"""
        try:
            alerts_data = []
            for alert in self.alerts_history[-100:]:  # Keep last 100 alerts
                alerts_data.append({
                    'pattern_type': alert.pattern_type,
                    'symbol': alert.symbol,
                    'confidence': alert.confidence,
                    'timestamp': alert.timestamp.isoformat(),
                    'suggested_action': alert.suggested_action,
                    'pattern_data': alert.pattern_data
                })
            
            os.makedirs('data', exist_ok=True)
            with open('data/alerts_history.json', 'w') as f:
                json.dump(alerts_data, f, indent=2)
        except Exception as e:
            print(f"Error saving alert history: {e}")
            
    def load_alert_history(self):
        """Load alerts from file"""
        try:
            if os.path.exists('data/alerts_history.json'):
                with open('data/alerts_history.json', 'r') as f:
                    alerts_data = json.load(f)
                
                self.alerts_history = []
                for data in alerts_data:
                    alert = PatternAlert(
                        pattern_type=data['pattern_type'],
                        symbol=data['symbol'],
                        confidence=data['confidence'],
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        suggested_action=data['suggested_action'],
                        pattern_data=data['pattern_data']
                    )
                    self.alerts_history.append(alert)
        except Exception as e:
            print(f"Error loading alert history: {e}")
            
    def save_settings(self):
        """Save notification settings"""
        try:
            os.makedirs('data', exist_ok=True)
            with open('data/alert_settings.json', 'w') as f:
                json.dump(self.notification_settings, f, indent=2)
        except Exception as e:
            print(f"Error saving alert settings: {e}")
            
    def load_settings(self):
        """Load notification settings"""
        try:
            if os.path.exists('data/alert_settings.json'):
                with open('data/alert_settings.json', 'r') as f:
                    saved_settings = json.load(f)
                    self.notification_settings.update(saved_settings)
        except Exception as e:
            print(f"Error loading alert settings: {e}")
            
    def update_settings(self, new_settings: Dict):
        """Update notification settings"""
        self.notification_settings.update(new_settings)
        self.save_settings()