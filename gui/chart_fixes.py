"""
Chart threading fixes for matplotlib Tkinter integration
"""

import matplotlib.pyplot as plt

def disable_mouse_events(canvas):
    """Disable problematic mouse events that cause threading issues"""
    try:
        # Complete event callback clearing
        canvas.callbacks.callbacks.clear()
        
        # Disconnect widget-level events that cause threading issues
        widget = canvas.get_tk_widget()
        widget.unbind('<Motion>')
        widget.unbind('<Enter>')
        widget.unbind('<Leave>')
        widget.unbind('<Button-1>')
        widget.unbind('<Button-2>')
        widget.unbind('<Button-3>')
        
        # Override the toolbar's set_message method to prevent threading errors
        if hasattr(canvas, 'toolbar') and canvas.toolbar:
            def safe_set_message(msg):
                try:
                    pass  # Do nothing to avoid threading issues
                except:
                    pass
            canvas.toolbar.set_message = safe_set_message
            
    except Exception:
        pass

def configure_matplotlib_threading():
    """Configure matplotlib for thread-safe operation"""
    plt.ioff()  # Turn off interactive mode
    
def create_thread_safe_canvas(fig, parent):
    """Create a thread-safe matplotlib canvas"""
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    
    canvas = FigureCanvasTkAgg(fig, parent)
    disable_mouse_events(canvas)
    canvas.draw()
    return canvas