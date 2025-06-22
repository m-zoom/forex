"""
Chart threading fixes for matplotlib Tkinter integration
"""

import matplotlib.pyplot as plt

def disable_mouse_events(canvas):
    """Disable problematic mouse events that cause threading issues"""
    try:
        # Disconnect mouse events to prevent threading conflicts
        for connection_id in list(canvas.callbacks.callbacks.get('motion_notify_event', {})):
            canvas.mpl_disconnect(connection_id)
        for connection_id in list(canvas.callbacks.callbacks.get('button_press_event', {})):
            canvas.mpl_disconnect(connection_id)
        for connection_id in list(canvas.callbacks.callbacks.get('button_release_event', {})):
            canvas.mpl_disconnect(connection_id)
    except:
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