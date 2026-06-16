from matplotlib.path import Path
from matplotlib.markers import MarkerStyle

def calcium_trace_symbol():
    """Create a custom calcium-like peak marker."""
    verts = [(-.3, -.2), (-.2, .4), (-.1, .2), (0.2, -.1), (0.3, -.2)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3]
    peak_path = Path(verts, codes)
    return MarkerStyle(peak_path)

def create_peak_markers(ax, left, peak, right, peak_data, marker_style):
    left_marker = ax.plot(left, peak_data["height"], marker=marker_style['left'], color='blue', picker=True)[0]
    peak_marker = ax.plot(peak, peak_data["height"], marker=marker_style['peak'], color='red', picker=True)[0]
    right_marker = ax.plot(right, peak_data["height"], marker=marker_style['right'], color='blue', picker=True)[0]

    # Make draggable
    from draggable_point import DraggablePoint
    left_drag = DraggablePoint(left_marker)
    peak_drag = DraggablePoint(peak_marker)
    right_drag = DraggablePoint(right_marker)

    return (left_drag, peak_drag, right_drag)
