"""
HUD (Heads-Up Display)
======================
2D overlay rendering for text, trends, and debug info.
Extracted from lightController_osc.py.
"""

from datetime import datetime
from typing import Optional, Dict, Any, Tuple

try:
    import pygame
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False


def draw_text_2d(x: int, y: int, text: str, font, color: Tuple[int, int, int] = (255, 255, 255)):
    """
    Draw text on screen (2D HUD).
    
    Args:
        x, y: Screen position
        text: Text to draw
        font: pygame font object
        color: RGB tuple
    """
    if not OPENGL_AVAILABLE:
        return
        
    text_surface = font.render(text, True, color)
    text_data = pygame.image.tostring(text_surface, "RGBA", True)
    glWindowPos2d(x, y)
    glDrawPixels(text_surface.get_width(), text_surface.get_height(),
                 GL_RGBA, GL_UNSIGNED_BYTE, text_data)


def draw_text_3d_billboard(position: Tuple[float, float, float], text: str, font, 
                           color: Tuple[int, int, int] = (255, 255, 255), offset_y: float = 0):
    """
    Draw text in 3D space as a billboard (always faces camera).
    
    Args:
        position: (x, y, z) world position
        text: Text to draw
        font: pygame font object
        color: RGB tuple
        offset_y: Vertical offset from position
    """
    if not OPENGL_AVAILABLE:
        return
        
    # Get current matrices
    modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection = glGetDoublev(GL_PROJECTION_MATRIX)
    viewport = glGetIntegerv(GL_VIEWPORT)
    
    # Project 3D position to screen coordinates
    try:
        screen_x, screen_y, screen_z = gluProject(
            position[0], position[1] + offset_y, position[2],
            modelview, projection, viewport
        )
        
        # Only draw if in front of camera
        if screen_z < 1.0:
            # Render text
            text_surface = font.render(text, True, color)
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            
            # Center text horizontally
            text_x = int(screen_x - text_surface.get_width() / 2)
            text_y = int(screen_y)
            
            glWindowPos2d(text_x, text_y)
            glDrawPixels(text_surface.get_width(), text_surface.get_height(),
                        GL_RGBA, GL_UNSIGNED_BYTE, text_data)
    except:
        pass  # Projection failed, skip


def draw_trends_visualization(report, x: int, y: int, width: int, height: int, 
                               font, font_small):
    """
    Draw a visualization of daily trends as a bar chart overlay.
    
    Args:
        report: The DailyReport object to visualize
        x, y: Bottom-left corner position
        width, height: Size of the visualization area
        font, font_small: Fonts for labels
    """
    if not OPENGL_AVAILABLE or not report:
        return
        
    if not hasattr(report, 'hourly_trends') or not report.hourly_trends:
        return
    
    # Background panel with transparency
    glColor4f(0.1, 0.1, 0.15, 0.9)
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + width, y)
    glVertex2f(x + width, y + height)
    glVertex2f(x, y + height)
    glEnd()
    
    # Border
    glColor4f(0.3, 0.5, 0.7, 1.0)
    glLineWidth(2)
    glBegin(GL_LINE_LOOP)
    glVertex2f(x, y)
    glVertex2f(x + width, y)
    glVertex2f(x + width, y + height)
    glVertex2f(x, y + height)
    glEnd()
    
    # Title
    title = f"Daily Report: {report.date}"
    draw_text_2d(x + 10, y + height - 25, title, font, (255, 255, 200))
    
    # Summary line
    summary = f"Total: {report.total_unique_people} people | Peak: {report.peak_hour}:00 ({report.peak_hour_count}) | Flow: {report.dominant_flow}"
    draw_text_2d(x + 10, y + height - 45, summary, font_small, (200, 200, 200))
    
    # Chart area
    chart_x = x + 50
    chart_y = y + 30
    chart_width = width - 70
    chart_height = height - 100
    
    # Find max values for scaling
    max_active = max((h.active_count for h in report.hourly_trends), default=1) or 1
    max_passive = max((h.passive_count for h in report.hourly_trends), default=1) or 1
    max_combined = max(max_active + max_passive // 3, 10)
    
    # Draw hour bars
    bar_width = chart_width / 24
    bar_gap = 2
    
    for trend in report.hourly_trends:
        bar_x = chart_x + trend.hour * bar_width + bar_gap / 2
        bar_w = bar_width - bar_gap
        
        # Active bar (green)
        active_height = (trend.active_count / max_combined) * chart_height
        glColor4f(0.2, 0.7, 0.3, 0.8)
        glBegin(GL_QUADS)
        glVertex2f(bar_x, chart_y)
        glVertex2f(bar_x + bar_w, chart_y)
        glVertex2f(bar_x + bar_w, chart_y + active_height)
        glVertex2f(bar_x, chart_y + active_height)
        glEnd()
        
        # Passive bar (blue, stacked)
        passive_height = (trend.passive_count / 3 / max_combined) * chart_height
        glColor4f(0.3, 0.3, 0.7, 0.8)
        glBegin(GL_QUADS)
        glVertex2f(bar_x, chart_y + active_height)
        glVertex2f(bar_x + bar_w, chart_y + active_height)
        glVertex2f(bar_x + bar_w, chart_y + active_height + passive_height)
        glVertex2f(bar_x, chart_y + active_height + passive_height)
        glEnd()
    
    # X-axis labels (hours)
    for hour in range(0, 24, 3):
        label_x = chart_x + hour * bar_width
        draw_text_2d(int(label_x), chart_y - 15, f"{hour:02d}", font_small, (150, 150, 150))
    
    # Y-axis label
    draw_text_2d(x + 5, chart_y + chart_height // 2, "Pop", font_small, (150, 150, 150))
    
    # Legend
    legend_y = y + height - 65
    glColor4f(0.2, 0.7, 0.3, 0.8)
    glBegin(GL_QUADS)
    glVertex2f(x + 10, legend_y)
    glVertex2f(x + 25, legend_y)
    glVertex2f(x + 25, legend_y + 10)
    glVertex2f(x + 10, legend_y + 10)
    glEnd()
    draw_text_2d(x + 30, legend_y - 2, "Active", font_small, (100, 200, 100))
    
    glColor4f(0.3, 0.3, 0.7, 0.8)
    glBegin(GL_QUADS)
    glVertex2f(x + 90, legend_y)
    glVertex2f(x + 105, legend_y)
    glVertex2f(x + 105, legend_y + 10)
    glVertex2f(x + 90, legend_y + 10)
    glEnd()
    draw_text_2d(x + 110, legend_y - 2, "Passive", font_small, (100, 100, 200))
    
    # Close hint
    draw_text_2d(x + width - 80, y + 10, "T to close", font_small, (120, 120, 120))


def draw_realtime_trends(idle_trends: dict, x: int, y: int, font, font_small, 
                         aggression: dict = None, flow: dict = None, 
                         almost_engaged: dict = None, feedback_learning: dict = None):
    """
    Draw real-time trends panel on the left side of the screen.
    
    Args:
        idle_trends: Dict from behavior_status.get('idle_trends')
        x, y: Top-left position
        font, font_small: Fonts for rendering
        aggression: Dict from behavior_status.get('aggression')
        flow: Dict from behavior_status.get('flow')
        almost_engaged: Dict from behavior_status.get('almost_engaged')
        feedback_learning: Dict from behavior_status.get('feedback_learning')
    """
    if not OPENGL_AVAILABLE or not idle_trends:
        return
    
    panel_width = 260
    panel_height = 520
    
    # Background panel
    glColor4f(0.08, 0.08, 0.12, 0.85)
    glBegin(GL_QUADS)
    glVertex2f(x, y - panel_height)
    glVertex2f(x + panel_width, y - panel_height)
    glVertex2f(x + panel_width, y)
    glVertex2f(x, y)
    glEnd()
    
    # Border
    glColor4f(0.3, 0.4, 0.6, 0.8)
    glLineWidth(1)
    glBegin(GL_LINE_LOOP)
    glVertex2f(x, y - panel_height)
    glVertex2f(x + panel_width, y - panel_height)
    glVertex2f(x + panel_width, y)
    glVertex2f(x, y)
    glEnd()
    
    # Title
    draw_text_2d(x + 10, y - 18, "REALTIME TRENDS", font, (100, 180, 255))
    
    # Update timing
    seconds_since = idle_trends.get('seconds_since_update', 0)
    update_color = (100, 255, 100) if seconds_since < 6 else (255, 200, 100) if seconds_since < 15 else (255, 100, 100)
    draw_text_2d(x + 130, y - 18, f"({seconds_since:.1f}s ago)", font_small, update_color)
    
    curr_y = y - 35
    line_height = 14
    min_y = y - panel_height + 15
    
    # Period indicator
    period = idle_trends.get('period', 'unknown')
    period_colors = {
        'late_night': (100, 100, 180),
        'morning': (255, 200, 100),
        'afternoon': (255, 255, 150),
        'evening': (180, 130, 200),
    }
    period_color = period_colors.get(period, (150, 150, 150))
    draw_text_2d(x + 10, curr_y, f"Period: {period.upper()}", font_small, period_color)
    curr_y -= line_height + 3
    
    # Database error if any
    db_error = idle_trends.get('database_error', '')
    if db_error:
        draw_text_2d(x + 10, curr_y, f"⚠ {db_error[:25]}", font_small, (255, 100, 100))
        curr_y -= line_height
    
    # Section: REALTIME (1 min)
    if curr_y < min_y: return
    has_recent = idle_trends.get('has_recent', False)
    status_char = "●" if has_recent else "○"
    status_color = (100, 255, 100) if has_recent else (100, 100, 100)
    draw_text_2d(x + 10, curr_y, f"{status_char} Now (1m)", font_small, status_color)
    recent_passive = idle_trends.get('recent_passive', 0)
    recent_active = idle_trends.get('recent_active', 0)
    draw_text_2d(x + 95, curr_y, f"P:{recent_passive}", font_small, (180, 180, 255))
    draw_text_2d(x + 140, curr_y, f"A:{recent_active}", font_small, (255, 180, 100))
    curr_y -= line_height
    
    # Section: SHORT TERM (5 min)
    has_short = idle_trends.get('has_short', False)
    status_char = "●" if has_short else "○"
    status_color = (100, 255, 100) if has_short else (100, 100, 100)
    draw_text_2d(x + 10, curr_y, f"{status_char} Short (5m)", font_small, status_color)
    short_passive = idle_trends.get('short_passive', 0)
    short_active = idle_trends.get('short_active', 0)
    draw_text_2d(x + 95, curr_y, f"P:{short_passive}", font_small, (180, 180, 255))
    draw_text_2d(x + 140, curr_y, f"A:{short_active}", font_small, (255, 180, 100))
    short_act = idle_trends.get('short_activity', 0)
    bar = "█" * int(short_act * 6) + "░" * (6 - int(short_act * 6))
    draw_text_2d(x + 180, curr_y, f"[{bar}]", font_small, (100, 200, 100))
    curr_y -= line_height
    
    # Section: MEDIUM TERM (30 min)
    has_medium = idle_trends.get('has_medium', False)
    status_char = "●" if has_medium else "○"
    status_color = (100, 255, 100) if has_medium else (100, 100, 100)
    draw_text_2d(x + 10, curr_y, f"{status_char} Med (30m)", font_small, status_color)
    med_passive = idle_trends.get('medium_passive', 0)
    med_active = idle_trends.get('medium_active', 0)
    draw_text_2d(x + 95, curr_y, f"P:{med_passive}", font_small, (180, 180, 255))
    draw_text_2d(x + 140, curr_y, f"A:{med_active}", font_small, (255, 180, 100))
    med_act = idle_trends.get('medium_activity', 0)
    bar = "█" * int(med_act * 6) + "░" * (6 - int(med_act * 6))
    draw_text_2d(x + 180, curr_y, f"[{bar}]", font_small, (100, 150, 200))
    curr_y -= line_height
    
    # Section: LONG TERM (1 hr)
    has_long = idle_trends.get('has_long', False)
    status_char = "●" if has_long else "○"
    status_color = (100, 255, 100) if has_long else (100, 100, 100)
    draw_text_2d(x + 10, curr_y, f"{status_char} Long (1h)", font_small, status_color)
    long_passive = idle_trends.get('long_passive', 0)
    long_active = idle_trends.get('long_active', 0)
    draw_text_2d(x + 95, curr_y, f"P:{long_passive}", font_small, (180, 180, 255))
    draw_text_2d(x + 140, curr_y, f"A:{long_active}", font_small, (255, 180, 100))
    long_act = idle_trends.get('long_activity', 0)
    bar = "█" * int(long_act * 6) + "░" * (6 - int(long_act * 6))
    draw_text_2d(x + 180, curr_y, f"[{bar}]", font_small, (150, 150, 255))
    curr_y -= line_height
    
    # Section: HISTORICAL
    has_hist = idle_trends.get('has_historical', False)
    status_char = "●" if has_hist else "○"
    status_color = (100, 255, 100) if has_hist else (100, 100, 100)
    draw_text_2d(x + 10, curr_y, f"{status_char} Historical (7d)", font_small, status_color)
    curr_y -= line_height + 6
    
    # Divider line
    if curr_y < min_y: return
    glColor4f(0.3, 0.4, 0.6, 0.5)
    glBegin(GL_LINES)
    glVertex2f(x + 10, curr_y + 3)
    glVertex2f(x + panel_width - 10, curr_y + 3)
    glEnd()
    
    # COMPUTED VALUES section
    draw_text_2d(x + 10, curr_y, "COMPUTED", font_small, (180, 180, 200))
    curr_y -= line_height
    
    # Anticipation
    anticipation = idle_trends.get('activity_anticipation', 0.5)
    ant_bar = "█" * int(anticipation * 10) + "░" * (10 - int(anticipation * 10))
    ant_color = (100, 255, 100) if anticipation > 0.6 else (255, 200, 100) if anticipation > 0.3 else (100, 100, 100)
    draw_text_2d(x + 10, curr_y, "Anticipation:", font_small, (180, 180, 180))
    draw_text_2d(x + 95, curr_y, f"[{ant_bar}]", font_small, ant_color)
    curr_y -= line_height
    
    # Flow momentum
    momentum = idle_trends.get('flow_momentum', 0)
    if abs(momentum) > 0.1:
        arrow_count = int(abs(momentum) * 5)
        arrows = "→" * arrow_count if momentum > 0 else "←" * arrow_count
        mom_color = (100, 200, 255) if momentum > 0 else (255, 200, 100)
        draw_text_2d(x + 10, curr_y, "Flow:", font_small, (180, 180, 180))
        draw_text_2d(x + 55, curr_y, f"{arrows} ({momentum:+.2f})", font_small, mom_color)
    else:
        draw_text_2d(x + 10, curr_y, "Flow: balanced", font_small, (100, 100, 100))
    curr_y -= line_height
    
    # Energy level
    energy = idle_trends.get('energy_level', 0.5)
    energy_bar = "█" * int(energy * 10) + "░" * (10 - int(energy * 10))
    energy_color = (255, 200, 100) if energy > 0.6 else (150, 200, 150) if energy > 0.3 else (100, 100, 150)
    draw_text_2d(x + 10, curr_y, "Energy:", font_small, (180, 180, 180))
    draw_text_2d(x + 65, curr_y, f"[{energy_bar}]", font_small, energy_color)
    curr_y -= line_height + 6
    
    # AGGRESSION SECTION
    if aggression and curr_y > min_y:
        _draw_aggression_section(x, curr_y, panel_width, min_y, line_height, 
                                 font_small, aggression)
        curr_y -= line_height * 4 + 6
    
    # FLOW SECTION
    if flow and curr_y > min_y:
        _draw_flow_section(x, curr_y, panel_width, min_y, line_height,
                           font_small, flow)
        curr_y -= line_height * 5 + 6
    
    # ALMOST-ENGAGED SECTION
    if almost_engaged and curr_y > min_y:
        _draw_almost_engaged_section(x, curr_y, panel_width, min_y, line_height,
                                     font_small, almost_engaged)


def _draw_aggression_section(x, curr_y, panel_width, min_y, line_height, font_small, aggression):
    """Draw aggression sub-section of trends panel."""
    # Divider line
    glColor4f(0.3, 0.4, 0.6, 0.5)
    glBegin(GL_LINES)
    glVertex2f(x + 10, curr_y + 3)
    glVertex2f(x + panel_width - 10, curr_y + 3)
    glEnd()
    
    draw_text_2d(x + 10, curr_y, "AGGRESSION", font_small, (255, 150, 100))
    curr_y -= line_height
    
    level = aggression.get('level', 0)
    cap = aggression.get('time_of_day_cap', 1.0)
    bar_filled = int(level * 10)
    bar_cap = int(cap * 10)
    
    bar = ""
    for i in range(10):
        if i < bar_filled:
            bar += "█"
        elif i < bar_cap:
            bar += "▒"
        else:
            bar += "░"
    
    if level < 0.3:
        agg_color = (100, 200, 100)
    elif level < 0.6:
        agg_color = (255, 200, 100)
    else:
        agg_color = (255, 100, 100)
    
    draw_text_2d(x + 10, curr_y, "Level:", font_small, (180, 180, 180))
    draw_text_2d(x + 55, curr_y, f"[{bar}]", font_small, agg_color)
    draw_text_2d(x + 175, curr_y, f"{level:.2f}", font_small, agg_color)
    curr_y -= line_height
    
    hour = datetime.now().hour
    draw_text_2d(x + 10, curr_y, f"ToD Cap ({hour:02d}:00):", font_small, (150, 150, 150))
    draw_text_2d(x + 115, curr_y, f"{cap:.1f}", font_small, (180, 180, 200))
    curr_y -= line_height
    
    since_eng = aggression.get('seconds_since_engagement', 0)
    if since_eng < 60:
        time_str = f"{since_eng:.0f}s"
    else:
        time_str = f"{since_eng/60:.1f}m"
    eng_color = (100, 255, 100) if since_eng < 30 else (255, 200, 100) if since_eng < 300 else (255, 100, 100)
    draw_text_2d(x + 10, curr_y, "Since engage:", font_small, (150, 150, 150))
    draw_text_2d(x + 100, curr_y, time_str, font_small, eng_color)
    
    if aggression.get('current_engagement'):
        draw_text_2d(x + 160, curr_y, "ENGAGED", font_small, (100, 255, 100))


def _draw_flow_section(x, curr_y, panel_width, min_y, line_height, font_small, flow):
    """Draw flow positioning sub-section of trends panel."""
    glColor4f(0.3, 0.4, 0.6, 0.5)
    glBegin(GL_LINES)
    glVertex2f(x + 10, curr_y + 3)
    glVertex2f(x + panel_width - 10, curr_y + 3)
    glEnd()
    
    draw_text_2d(x + 10, curr_y, "FLOW", font_small, (100, 200, 255))
    curr_y -= line_height
    
    direction = flow.get('direction', 0)
    strength = flow.get('strength', 0)
    x_offset = flow.get('x_offset', 0)
    
    if strength > 0.2 and abs(direction) > 0.1:
        arrow_count = min(5, max(1, int(strength * 5)))
        if direction > 0:
            arrows = "→" * arrow_count
            flow_label = "L→R"
            flow_color = (100, 200, 255)
        else:
            arrows = "←" * arrow_count
            flow_label = "R→L"
            flow_color = (255, 180, 100)
        draw_text_2d(x + 10, curr_y, f"Flow: {flow_label}", font_small, (180, 180, 180))
        draw_text_2d(x + 80, curr_y, arrows, font_small, flow_color)
        draw_text_2d(x + 150, curr_y, f"({direction:+.2f})", font_small, flow_color)
    else:
        draw_text_2d(x + 10, curr_y, "Flow: none/mixed", font_small, (100, 100, 100))
    curr_y -= line_height
    
    strength_bar = "█" * int(strength * 6) + "░" * (6 - int(strength * 6))
    strength_color = (100, 255, 100) if strength > 0.5 else (200, 200, 100) if strength > 0.2 else (100, 100, 100)
    draw_text_2d(x + 10, curr_y, "Strength:", font_small, (150, 150, 150))
    draw_text_2d(x + 75, curr_y, f"[{strength_bar}]", font_small, strength_color)
    curr_y -= line_height
    
    if abs(x_offset) > 1:
        offset_dir = "←" if x_offset < 0 else "→"
        offset_color = (100, 255, 200)
        draw_text_2d(x + 10, curr_y, "Box offset:", font_small, (150, 150, 150))
        draw_text_2d(x + 85, curr_y, f"{offset_dir} {abs(x_offset):.0f}cm", font_small, offset_color)
    else:
        draw_text_2d(x + 10, curr_y, "Box offset: centered", font_small, (100, 100, 100))
    curr_y -= line_height
    
    ltr = flow.get('left_to_right', 0)
    rtl = flow.get('right_to_left', 0)
    total = flow.get('total_events', 0)
    draw_text_2d(x + 10, curr_y, f"30s: L→R:{ltr} R→L:{rtl} ({total})", font_small, (120, 120, 150))


def _draw_almost_engaged_section(x, curr_y, panel_width, min_y, line_height, font_small, almost_engaged):
    """Draw almost-engaged sub-section of trends panel."""
    glColor4f(0.3, 0.4, 0.6, 0.5)
    glBegin(GL_LINES)
    glVertex2f(x + 10, curr_y + 3)
    glVertex2f(x + panel_width - 10, curr_y + 3)
    glEnd()
    
    draw_text_2d(x + 10, curr_y, "ALMOST-ENGAGED", font_small, (255, 200, 100))
    curr_y -= line_height
    
    total_det = almost_engaged.get('total_detected', 0)
    total_conv = almost_engaged.get('total_converted', 0)
    conv_rate = almost_engaged.get('conversion_rate', 0) * 100
    
    rate_color = (100, 255, 100) if conv_rate > 30 else (255, 200, 100) if conv_rate > 10 else (150, 150, 150)
    draw_text_2d(x + 10, curr_y, f"Detected: {total_det}", font_small, (180, 180, 180))
    draw_text_2d(x + 100, curr_y, f"Conv: {total_conv}", font_small, (180, 180, 180))
    draw_text_2d(x + 170, curr_y, f"({conv_rate:.0f}%)", font_small, rate_color)
    curr_y -= line_height
    
    if almost_engaged.get('active_attraction'):
        strategy = almost_engaged.get('current_strategy', 'none')
        target_id = almost_engaged.get('target_id', -1)
        draw_text_2d(x + 10, curr_y, f"→ Attracting #{target_id}", font_small, (100, 255, 200))
        draw_text_2d(x + 130, curr_y, f"[{strategy}]", font_small, (255, 200, 100))
    else:
        cand_count = almost_engaged.get('candidate_count', 0)
        if cand_count > 0:
            draw_text_2d(x + 10, curr_y, f"Watching {cand_count} candidate(s)", font_small, (200, 200, 150))
        else:
            draw_text_2d(x + 10, curr_y, "No candidates", font_small, (100, 100, 100))
