"""
OpenGL Primitives
=================
Basic drawing functions for 3D scene elements.
Extracted from lightController_osc.py.
"""

import math
from typing import Tuple, Optional, Callable, Any

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False


def draw_box_wireframe(bounds: Tuple[float, ...], color: Tuple[float, ...]):
    """
    Draw wireframe box from bounds.
    
    Args:
        bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
        color: RGBA tuple (r, g, b, a)
    """
    if not OPENGL_AVAILABLE:
        return
        
    x0, x1, y0, y1, z0, z1 = bounds
    
    glColor4f(*color)
    glBegin(GL_LINES)
    
    # Bottom face
    glVertex3f(x0, y0, z0); glVertex3f(x1, y0, z0)
    glVertex3f(x1, y0, z0); glVertex3f(x1, y0, z1)
    glVertex3f(x1, y0, z1); glVertex3f(x0, y0, z1)
    glVertex3f(x0, y0, z1); glVertex3f(x0, y0, z0)
    
    # Top face
    glVertex3f(x0, y1, z0); glVertex3f(x1, y1, z0)
    glVertex3f(x1, y1, z0); glVertex3f(x1, y1, z1)
    glVertex3f(x1, y1, z1); glVertex3f(x0, y1, z1)
    glVertex3f(x0, y1, z1); glVertex3f(x0, y1, z0)
    
    # Vertical edges
    glVertex3f(x0, y0, z0); glVertex3f(x0, y1, z0)
    glVertex3f(x1, y0, z0); glVertex3f(x1, y1, z0)
    glVertex3f(x1, y0, z1); glVertex3f(x1, y1, z1)
    glVertex3f(x0, y0, z1); glVertex3f(x0, y1, z1)
    
    glEnd()


def draw_panel(center: Tuple[float, float, float], angle: float, size: float, brightness: float):
    """
    Draw a panel as a quad.
    
    Args:
        center: (x, y, z) position
        angle: Rotation angle in degrees
        size: Panel size in cm
        brightness: 0.0 to 1.0
    """
    if not OPENGL_AVAILABLE:
        return
        
    half = size / 2
    
    glPushMatrix()
    glTranslatef(*center)
    glRotatef(-angle, 1, 0, 0)
    
    gray = 0.2 + brightness * 0.8
    glColor4f(gray, gray, gray, 1.0)
    
    glBegin(GL_QUADS)
    glVertex3f(-half, -half, 0)
    glVertex3f(half, -half, 0)
    glVertex3f(half, half, 0)
    glVertex3f(-half, half, 0)
    glEnd()
    
    glColor4f(0.3, 0.3, 0.3, 1.0)
    glLineWidth(2)
    glBegin(GL_LINE_LOOP)
    glVertex3f(-half, -half, 0)
    glVertex3f(half, -half, 0)
    glVertex3f(half, half, 0)
    glVertex3f(-half, half, 0)
    glEnd()
    
    glPopMatrix()


def draw_sphere(center: Tuple[float, float, float], radius: float, 
                color: Tuple[float, ...], segments: int = 12):
    """
    Draw a solid sphere.
    
    Args:
        center: (x, y, z) position
        radius: Sphere radius
        color: RGBA tuple
        segments: Number of segments for smoothness
    """
    if not OPENGL_AVAILABLE:
        return
        
    glPushMatrix()
    glTranslatef(*center)
    glColor4f(*color)
    
    quadric = gluNewQuadric()
    gluSphere(quadric, radius, segments, segments)
    gluDeleteQuadric(quadric)
    
    glPopMatrix()


def draw_sphere_wireframe(center: Tuple[float, float, float], radius: float,
                          color: Tuple[float, ...], segments: int = 16):
    """
    Draw a wireframe sphere.
    
    Args:
        center: (x, y, z) position
        radius: Sphere radius
        color: RGBA tuple
        segments: Number of segments
    """
    if not OPENGL_AVAILABLE:
        return
        
    glPushMatrix()
    glTranslatef(*center)
    glColor4f(*color)
    glLineWidth(1)
    
    # Latitude lines
    for i in range(segments // 2 + 1):
        lat = math.pi * i / (segments // 2) - math.pi / 2
        r = radius * math.cos(lat)
        y = radius * math.sin(lat)
        
        glBegin(GL_LINE_LOOP)
        for j in range(segments):
            lon = 2 * math.pi * j / segments
            x = r * math.cos(lon)
            z = r * math.sin(lon)
            glVertex3f(x, y, z)
        glEnd()
    
    # Longitude lines
    for j in range(segments // 2):
        lon = math.pi * j / (segments // 2)
        
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            lat = 2 * math.pi * i / segments
            x = radius * math.cos(lat) * math.sin(lon)
            y = radius * math.sin(lat)
            z = radius * math.cos(lat) * math.cos(lon)
            glVertex3f(x, y, z)
        glEnd()
    
    glPopMatrix()


def draw_tracked_person(position: Tuple[float, float, float], zone: str = "unknown"):
    """
    Draw a tracked person as a cylinder/capsule.
    
    Args:
        position: (x, y, z) world position
        zone: 'active', 'passive', or 'unknown' for coloring
    """
    if not OPENGL_AVAILABLE:
        return
    
    # Color based on zone
    if zone == "active":
        color = (0.2, 0.8, 0.2, 0.8)  # Green for active
    elif zone == "passive":
        color = (0.8, 0.8, 0.2, 0.8)  # Yellow for passive
    else:
        color = (0.5, 0.5, 0.5, 0.6)  # Gray for unknown
    
    # Draw as a colored cylinder (person height ~170cm)
    height = 170
    radius = 20
    
    glPushMatrix()
    glTranslatef(position[0], position[1], position[2])
    
    # Body cylinder
    glColor4f(*color)
    
    quadric = gluNewQuadric()
    glRotatef(-90, 1, 0, 0)  # Rotate to stand upright
    gluCylinder(quadric, radius, radius, height, 16, 1)
    
    # Top cap (head)
    glTranslatef(0, 0, height)
    gluSphere(quadric, radius, 12, 12)
    
    gluDeleteQuadric(quadric)
    glPopMatrix()


def draw_floor(y_level: float, color: Tuple[float, ...], z_max: Optional[float] = None):
    """
    Draw a floor plane.
    
    Args:
        y_level: Y coordinate of floor
        color: RGBA tuple
        z_max: Maximum Z depth (defaults to 400)
    """
    if not OPENGL_AVAILABLE:
        return
        
    glColor4f(*color)
    # Floor extends from X=110 to X=-390 (toward Unit 3), Z=-200 to z_max
    z_back = z_max if z_max is not None else 400
    glBegin(GL_QUADS)
    glVertex3f(110, y_level, -200)
    glVertex3f(-390, y_level, -200)
    glVertex3f(-390, y_level, z_back)
    glVertex3f(110, y_level, z_back)
    glEnd()


def draw_cylinder(position: Tuple[float, float, float], radius: float, height: float,
                  color: Tuple[float, ...], segments: int = 16):
    """
    Draw a vertical cylinder.
    
    Args:
        position: Base (x, y, z) position
        radius: Cylinder radius
        height: Cylinder height
        color: RGBA tuple
        segments: Number of segments
    """
    if not OPENGL_AVAILABLE:
        return
        
    glPushMatrix()
    glTranslatef(position[0], position[1], position[2])
    glColor4f(*color)
    
    quadric = gluNewQuadric()
    glRotatef(-90, 1, 0, 0)  # Rotate to stand upright
    gluCylinder(quadric, radius, radius, height, segments, 1)
    gluDeleteQuadric(quadric)
    
    glPopMatrix()


def draw_cone(position: Tuple[float, float, float], base_radius: float, height: float,
              color: Tuple[float, ...], segments: int = 16):
    """
    Draw a cone pointing upward.
    
    Args:
        position: Base (x, y, z) position
        base_radius: Radius at base
        height: Cone height
        color: RGBA tuple
        segments: Number of segments
    """
    if not OPENGL_AVAILABLE:
        return
        
    glPushMatrix()
    glTranslatef(position[0], position[1], position[2])
    glColor4f(*color)
    
    quadric = gluNewQuadric()
    glRotatef(-90, 1, 0, 0)
    gluCylinder(quadric, base_radius, 0, height, segments, 1)
    gluDeleteQuadric(quadric)
    
    glPopMatrix()


def draw_axis_lines(origin: Tuple[float, float, float], length: float = 50):
    """
    Draw XYZ axis lines from origin.
    
    Args:
        origin: (x, y, z) origin point
        length: Length of each axis line
    """
    if not OPENGL_AVAILABLE:
        return
        
    glLineWidth(3)
    glBegin(GL_LINES)
    
    # X axis - Red (pointing right/positive)
    glColor4f(1, 0, 0, 1)
    glVertex3f(origin[0], origin[1], origin[2])
    glVertex3f(origin[0] + length, origin[1], origin[2])
    
    # Y axis - Green (pointing up)
    glColor4f(0, 1, 0, 1)
    glVertex3f(origin[0], origin[1], origin[2])
    glVertex3f(origin[0], origin[1] + length, origin[2])
    
    # Z axis - Blue (pointing forward)
    glColor4f(0, 0, 1, 1)
    glVertex3f(origin[0], origin[1], origin[2])
    glVertex3f(origin[0], origin[1], origin[2] + length)
    
    glEnd()
    glLineWidth(1)


def draw_camera_cone(position: Tuple[float, float, float], 
                     rotation: dict, 
                     color: Tuple[float, ...],
                     cone_length: float = 80,
                     cone_half_width: float = 30):
    """
    Draw a camera viewing cone with rotation.
    
    Args:
        position: Camera (x, y, z) position
        rotation: Dict with 'pitch', 'yaw', 'roll' in degrees
        color: RGBA tuple
        cone_length: Length of viewing cone
        cone_half_width: Half-width at end
    """
    if not OPENGL_AVAILABLE:
        return
        
    glPushMatrix()
    glTranslatef(position[0], position[1], position[2])
    
    # Apply rotations: Yaw (Y), then Pitch (X), then Roll (Z)
    glRotatef(rotation.get('yaw', 0), 0, 1, 0)
    glRotatef(rotation.get('pitch', 0), 1, 0, 0)
    glRotatef(rotation.get('roll', 0), 0, 0, 1)
    
    glColor4f(*color)
    
    glBegin(GL_LINES)
    # Lines from camera to viewing direction corners
    glVertex3f(0, 0, 0)
    glVertex3f(-cone_half_width, -cone_half_width * 0.6, cone_length)
    glVertex3f(0, 0, 0)
    glVertex3f(cone_half_width, -cone_half_width * 0.6, cone_length)
    glVertex3f(0, 0, 0)
    glVertex3f(cone_half_width, cone_half_width * 0.6, cone_length)
    glVertex3f(0, 0, 0)
    glVertex3f(-cone_half_width, cone_half_width * 0.6, cone_length)
    # Connect the corners to form rectangle at end
    glVertex3f(-cone_half_width, -cone_half_width * 0.6, cone_length)
    glVertex3f(cone_half_width, -cone_half_width * 0.6, cone_length)
    glVertex3f(cone_half_width, -cone_half_width * 0.6, cone_length)
    glVertex3f(cone_half_width, cone_half_width * 0.6, cone_length)
    glVertex3f(cone_half_width, cone_half_width * 0.6, cone_length)
    glVertex3f(-cone_half_width, cone_half_width * 0.6, cone_length)
    glVertex3f(-cone_half_width, cone_half_width * 0.6, cone_length)
    glVertex3f(-cone_half_width, -cone_half_width * 0.6, cone_length)
    # Center line (optical axis)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, cone_length)
    glEnd()
    
    glPopMatrix()
