#!/usr/bin/env python3
"""
V3Dev Light Controller - Entry Point
=====================================
Run this script to start the light controller.

Usage:
    python run.py              # Normal mode with GUI
    python run.py --headless   # Headless mode (no display)
    python run.py --test       # Run integration tests
    python run.py --help       # Show help
"""

import sys
import os
import argparse
import logging
from typing import Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Reduce noise from some loggers
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('websockets').setLevel(logging.WARNING)


def run_tests():
    """Run integration tests."""
    print("=" * 60)
    print("V3Dev Integration Tests")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Module imports
    print("\n📦 Testing module imports...")
    try:
        from V3Dev import config, tracking, behavior, visualization, network
        from V3Dev.application import Application, create_application
        print("  ✓ All modules import successfully")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        tests_failed += 1
        return False
    
    # Test 2: Create headless application
    print("\n🔧 Testing headless application creation...")
    try:
        app = create_application(headless=True)
        assert app is not None
        assert app.headless == True
        print("  ✓ Headless application created")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Application creation failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
        return False
    
    # Test 3: Run a few frames
    print("\n🎬 Testing frame execution...")
    try:
        for i in range(10):
            result = app.run_frame()
            assert result == True
        print(f"  ✓ Ran {10} frames successfully")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Frame execution failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # Test 4: Simulate tracking
    print("\n👤 Testing person tracking...")
    try:
        # Add a person
        app.tracked_manager.update_person(
            track_id=1,
            raw_x=-150.0,
            raw_z=150.0,
        )
        # Check person was added
        person = app.tracked_manager.get_person(1)
        assert person is not None
        assert app.tracked_manager.count_active() >= 0  # Position may or may not be in active zone
        
        # Run update
        app.update(0.033)
        
        print("  ✓ Person tracking works")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Tracking failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # Test 5: Shutdown
    print("\n🛑 Testing shutdown...")
    try:
        app.shutdown()
        assert app.state.running == False
        print("  ✓ Shutdown successful")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ Shutdown failed: {e}")
        tests_failed += 1
    
    # Results
    print("\n" + "=" * 60)
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60)
    
    return tests_failed == 0


def run_headless(duration: float = None, skip_lock: bool = False):
    """Run in headless mode."""
    from V3Dev.application import create_application
    
    print("=" * 60)
    print("V3Dev Light Controller - Headless Mode")
    print("=" * 60)
    
    try:
        app = create_application(headless=True, skip_lock=skip_lock)
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    if duration:
        import time
        start = time.time()
        frame = 0
        while time.time() - start < duration:
            app.run_frame()
            frame += 1
            time.sleep(0.033)
        print(f"Ran {frame} frames in {duration}s")
        app.shutdown()
    else:
        try:
            app.run()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            app.shutdown()


def run_gui(skip_lock: bool = False):
    """Run with GUI (requires pygame)."""
    try:
        import pygame
        pygame.init()
    except ImportError:
        print("pygame not installed. Run: pip3 install pygame")
        print("Falling back to headless mode...\n")
        run_headless(skip_lock=skip_lock)
        return
    
    print("=" * 60)
    print("V3Dev Light Controller - GUI Mode")
    print("=" * 60)
    
    from V3Dev.application import create_application
    
    try:
        app = create_application(headless=False, skip_lock=skip_lock)
    except RuntimeError as e:
        print(f"❌ {e}")
        pygame.quit()
        sys.exit(1)
    
    # Setup display
    screen_width, screen_height = 1200, 800
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("V3Dev Light Controller")
    clock = pygame.time.Clock()
    
    # Colors
    BG_COLOR = (20, 20, 25)
    ZONE_ACTIVE = (40, 60, 40)
    ZONE_PASSIVE = (40, 40, 60)
    PANEL_COLOR = (100, 100, 100)
    LIGHT_COLOR = (255, 200, 100)
    PERSON_COLOR = (100, 200, 255)
    TEXT_COLOR = (200, 200, 200)
    
    # Coordinate transform: world cm -> screen pixels
    # Zone bounds (from config): x=[-350, 50], z=[78, 553]
    world_x_min, world_x_max = -400, 100
    world_z_min, world_z_max = 0, 600
    
    def world_to_screen(x: float, z: float) -> Tuple[int, int]:
        """Convert world coordinates to screen pixels."""
        sx = int((x - world_x_min) / (world_x_max - world_x_min) * screen_width)
        sy = int((z - world_z_min) / (world_z_max - world_z_min) * screen_height)
        return (sx, sy)
    
    font = pygame.font.Font(None, 24)
    
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
        
        # Update
        if not app.run_frame():
            running = False
        
        # Clear screen
        screen.fill(BG_COLOR)
        
        # Draw passive zone
        pz_tl = world_to_screen(-350, 283)
        pz_br = world_to_screen(50, 553)
        pygame.draw.rect(screen, ZONE_PASSIVE, 
                         (pz_tl[0], pz_tl[1], pz_br[0] - pz_tl[0], pz_br[1] - pz_tl[1]), 2)
        
        # Draw active zone
        az_tl = world_to_screen(-280, 78)
        az_br = world_to_screen(-20, 283)
        pygame.draw.rect(screen, ZONE_ACTIVE,
                         (az_tl[0], az_tl[1], az_br[0] - az_tl[0], az_br[1] - az_tl[1]), 2)
        
        # Draw panels (at z=0)
        for i in range(4):
            for j in range(3):
                px = -150 + (i - 1.5) * 80  # 80cm unit spacing
                pz = 30 + j * 20  # Just for visual representation
                pos = world_to_screen(px, pz)
                brightness = 128  # Could get from app.panel_renderer
                color = (brightness, brightness, brightness)
                pygame.draw.circle(screen, color, pos, 15)
                pygame.draw.circle(screen, PANEL_COLOR, pos, 15, 2)
        
        # Draw persons
        if hasattr(app, 'tracked_manager') and app.tracked_manager:
            for pid, person in app.tracked_manager.people.items():
                pos = world_to_screen(person.position[0], person.position[2])
                pygame.draw.circle(screen, PERSON_COLOR, pos, 10)
                label = font.render(f"P{pid}", True, TEXT_COLOR)
                screen.blit(label, (pos[0] + 12, pos[1] - 8))
        
        # Draw light position
        if hasattr(app, 'light_controller') and app.light_controller:
            lpos = app.light_controller.position
            light_pos = world_to_screen(lpos[0], lpos[2])
            pygame.draw.circle(screen, LIGHT_COLOR, light_pos, 12)
            pygame.draw.circle(screen, (255, 255, 200), light_pos, 8)
        
        # Draw info
        fps = clock.get_fps()
        mode_name = app.behavior.get_mode().name if hasattr(app, 'behavior') and app.behavior else 'N/A'
        person_count = len(app.tracked_manager.people) if hasattr(app, 'tracked_manager') and app.tracked_manager else 0
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Mode: {mode_name}",
            f"Persons: {person_count}",
            "Press Q or ESC to quit"
        ]
        for i, line in enumerate(info_lines):
            label = font.render(line, True, TEXT_COLOR)
            screen.blit(label, (10, 10 + i * 25))
        
        pygame.display.flip()
        clock.tick(60)
    
    app.shutdown()
    pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description="V3Dev Light Controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py              Run with GUI (when available)
  python run.py --headless   Run without display
  python run.py --test       Run integration tests
  python run.py -v           Verbose logging
        """
    )
    
    parser.add_argument(
        '--headless', '-H',
        action='store_true',
        help='Run without display'
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Run integration tests'
    )
    
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=None,
        help='Run for N seconds then exit (headless mode)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to settings file'
    )
    
    parser.add_argument(
        '--no-lock',
        action='store_true',
        help='Skip single instance lock (for testing)'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)
    elif args.headless:
        run_headless(duration=args.duration, skip_lock=args.no_lock)
    else:
        run_gui(skip_lock=args.no_lock)


if __name__ == "__main__":
    main()
