#!/usr/bin/env python3
"""
Test script for V3Dev network module.
Run: python -m V3Dev.test_network
"""

import os
import sys
import tempfile
import time

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_websocket():
    """Test WebSocket components."""
    print("\n📡 Testing WebSocket module...")
    
    from V3Dev.network.websocket import (
        WebSocketConfig,
        StateSerializer,
        MockWebSocketBroadcaster
    )
    
    # Test config
    config = WebSocketConfig(host="127.0.0.1", port=8080)
    assert config.host == "127.0.0.1"
    assert config.port == 8080
    print("  ✓ WebSocketConfig works")
    
    # Test serializer
    serializer = StateSerializer()
    
    # First call should compute
    state1 = {
        "people": [{"id": 1, "x": 100, "z": 200}],
        "lights": [0.5, 0.6, 0.7]
    }
    result1 = serializer.serialize(state1)
    assert result1 is not None
    assert "people" in result1
    print("  ✓ StateSerializer serializes correctly")
    
    # Test cache - same state should return same result
    result2 = serializer.serialize(state1)
    assert result2 == result1
    print("  ✓ StateSerializer caches results")
    
    # Different state should produce different result
    state2 = {"people": [], "lights": [0.1]}
    result3 = serializer.serialize(state2)
    assert result3 != result1
    print("  ✓ StateSerializer detects changes")
    
    # Test mock broadcaster
    mock_ws = MockWebSocketBroadcaster()
    mock_ws.update_state({"test": "data"})
    assert len(mock_ws._states) == 1
    mock_ws.start()
    assert mock_ws.is_running
    mock_ws.stop()
    assert not mock_ws.is_running
    print("  ✓ MockWebSocketBroadcaster works")
    
    print("  ✅ WebSocket module OK")


def test_health():
    """Test health monitoring components."""
    print("\n🏥 Testing Health module...")
    
    from V3Dev.network.health import (
        HealthMonitor,
        HealthStats,
        ErrorTracker,
        UptimeTracker
    )
    
    # Test error tracker
    tracker = ErrorTracker()
    tracker.record("test_component")
    tracker.record("test_component")
    tracker.record("other_component")
    
    total = tracker.get_total()
    assert total == 3
    test_count = tracker.get_total("test_component")
    assert test_count == 2
    print("  ✓ ErrorTracker records errors")
    
    # Test rate calculation
    rate = tracker.get_rate()
    assert rate >= 0
    print(f"  ✓ ErrorTracker rate: {rate:.4f}/s")
    
    # Test uptime tracker
    uptime = UptimeTracker()
    uptime.start()
    time.sleep(0.05)
    uptime_sec = uptime.uptime_seconds
    assert uptime_sec >= 0.04
    
    uptime_str = uptime.format_uptime()
    # Should have some time format
    assert len(uptime_str) > 0
    print("  ✓ UptimeTracker works")
    
    # Test health stats
    stats = HealthStats()
    assert stats.frame_count == 0
    assert stats.avg_fps == 0.0
    print("  ✓ HealthStats dataclass works")
    
    # Test health monitor
    monitor = HealthMonitor()
    monitor.start()
    
    # Simulate some frames
    for _ in range(10):
        monitor.tick()
        time.sleep(0.01)
    
    assert monitor._frame_count == 10
    print("  ✓ HealthMonitor frame tracking")
    
    # Update state
    monitor.update_state(mode="idle", active_count=3)
    
    # Get stats
    stats = monitor.get_stats()
    assert stats.mode == "idle"
    assert stats.active_count == 3
    print("  ✓ HealthMonitor state tracking")
    
    # Get log message
    log_msg = monitor.get_log_message()
    assert "HEALTH" in log_msg
    assert "mode=idle" in log_msg
    print("  ✓ HealthMonitor log message")
    
    monitor.stop()
    
    print("  ✅ Health module OK")


def test_persistence():
    """Test settings persistence components."""
    print("\n💾 Testing Persistence module...")
    
    from V3Dev.network.persistence import (
        SettingsStore,
        TrackerSettings,
        BehaviorSettings,
        PersistenceConfig
    )
    
    # Test with temp file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        # Test settings store
        store = SettingsStore(temp_path)
        
        # Set some values
        store.set("brightness", 0.75)
        store.set("wander_enabled", True)
        store.set("speed", 50.0)
        
        assert store.is_dirty
        assert store.get("brightness") == 0.75
        print("  ✓ SettingsStore set/get works")
        
        # Save
        saved = store.save()
        assert saved
        assert not store.is_dirty
        print("  ✓ SettingsStore save works")
        
        # Load in new store
        store2 = SettingsStore(temp_path)
        values = store2.load()
        assert values["brightness"] == 0.75
        assert values["wander_enabled"] == True
        print("  ✓ SettingsStore load works")
        
        # Test set_many
        store.set_many({"a": 1, "b": 2, "c": 3})
        assert store.is_dirty
        assert store.get("b") == 2
        print("  ✓ SettingsStore set_many works")
        
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_path + ".tmp"):
            os.remove(temp_path + ".tmp")
    
    # Test TrackerSettings
    ts = TrackerSettings(offset_x=10.0, scale_x=1.5)
    assert ts.offset_x == 10.0
    
    ts_dict = ts.to_dict()
    assert ts_dict["offset_x"] == 10.0
    
    ts2 = TrackerSettings.from_dict({"offset_x": 20.0, "unknown_field": 999})
    assert ts2.offset_x == 20.0
    print("  ✓ TrackerSettings dataclass works")
    
    # Test BehaviorSettings
    bs = BehaviorSettings(brightness_min=10, move_speed=100.0)
    assert bs.brightness_min == 10
    assert bs.move_speed == 100.0
    print("  ✓ BehaviorSettings dataclass works")
    
    # Test PersistenceConfig
    config = PersistenceConfig(auto_save_interval=5.0)
    assert config.auto_save_interval == 5.0
    print("  ✓ PersistenceConfig works")
    
    print("  ✅ Persistence module OK")


def test_imports():
    """Test that all imports work correctly."""
    print("\n📦 Testing module imports...")
    
    # Test top-level network import
    from V3Dev import network
    print("  ✓ V3Dev.network imports")
    
    # Test individual imports
    from V3Dev.network import WebSocketBroadcaster, MockWebSocketBroadcaster
    from V3Dev.network import HealthMonitor, HealthStats
    from V3Dev.network import SettingsStore, SettingsManager
    print("  ✓ All network components importable")
    
    # Test that V3Dev imports all modules
    import V3Dev
    assert hasattr(V3Dev, 'config')
    assert hasattr(V3Dev, 'tracking')
    assert hasattr(V3Dev, 'behavior')
    assert hasattr(V3Dev, 'visualization')
    assert hasattr(V3Dev, 'network')
    print("  ✓ V3Dev has all 5 modules")
    
    print("  ✅ Imports OK")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("V3Dev Network Module Tests")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("WebSocket", test_websocket),
        ("Health", test_health),
        ("Persistence", test_persistence),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n  ❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
