/**
 * Drop Ceiling - Public Viewer
 * Three.js visualization of the light installation
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// =============================================================================
// CONFIGURATION (matching Python controller)
// =============================================================================

const CONFIG = {
    // Panel dimensions (cm)
    PANEL_SIZE: 60,
    UNIT_SPACING: 80,
    
    // Panel positions relative to unit center (y, z)
    PANEL_LOCAL_POSITIONS: {
        1: [90, 0],
        2: [30, 12],
        3: [30, -12],
    },
    
    // Panel angles (degrees from vertical)
    PANEL_ANGLES: {
        1: 0,
        2: 22.5,
        3: -22.5,
    },
    
    // Trackzone dimensions (matches Python TRACKZONE_PARAMS)
    TRACKZONE: {
        width: 260,       // X dimension
        depth: 205,       // Z dimension
        height: 300,      // Y dimension
        offset_z: 78,
        offset_y: -66,
        center_x: -150,   // Negative X (units are at negative X)
    },
    
    // Wander box (matches Python WANDER_BOX)
    WANDER_BOX: {
        min_x: -280,
        max_x: -20,
        min_y: 0,
        max_y: 150,
        min_z: -28,
        max_z: 32,
    },
    
    // WebSocket settings
    WS_URL: 'wss://cvtower.tail830204.ts.net/',  // Production Tailscale Funnel URL
    WS_PORT: 8765,
    RECONNECT_DELAY: 3000,
};

// =============================================================================
// STATE
// =============================================================================

let scene, camera, renderer, controls;
let panels = [];
let lightSphere, lightGlow, falloffSphere;
let wanderBoxMesh = null;  // Dynamic wander box wireframe
let trackedPeople = {};
let wsConnection = null;
let currentState = null;
let lastReportVersion = -1;  // Track report version to avoid redundant updates
let latestDailyReport = null;  // Store latest report for when panel opens
let latestRealtimeTrends = null;  // Store realtime trends
let lastAutoTuning = null;  // Store latest auto-tuning data

// Wander box lerping state
let currentWanderBox = { ...CONFIG.WANDER_BOX };  // Current displayed values
let targetWanderBox = { ...CONFIG.WANDER_BOX };   // Target values from broadcast
const WANDER_BOX_LERP_SPEED = 0.08;  // Lerp factor (0-1, higher = faster)

// Trends overlay state
const TRENDS_HISTORY_SIZE = 60;  // Number of data points to display (1 minute at 1/sec)
const TRENDS_SAMPLE_INTERVAL = 1000;  // Sample every 1 second
let trendsHistory = [];  // Array of { active, passive, timestamp }
let lastTrendsSample = 0;
let trendsInputCanvas = null;
let trendsInputCtx = null;

// =============================================================================
// INITIALIZATION
// =============================================================================

function init() {
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0c);
    scene.fog = new THREE.Fog(0x0a0a0c, 800, 2000);
    
    // Camera - fixed position for mobile portrait view
    camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 1, 3000);
    
    // Position camera higher and farther back to include more passive zone
    // Panels center at X = -150, passive zone ends around Z = 553
    camera.position.set(-138.86, 607.56, 889.87);
    camera.lookAt(-140.45, 18.08, 434.19);
    
    // Renderer
    const canvas = document.getElementById('viewer');
    renderer = new THREE.WebGLRenderer({ 
        canvas,
        antialias: true,
        alpha: true,
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    
    // Orbit controls - allows user to rotate/pan/zoom camera
    controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(-140.45, 18.08, 434.19);  // Look deeper into passive zone
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 100;
    controls.maxDistance = 2500;
    controls.update();
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0x222233, 0.5);
    scene.add(ambientLight);
    
    // Build scene
    createFloor();
    createPanels();
    createPointLight();
    createTrackzone();
    initTrendsOverlay();
    
    // Events
    window.addEventListener('resize', onWindowResize);
    window.addEventListener('keydown', (event) => {
        if (event.key.toLowerCase() === 'p') {
            printCameraState();
        }
    });

    const trendsOverlay = document.getElementById('trends-overlay');
    if (trendsOverlay) {
        trendsOverlay.addEventListener('wheel', (event) => event.stopPropagation(), { passive: true });
        trendsOverlay.addEventListener('pointerdown', () => { controls.enabled = false; });
        trendsOverlay.addEventListener('pointerup', () => { controls.enabled = true; });
        trendsOverlay.addEventListener('pointercancel', () => { controls.enabled = true; });
        trendsOverlay.addEventListener('pointerleave', () => { controls.enabled = true; });
    }
    
    // About modal
    document.getElementById('about-btn').addEventListener('click', () => {
        document.getElementById('about-overlay').classList.remove('hidden');
    });
    document.getElementById('about-close').addEventListener('click', () => {
        document.getElementById('about-overlay').classList.add('hidden');
    });
    document.getElementById('about-overlay').addEventListener('click', (e) => {
        if (e.target.id === 'about-overlay') {
            document.getElementById('about-overlay').classList.add('hidden');
        }
    });

    
    // Trends panel toggle (elements may be commented out in HTML)
    const trendsPanel = document.getElementById('trends-panel');
    const trendsBtn = document.getElementById('trends-btn');
    const trendsHeader = document.getElementById('trends-header');
    
    // Only set up trends panel if elements exist
    if (trendsPanel && trendsBtn) {
        // Start hidden
        trendsPanel.classList.add('hidden');
        
        trendsBtn.addEventListener('click', () => {
            if (trendsPanel.classList.contains('hidden')) {
                trendsPanel.classList.remove('hidden');
                trendsPanel.classList.remove('collapsed');
                // Refresh the display with latest data when opening
                updateTrendsDisplay(latestDailyReport, latestRealtimeTrends);
            } else {
                trendsPanel.classList.add('hidden');
            }
        });
        
        if (trendsHeader) {
            trendsHeader.addEventListener('click', () => {
                trendsPanel.classList.toggle('collapsed');
            });
        }
    }
    
    // Auto-connect to Tailscale endpoint
    connectWebSocket(CONFIG.WS_URL);
    
    // Start render loop
    animate();
}

// =============================================================================
// SCENE CONSTRUCTION
// =============================================================================

function createFloor() {
    // Floor plane (storefront level) - stops at trackzone
    const floorGeom = new THREE.PlaneGeometry(500, CONFIG.TRACKZONE.offset_z + 200);
    const floorMat = new THREE.MeshBasicMaterial({ 
        color: 0x1a1a1f,
        transparent: true,
        opacity: 0.8,
    });
    const floor = new THREE.Mesh(floorGeom, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(CONFIG.TRACKZONE.center_x, 0, (CONFIG.TRACKZONE.offset_z - 200) / 2);
    scene.add(floor);
    
    // Grid lines for depth
    const gridHelper = new THREE.GridHelper(500, 20, 0x333340, 0x222230);
    gridHelper.position.set(CONFIG.TRACKZONE.center_x, 0.1, 0);
    scene.add(gridHelper);
}

function createPanels() {
    // Create all 12 panels (4 units × 3 panels each)
    const PANEL_DEPTH = 1.5; // Box depth in cm
    const FRAME_WIDTH = 4; // Frame border width in cm
    const INNER_SIZE = CONFIG.PANEL_SIZE - (FRAME_WIDTH * 2); // Lit area size
    
    for (let unit = 0; unit < 4; unit++) {
        // Unit X positions: Unit 0 at -30, Unit 1 at -110, Unit 2 at -190, Unit 3 at -270
        const unitX = -(unit * CONFIG.UNIT_SPACING + 30);
        
        for (let panelNum = 1; panelNum <= 3; panelNum++) {
            const [localY, localZ] = CONFIG.PANEL_LOCAL_POSITIONS[panelNum];
            const angle = CONFIG.PANEL_ANGLES[panelNum];
            
            // Create a group to hold frame and lit area
            const panelGroup = new THREE.Group();
            panelGroup.position.set(unitX, localY, localZ);
            panelGroup.rotation.x = THREE.MathUtils.degToRad(-angle);
            
            // Frame (outer box) - stays dark
            const frameGeom = new THREE.BoxGeometry(CONFIG.PANEL_SIZE, CONFIG.PANEL_SIZE, PANEL_DEPTH);
            const frameMat = new THREE.MeshBasicMaterial({
                color: 0x2a2a2f,
            });
            const frame = new THREE.Mesh(frameGeom, frameMat);
            panelGroup.add(frame);
            
            // Add wireframe edges to frame
            const edges = new THREE.EdgesGeometry(frameGeom);
            const line = new THREE.LineSegments(
                edges,
                new THREE.LineBasicMaterial({ color: 0x333344, linewidth: 1 })
            );
            frame.add(line);
            
            // Lit area (inner plane on front face)
            const litGeom = new THREE.PlaneGeometry(INNER_SIZE, INNER_SIZE);
            const litMat = new THREE.MeshBasicMaterial({
                color: 0x333333,
                transparent: true,
            });
            const litArea = new THREE.Mesh(litGeom, litMat);
            litArea.position.z = PANEL_DEPTH / 2 + 0.1; // Slightly in front of frame
            panelGroup.add(litArea);
            
            scene.add(panelGroup);
            panels.push({
                mesh: litArea, // The lit area is what changes brightness
                frame: frame,
                group: panelGroup,
                unit: unit,
                panelNum: panelNum,
                brightness: 0,
            });
        }
    }
}

function createPointLight() {
    // Main light sphere
    const sphereGeom = new THREE.SphereGeometry(8, 24, 24);
    const sphereMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    lightSphere = new THREE.Mesh(sphereGeom, sphereMat);
    lightSphere.position.set(120, 60, -30);
    scene.add(lightSphere);
    
    // Glow effect (larger transparent sphere)
    const glowGeom = new THREE.SphereGeometry(15, 24, 24);
    const glowMat = new THREE.MeshBasicMaterial({
        color: 0xffffcc,
        transparent: true,
        opacity: 0.3,
    });
    lightGlow = new THREE.Mesh(glowGeom, glowMat);
    lightSphere.add(lightGlow);
    
    // Falloff radius indicator
    const falloffGeom = new THREE.SphereGeometry(50, 32, 16);
    const falloffMat = new THREE.MeshBasicMaterial({
        color: 0xffcc00,
        transparent: true,
        opacity: 0.05,
        wireframe: true,
    });
    falloffSphere = new THREE.Mesh(falloffGeom, falloffMat);
    lightSphere.add(falloffSphere);
}

function createTrackzone() {
    // Active trackzone wireframe (cyan)
    const tz = CONFIG.TRACKZONE;
    const tzGeom = new THREE.BoxGeometry(tz.width, tz.height, tz.depth);
    const tzEdges = new THREE.EdgesGeometry(tzGeom);
    const tzLine = new THREE.LineSegments(
        tzEdges,
        new THREE.LineBasicMaterial({ color: 0x00ffff, transparent: true, opacity: 0.3 })
    );
    tzLine.position.set(tz.center_x, tz.offset_y + tz.height / 2, tz.offset_z + tz.depth / 2);
    scene.add(tzLine);
    
    // Initial wander box wireframe (green) - will be updated dynamically
    updateWanderBox(CONFIG.WANDER_BOX);
}

function updateWanderBox(wb) {
    // Remove old wander box if exists
    if (wanderBoxMesh) {
        scene.remove(wanderBoxMesh);
        wanderBoxMesh.geometry.dispose();
        wanderBoxMesh.material.dispose();
    }
    
    // Create new wander box with updated dimensions
    const wbWidth = wb.max_x - wb.min_x;
    const wbHeight = wb.max_y - wb.min_y;
    const wbDepth = wb.max_z - wb.min_z;
    const wbGeom = new THREE.BoxGeometry(wbWidth, wbHeight, wbDepth);
    const wbEdges = new THREE.EdgesGeometry(wbGeom);
    wanderBoxMesh = new THREE.LineSegments(
        wbEdges,
        new THREE.LineBasicMaterial({ color: 0x00ff00, transparent: true, opacity: 0.5 })
    );
    // Position at center of wander box
    wanderBoxMesh.position.set(
        (wb.min_x + wb.max_x) / 2,
        (wb.min_y + wb.max_y) / 2,
        (wb.min_z + wb.max_z) / 2
    );
    scene.add(wanderBoxMesh);
}

// =============================================================================
// WEBSOCKET CONNECTION
// =============================================================================

function connectWebSocket() {
    const url = CONFIG.WS_URL;
    
    updateStatus('connecting', 'Connecting...');
    
    try {
        wsConnection = new WebSocket(url);
        
        wsConnection.onopen = () => {
            updateStatus('connected', 'Live');
            console.log('WebSocket connected to', url);
        };
        
        wsConnection.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleStateUpdate(data);
            } catch (e) {
                console.error('Failed to parse message:', e);
            }
        };
        
        wsConnection.onclose = () => {
            updateStatus('error', 'Disconnected');
            console.log('WebSocket disconnected, reconnecting...');
            // Auto-reconnect after delay
            setTimeout(connectWebSocket, CONFIG.RECONNECT_DELAY);
        };
        
        wsConnection.onerror = (error) => {
            updateStatus('error', 'Connection Error');
            console.error('WebSocket error:', error);
        };
        
    } catch (e) {
        updateStatus('error', 'Failed to connect');
        console.error('WebSocket connection failed:', e);
        // Auto-retry after delay
        setTimeout(connectWebSocket, CONFIG.RECONNECT_DELAY);
    }
}

function updateStatus(state, text) {
    const statusEl = document.getElementById('connection-status');
    if (!statusEl) return;
    statusEl.textContent = text;
    statusEl.className = `subtitle ${state}`;
}

// =============================================================================
// STATE UPDATES
// =============================================================================

function handleStateUpdate(data) {
    currentState = data;
    
    // Update light position
    if (data.light) {
        lightSphere.position.set(data.light.x, data.light.y, data.light.z);
        
        // Update brightness/glow
        const brightness = data.light.brightness || 0.5;
        lightSphere.scale.setScalar(0.8 + brightness * 0.4);
        lightGlow.material.opacity = 0.2 + brightness * 0.3;
        
        // Update falloff radius
        const radius = data.light.falloff_radius || 50;
        falloffSphere.scale.setScalar(radius / 50);
    }
    
    // Update panel brightness
    // DMX values from Python are in range 1-50 (DMX_MIN to DMX_MAX)
    // Python sends panels in order: Unit 0 (panels 1-3), Unit 1 (panels 1-3), etc.
    // This matches the order panels are created in createPanels()
    if (data.panels) {
        data.panels.forEach((dmxValue, index) => {
            if (panels[index]) {
                const normalizedBrightness = (dmxValue - 1) / 49; // Map 1-50 to 0-1
                const curved = Math.pow(normalizedBrightness, 0.8); // Boost low values
                
                // Lit area: warm white, fully visible
                const r = 0.03 + curved * 0.97;
                const g = 0.03 + curved * 0.94;
                const b = 0.02 + curved * 0.78;
                panels[index].mesh.material.color.setRGB(r, g, b);
                panels[index].mesh.material.opacity = 0.1 + curved * 0.9;
                
                // Frame also picks up light - subtle ambient glow
                const frameGlow = curved * 0.35;
                panels[index].frame.material.color.setRGB(
                    0.16 + frameGlow,
                    0.16 + frameGlow * 0.95,
                    0.18 + frameGlow * 0.7
                );
                
                // Create/update glow plane if it doesn't exist
                if (!panels[index].glowPlane) {
                    const glowGeom = new THREE.PlaneGeometry(60, 60);
                    const glowMat = new THREE.MeshBasicMaterial({
                        color: 0xfff5e0,
                        transparent: true,
                        opacity: 0,
                        side: THREE.DoubleSide,
                        depthWrite: false,
                    });
                    const glowPlane = new THREE.Mesh(glowGeom, glowMat);
                    glowPlane.position.z = 1.2; // In front of panel
                    panels[index].group.add(glowPlane);
                    panels[index].glowPlane = glowPlane;
                }
                // Glow plane becomes visible at mid-high brightness
                const glowOpacity = Math.max(0, (curved - 0.2) * 0.5);
                panels[index].glowPlane.material.opacity = glowOpacity;
                panels[index].glowPlane.scale.setScalar(1.0 + curved * 0.3);
                
                panels[index].brightness = normalizedBrightness;
            }
        });
    }
    
    // Update tracked people
    if (data.people) {
        updateTrackedPeople(data.people);
    }
    
    // Update header subheading with status text
    if (data.status !== undefined) {
        const statusEl = document.getElementById('behavior-status');
        if (statusEl) statusEl.textContent = data.status || '';
    }
    
    // Update auto-tuning parameter display
    if (data.auto_tuning) {
        lastAutoTuning = data.auto_tuning;
        updateAutoTuneDisplay(data.auto_tuning);
    }

    
    // Update wander box target if changed (will lerp in animate loop)
    if (data.wander_box) {
        targetWanderBox = { ...data.wander_box };
    }
    
    // Store realtime trends (always update)
    if (data.realtime_trends) {
        latestRealtimeTrends = data.realtime_trends;
        // Update the trends display if panel is visible
        const trendsPanel = document.getElementById('trends-panel');
        if (trendsPanel && !trendsPanel.classList.contains('hidden')) {
            updateTrendsDisplay(latestDailyReport, latestRealtimeTrends);
        }
    }
    
    // Update daily report (only when report version changes)
    const reportVersion = data.report_version ?? 0;
    if (reportVersion !== lastReportVersion) {
        lastReportVersion = reportVersion;
        latestDailyReport = data.daily_report || null;
        // Update display if panel is visible
        const trendsPanel = document.getElementById('trends-panel');
        if (trendsPanel && !trendsPanel.classList.contains('hidden')) {
            updateTrendsDisplay(latestDailyReport, latestRealtimeTrends);
        }
    }

    updateTrendStats();
    updateTuningBars();
}

function printCameraState() {
    const state = {
        position: {
            x: Number(camera.position.x.toFixed(2)),
            y: Number(camera.position.y.toFixed(2)),
            z: Number(camera.position.z.toFixed(2)),
        },
        target: {
            x: Number(controls.target.x.toFixed(2)),
            y: Number(controls.target.y.toFixed(2)),
            z: Number(controls.target.z.toFixed(2)),
        },
        zoom: Number(camera.zoom.toFixed(2)),
        fov: Number(camera.fov.toFixed(2)),
    };
    console.log('Camera state:', state);
}

function updateTrackedPeople(peopleData) {
    const currentIds = new Set(Object.keys(trackedPeople).map(Number));
    const newIds = new Set(peopleData.map(p => p.id));
    
    // Remove people who left
    currentIds.forEach(id => {
        if (!newIds.has(id)) {
            scene.remove(trackedPeople[id]);
            delete trackedPeople[id];
        }
    });
    
    // Add or update people
    peopleData.forEach(person => {
        if (!trackedPeople[person.id]) {
            // Create new person representation
            const personMesh = createPersonMesh();
            scene.add(personMesh);
            trackedPeople[person.id] = personMesh;
        }
        
        // Update position
        const personMesh = trackedPeople[person.id];
        personMesh.position.set(person.x, person.y + 85, person.z);
        const labelId = person.daily_id ?? person.id;
        updateTextSprite(personMesh.userData.labelSprite, `#${labelId}`);
    });
}

function createPersonMesh() {
    const group = new THREE.Group();
    
    // Simple cylinder for body
    const bodyGeom = new THREE.CylinderGeometry(15, 15, 150, 12);
    const bodyMat = new THREE.MeshBasicMaterial({ 
        color: 0x66ee88,
        transparent: true,
        opacity: 0.85,
    });
    const body = new THREE.Mesh(bodyGeom, bodyMat);
    group.add(body);
    
    // Sphere for head
    const headGeom = new THREE.SphereGeometry(15, 12, 12);
    const head = new THREE.Mesh(headGeom, bodyMat);
    head.position.y = 85;
    group.add(head);
    
    const label = createTextSprite('');
    label.position.y = 120;
    group.add(label);
    group.userData.labelSprite = label;
    
    return group;
}

function createTextSprite(text) {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(70, 18, 1);
    sprite.userData = { canvas, ctx, texture, text: '' };
    updateTextSprite(sprite, text);
    return sprite;
}

function updateTextSprite(sprite, text) {
    if (!sprite || !sprite.userData) return;
    if (sprite.userData.text === text) return;
    const { canvas, ctx, texture } = sprite.userData;
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#ffffff';
    ctx.font = '28px Space Grotesk, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, w / 2, h / 2 + 2);
    texture.needsUpdate = true;
    sprite.userData.text = text;
}

function updateTrendStats() {
    const currentCountEl = document.getElementById('trend-current');
    const currentActiveEl = document.getElementById('trend-current-active');
    const currentPassiveEl = document.getElementById('trend-current-passive');
    if (currentCountEl) {
        const currentCount = currentState?.population?.current ?? currentState?.counts?.total ?? 0;
        currentCountEl.textContent = currentCount;
    }
    if (currentActiveEl) {
        const active = currentState?.population?.active ?? currentState?.counts?.active ?? 0;
        currentActiveEl.textContent = active;
    }
    if (currentPassiveEl) {
        const passive = currentState?.population?.passive ?? currentState?.counts?.passive ?? 0;
        currentPassiveEl.textContent = passive;
    }

}

function updateTuningBars() {
    const lastAdjustment = lastAutoTuning?.last_adjustment || {};
    const tuningRanges = [
        { key: 'short', weight: lastAdjustment.short_activity, available: latestRealtimeTrends?.short?.available },
        { key: 'med', weight: lastAdjustment.medium_activity, available: latestRealtimeTrends?.medium?.available },
        { key: 'long', weight: lastAdjustment.long_activity, available: latestRealtimeTrends?.long?.available },
    ];

    tuningRanges.forEach((range) => {
        const weight = typeof range.weight === 'number' ? Math.max(0, Math.min(range.weight, 1)) : null;
        const isAvailable = range.available && weight !== null;
        const activeWidth = isAvailable ? weight * 100 : 0;

        if (range.key === 'long') {
            const longRow = document.getElementById('trend-long-row');
            if (longRow) {
                longRow.style.display = isAvailable ? 'grid' : 'none';
            }
        }

        const activeEl = document.getElementById(`trend-${range.key}-bar-active`);
        const passiveEl = document.getElementById(`trend-${range.key}-bar-passive`);
        const totalEl = document.getElementById(`trend-${range.key}-total`);
        const weightEl = document.getElementById(`trend-${range.key}-weight`);

        if (activeEl) activeEl.style.width = `${activeWidth}%`;
        if (passiveEl) passiveEl.style.width = '0%';
        if (weightEl) weightEl.textContent = isAvailable ? `${Math.round(weight * 100)}%` : '--';
        if (totalEl) totalEl.textContent = isAvailable ? '' : '-';
    });
}

function updateAutoTuneDisplay(autoTuning) {
    const statusEl = document.getElementById('behavior-status-label');
    if (statusEl) {
        const enabled = autoTuning?.enabled;
        statusEl.textContent = enabled ? 'On' : 'Off';
        statusEl.classList.toggle('on', Boolean(enabled));
    }

    const params = autoTuning?.params || {};
    const paramKeys = [
        'responsiveness',
        'energy',
        'attention_span',
        'sociability',
        'exploration',
        'memory',
    ];

    paramKeys.forEach((key) => {
        const value = typeof params[key] === 'number' ? params[key] : 0;
        const clamped = Math.min(Math.max(value, 0), 1);
        const bar = document.getElementById(`param-${key}`);
        if (bar) {
            bar.style.width = `${clamped * 100}%`;
        }
    });

    const pulse = document.getElementById('tuning-pulse');
    if (pulse) {
        const lastAdjustment = autoTuning?.last_adjustment?.timestamp ?? 0;
        const age = Date.now() / 1000 - lastAdjustment;
        pulse.classList.toggle('active', age >= 0 && age < 6);
    }

}

// =============================================================================
// ANIMATION
// =============================================================================

function animate() {
    requestAnimationFrame(animate);
    
    // Update orbit controls
    controls.update();
    
    // Lerp wander box towards target
    lerpWanderBox();
    
    // Pulse the light glow slightly
    if (lightGlow && currentState?.light) {
        const pulse = Math.sin(Date.now() * 0.003) * 0.1 + 1;
        lightGlow.scale.setScalar(pulse);
    }
    
    renderer.render(scene, camera);
}

function lerpWanderBox() {
    // Check if we need to update (any value different)
    const needsUpdate = 
        Math.abs(currentWanderBox.min_x - targetWanderBox.min_x) > 0.01 ||
        Math.abs(currentWanderBox.max_x - targetWanderBox.max_x) > 0.01 ||
        Math.abs(currentWanderBox.min_y - targetWanderBox.min_y) > 0.01 ||
        Math.abs(currentWanderBox.max_y - targetWanderBox.max_y) > 0.01 ||
        Math.abs(currentWanderBox.min_z - targetWanderBox.min_z) > 0.01 ||
        Math.abs(currentWanderBox.max_z - targetWanderBox.max_z) > 0.01;
    
    if (!needsUpdate) return;
    
    // Lerp each value towards target
    currentWanderBox.min_x += (targetWanderBox.min_x - currentWanderBox.min_x) * WANDER_BOX_LERP_SPEED;
    currentWanderBox.max_x += (targetWanderBox.max_x - currentWanderBox.max_x) * WANDER_BOX_LERP_SPEED;
    currentWanderBox.min_y += (targetWanderBox.min_y - currentWanderBox.min_y) * WANDER_BOX_LERP_SPEED;
    currentWanderBox.max_y += (targetWanderBox.max_y - currentWanderBox.max_y) * WANDER_BOX_LERP_SPEED;
    currentWanderBox.min_z += (targetWanderBox.min_z - currentWanderBox.min_z) * WANDER_BOX_LERP_SPEED;
    currentWanderBox.max_z += (targetWanderBox.max_z - currentWanderBox.max_z) * WANDER_BOX_LERP_SPEED;
    
    // Update the visual
    updateWanderBox(currentWanderBox);
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    resizeTrendsCanvas();
}

// =============================================================================
// TRENDS OVERLAY (Bottom of screen time visualization)
// =============================================================================

function initTrendsOverlay() {
    trendsInputCanvas = document.getElementById('trends-input-canvas');
    if (!trendsInputCanvas) return;
    
    trendsInputCtx = trendsInputCanvas.getContext('2d');
    resizeTrendsCanvas();
    
    // Start the render loop for trends
    requestAnimationFrame(renderTrendsOverlay);
}

function resizeTrendsCanvas() {
    if (!trendsInputCanvas || !trendsInputCtx) return;

    const dpr = window.devicePixelRatio || 1;
    const inputRect = trendsInputCanvas.parentElement.getBoundingClientRect();

    trendsInputCanvas.width = inputRect.width * dpr;
    trendsInputCanvas.height = inputRect.height * dpr;
    trendsInputCtx.setTransform(1, 0, 0, 1, 0, 0);
    trendsInputCtx.scale(dpr, dpr);
}

function sampleTrendsData() {
    // Sample event counts at regular intervals
    // These are cumulative counts of zone-crossing events within time windows
    const now = Date.now();
    if (now - lastTrendsSample < TRENDS_SAMPLE_INTERVAL) return;
    
    lastTrendsSample = now;
    
    // Get event counts from realtime trends
    // Engaged = active zone events (triggered installation response)
    // Passing = passive zone events (walked through without engaging)
    let engaged = 0;
    let passing = 0;
    
    if (latestRealtimeTrends?.recent?.available) {
        // Use realtime trends data (events in last 1 minute window)
        engaged = latestRealtimeTrends.recent.active || 0;
        passing = latestRealtimeTrends.recent.passive || 0;
    } else if (currentState?.population) {
        // Use population counts (active/passive from zone classification)
        engaged = currentState.population.active || 0;
        passing = currentState.population.passive || 0;
    } else if (currentState?.counts) {
        // Legacy fallback using counts field
        engaged = currentState.counts.active || 0;
        passing = currentState.counts.passive || 0;
    } else if (currentState?.people) {
        // Last resort: use people array length
        const peopleCount = currentState.people.length;
        if (currentState.mode === 'engaged' || currentState.mode === 'crowd') {
            engaged = peopleCount;
        } else {
            passing = peopleCount;
        }
    }
    
    trendsHistory.push({
        active: engaged,
        passive: passing,
        timestamp: now
    });

    // Trim old data
    while (trendsHistory.length > TRENDS_HISTORY_SIZE) {
        trendsHistory.shift();
    }
}

function renderTrendsOverlay() {
    requestAnimationFrame(renderTrendsOverlay);
    
    // Sample new data
    sampleTrendsData();
    
    renderInputTrends();
}

function renderInputTrends() {
    if (!trendsInputCtx || !trendsInputCanvas) return;

    const rect = trendsInputCanvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    trendsInputCtx.clearRect(0, 0, width, height);

    if (trendsHistory.length < 2) return;

    let maxValue = 1;
    for (const point of trendsHistory) {
        maxValue = Math.max(maxValue, point.active + point.passive);
    }
    maxValue = Math.ceil(maxValue * 1.2) || 1;

    const padding = { left: 26, right: 8, top: 8, bottom: 16 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    const pointSpacing = chartWidth / (TRENDS_HISTORY_SIZE - 1);

    trendsInputCtx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    trendsInputCtx.lineWidth = 1;
    for (let i = 0; i <= 2; i++) {
        const y = padding.top + (chartHeight * i / 2);
        trendsInputCtx.beginPath();
        trendsInputCtx.moveTo(padding.left, y);
        trendsInputCtx.lineTo(width - padding.right, y);
        trendsInputCtx.stroke();
    }

    trendsInputCtx.fillStyle = 'rgba(255, 255, 255, 0.35)';
    trendsInputCtx.font = '8px Space Grotesk, sans-serif';
    trendsInputCtx.textAlign = 'right';
    trendsInputCtx.textBaseline = 'middle';
    trendsInputCtx.fillText(maxValue.toString(), padding.left - 4, padding.top + 6);

    trendsInputCtx.textAlign = 'left';
    trendsInputCtx.textBaseline = 'bottom';
    trendsInputCtx.fillText('60s ago', padding.left, height - 2);
    trendsInputCtx.textAlign = 'right';
    trendsInputCtx.fillText('now', width - padding.right, height - 2);

    trendsInputCtx.beginPath();
    trendsInputCtx.moveTo(padding.left, height - padding.bottom);
    for (let i = 0; i < trendsHistory.length; i++) {
        const x = padding.left + (i * pointSpacing);
        const passiveHeight = (trendsHistory[i].passive / maxValue) * chartHeight;
        const y = height - padding.bottom - passiveHeight;
        trendsInputCtx.lineTo(x, y);
    }
    const lastX = padding.left + ((trendsHistory.length - 1) * pointSpacing);
    trendsInputCtx.lineTo(lastX, height - padding.bottom);
    trendsInputCtx.closePath();

    const passingGradient = trendsInputCtx.createLinearGradient(0, padding.top, 0, height - padding.bottom);
    passingGradient.addColorStop(0, 'rgba(102, 238, 136, 0.4)');
    passingGradient.addColorStop(1, 'rgba(102, 238, 136, 0.1)');
    trendsInputCtx.fillStyle = passingGradient;
    trendsInputCtx.fill();

    trendsInputCtx.beginPath();
    for (let i = trendsHistory.length - 1; i >= 0; i--) {
        const x = padding.left + (i * pointSpacing);
        const passiveHeight = (trendsHistory[i].passive / maxValue) * chartHeight;
        const y = height - padding.bottom - passiveHeight;
        if (i === trendsHistory.length - 1) {
            trendsInputCtx.moveTo(x, y);
        } else {
            trendsInputCtx.lineTo(x, y);
        }
    }
    for (let i = 0; i < trendsHistory.length; i++) {
        const x = padding.left + (i * pointSpacing);
        const totalHeight = ((trendsHistory[i].passive + trendsHistory[i].active) / maxValue) * chartHeight;
        const y = height - padding.bottom - totalHeight;
        trendsInputCtx.lineTo(x, y);
    }
    trendsInputCtx.closePath();

    const engagedGradient = trendsInputCtx.createLinearGradient(0, padding.top, 0, height - padding.bottom);
    engagedGradient.addColorStop(0, 'rgba(255, 170, 68, 0.5)');
    engagedGradient.addColorStop(1, 'rgba(255, 170, 68, 0.15)');
    trendsInputCtx.fillStyle = engagedGradient;
    trendsInputCtx.fill();

    trendsInputCtx.beginPath();
    for (let i = 0; i < trendsHistory.length; i++) {
        const x = padding.left + (i * pointSpacing);
        const totalHeight = ((trendsHistory[i].passive + trendsHistory[i].active) / maxValue) * chartHeight;
        const y = height - padding.bottom - totalHeight;
        if (i === 0) {
            trendsInputCtx.moveTo(x, y);
        } else {
            trendsInputCtx.lineTo(x, y);
        }
    }
    trendsInputCtx.strokeStyle = 'rgba(255, 170, 68, 0.85)';
    trendsInputCtx.lineWidth = 1.5;
    trendsInputCtx.stroke();

    trendsInputCtx.beginPath();
    for (let i = 0; i < trendsHistory.length; i++) {
        const x = padding.left + (i * pointSpacing);
        const passiveHeight = (trendsHistory[i].passive / maxValue) * chartHeight;
        const y = height - padding.bottom - passiveHeight;
        if (i === 0) {
            trendsInputCtx.moveTo(x, y);
        } else {
            trendsInputCtx.lineTo(x, y);
        }
    }
    trendsInputCtx.strokeStyle = 'rgba(102, 238, 136, 0.85)';
    trendsInputCtx.lineWidth = 1.5;
    trendsInputCtx.stroke();
}


// =============================================================================
// TRENDS DISPLAY
// =============================================================================

function updateTrendsDisplay(report, realtime) {
    // Update realtime section
    updateRealtimeSection(realtime);
    
    // Update daily section
    updateDailySection(report);
}

function updateRealtimeSection(realtime) {
    // Check if realtime elements exist (they may be commented out in HTML)
    const periodEl = document.getElementById('stat-period');
    if (!periodEl) return; // Elements not in DOM, skip update
    
    if (!realtime) {
        // No realtime data - show placeholders
        periodEl.textContent = '--';
        const els = ['stat-1m-active', 'stat-1m-passive', 'stat-5m-active', 'stat-5m-passive', 
                     'stat-15m-active', 'stat-15m-passive', 'stat-60m-active', 'stat-60m-passive'];
        els.forEach(id => { const el = document.getElementById(id); if (el) el.textContent = '-'; });
        return;
    }
    
    // Period
    const period = realtime.period || 'unknown';
    const periodDisplay = period.replace('_', ' ').toUpperCase();
    periodEl.textContent = periodDisplay;
    
    // Helper to safely set text content
    const setText = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
    
    // 1 minute (recent)
    if (realtime.recent?.available) {
        setText('stat-1m-active', realtime.recent.active || 0);
        setText('stat-1m-passive', realtime.recent.passive || 0);
    } else {
        setText('stat-1m-active', '-');
        setText('stat-1m-passive', '-');
    }
    
    // 5 minute (short)
    if (realtime.short?.available) {
        setText('stat-5m-active', realtime.short.active || 0);
        setText('stat-5m-passive', realtime.short.passive || 0);
    } else {
        setText('stat-5m-active', '-');
        setText('stat-5m-passive', '-');
    }
    
    // 15 minute (medium)
    if (realtime.medium?.available) {
        setText('stat-15m-active', realtime.medium.active || 0);
        setText('stat-15m-passive', realtime.medium.passive || 0);
    } else {
        setText('stat-15m-active', '-');
        setText('stat-15m-passive', '-');
    }
    
    // 60 minute (long)
    if (realtime.long?.available) {
        setText('stat-60m-active', realtime.long.active || 0);
        setText('stat-60m-passive', realtime.long.passive || 0);
    } else {
        setText('stat-60m-active', '-');
        setText('stat-60m-passive', '-');
    }
}

function updateDailySection(report) {
    // Check if daily elements exist (they may be commented out in HTML)
    const totalEl = document.getElementById('stat-total');
    if (!totalEl) return; // Elements not in DOM, skip update
    
    if (!report) {
        // No report available - show placeholder
        totalEl.textContent = '--';
        const currentEl = document.getElementById('stat-current');
        const peakEl = document.getElementById('stat-peak');
        if (currentEl) currentEl.textContent = '--';
        if (peakEl) peakEl.textContent = '--';
        
        // Clear the chart
        const canvas = document.getElementById('hourly-chart');
        if (canvas) {
            const ctx = canvas.getContext('2d');
            const rect = canvas.parentElement.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);
            ctx.clearRect(0, 0, rect.width, rect.height);
            ctx.fillStyle = '#444455';
            ctx.font = '10px Space Grotesk, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Press R on server', rect.width / 2, rect.height / 2);
        }
        return;
    }
    
    // Update summary stats
    const summary = report.summary || {};
    totalEl.textContent = summary.total_unique_people || 0;
    
    // Find current hour and peak hour from hourly data
    const hourlyData = report.hourly_trends || [];
    const currentHour = new Date().getHours();
    
    // Find current hour count
    const currentHourData = hourlyData.find(h => h.hour === currentHour);
    const currentEl = document.getElementById('stat-current');
    if (currentEl) currentEl.textContent = currentHourData ? currentHourData.total_people : 0;
    
    // Use peak_times from report
    const peakTimes = report.peak_times || {};
    const peakEl = document.getElementById('stat-peak');
    if (peakEl) {
        if (peakTimes.peak_hour !== null && peakTimes.peak_hour !== undefined) {
            const peakHour = peakTimes.peak_hour;
            const label = peakHour === 0 ? '12a' : peakHour === 12 ? '12p' : peakHour < 12 ? `${peakHour}a` : `${peakHour-12}p`;
            peakEl.textContent = label;
        } else {
            peakEl.textContent = '--';
        }
    }
    
    // Draw hourly chart
    drawHourlyChart(hourlyData, currentHour);
}

function drawHourlyChart(hourlyData, currentHour) {
    const canvas = document.getElementById('hourly-chart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const rect = canvas.parentElement.getBoundingClientRect();
    
    // Set canvas size with device pixel ratio for sharp rendering
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    
    const width = rect.width;
    const height = rect.height;
    
    // Clear
    ctx.clearRect(0, 0, width, height);
    
    // Create hour map for quick lookup (24 hours) - use total_people
    const hourMap = new Map();
    hourlyData.forEach(h => hourMap.set(h.hour, h.total_people || 0));
    
    // Find max count for scaling
    const maxCount = Math.max(...hourlyData.map(h => h.total_people || 0), 1);
    
    // Bar dimensions
    const barPadding = 1;
    const barWidth = (width - barPadding * 23) / 24;
    const chartHeight = height - 16; // Leave room for labels
    
    // Draw bars for each hour
    for (let hour = 0; hour < 24; hour++) {
        const count = hourMap.get(hour) || 0;
        const barHeight = (count / maxCount) * chartHeight;
        const x = hour * (barWidth + barPadding);
        const y = chartHeight - barHeight;
        
        // Color: highlight current hour, dim past hours, accent for future
        if (hour === currentHour) {
            ctx.fillStyle = '#4a9eff';  // Accent (current)
        } else if (hour < currentHour) {
            ctx.fillStyle = count > 0 ? '#444466' : '#222233';  // Past
        } else {
            ctx.fillStyle = '#333344';  // Future (no data yet)
        }
        
        // Draw bar
        ctx.fillRect(x, y, barWidth, barHeight);
    }
    
    // Draw hour labels (every 6 hours)
    ctx.fillStyle = '#666677';
    ctx.font = '8px Space Grotesk, sans-serif';
    ctx.textAlign = 'center';
    [0, 6, 12, 18].forEach(hour => {
        const x = hour * (barWidth + barPadding) + barWidth / 2;
        const label = hour === 0 ? '12a' : hour === 12 ? '12p' : hour < 12 ? `${hour}a` : `${hour-12}p`;
        ctx.fillText(label, x, height - 2);
    });
}

// =============================================================================
// START
// =============================================================================

init();
