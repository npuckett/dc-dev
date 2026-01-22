/**
 * Drop Ceiling - Public Viewer
 * Three.js visualization of the light installation
 */

import * as THREE from 'three';

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
    
    // Trackzone dimensions
    TRACKZONE: {
        width: 475,
        depth: 205,
        height: 300,
        offset_z: 78,
        offset_y: -66,
        center_x: 120,
    },
    
    // WebSocket settings
    WS_PORT: 8765,
    RECONNECT_DELAY: 3000,
};

// =============================================================================
// STATE
// =============================================================================

let scene, camera, renderer;
let panels = [];
let lightSphere, lightGlow, falloffSphere;
let trackedPeople = {};
let wsConnection = null;
let currentState = null;

// =============================================================================
// INITIALIZATION
// =============================================================================

function init() {
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0c);
    scene.fog = new THREE.Fog(0x0a0a0c, 400, 1200);
    
    // Camera - fixed position for mobile portrait view
    camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 1, 2000);
    
    // Position camera for a good view of the panels
    // Looking from front-right, slightly above
    camera.position.set(300, 200, 350);
    camera.lookAt(120, 60, 0);
    
    // Renderer
    const canvas = document.getElementById('viewer');
    renderer = new THREE.WebGLRenderer({ 
        canvas,
        antialias: true,
        alpha: true,
    });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0x222233, 0.5);
    scene.add(ambientLight);
    
    // Build scene
    createFloor();
    createPanels();
    createPointLight();
    createTrackzone();
    
    // Events
    window.addEventListener('resize', onWindowResize);
    
    // Check for stored IP or show connect dialog
    const storedIP = localStorage.getItem('dropceiling_ip');
    if (storedIP) {
        connectWebSocket(storedIP);
    } else {
        showConnectDialog();
    }
    
    // Setup connect button
    document.getElementById('connect-btn').addEventListener('click', onConnectClick);
    document.getElementById('ip-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') onConnectClick();
    });
    
    // Start render loop
    animate();
}

function showConnectDialog() {
    document.getElementById('connect-overlay').classList.remove('hidden');
}

function hideConnectDialog() {
    document.getElementById('connect-overlay').classList.add('hidden');
}

function onConnectClick() {
    const ip = document.getElementById('ip-input').value.trim();
    if (ip) {
        localStorage.setItem('dropceiling_ip', ip);
        connectWebSocket(ip);
        hideConnectDialog();
    }
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
    floor.position.set(120, 0, (CONFIG.TRACKZONE.offset_z - 200) / 2);
    scene.add(floor);
    
    // Grid lines for depth
    const gridHelper = new THREE.GridHelper(500, 20, 0x333340, 0x222230);
    gridHelper.position.set(120, 0.1, 0);
    scene.add(gridHelper);
}

function createPanels() {
    // Create all 12 panels (4 units × 3 panels each)
    for (let unit = 0; unit < 4; unit++) {
        const unitX = unit * CONFIG.UNIT_SPACING;
        
        for (let panelNum = 1; panelNum <= 3; panelNum++) {
            const [localY, localZ] = CONFIG.PANEL_LOCAL_POSITIONS[panelNum];
            const angle = CONFIG.PANEL_ANGLES[panelNum];
            
            // Panel geometry
            const panelGeom = new THREE.PlaneGeometry(CONFIG.PANEL_SIZE, CONFIG.PANEL_SIZE);
            const panelMat = new THREE.MeshBasicMaterial({
                color: 0x333333,
                side: THREE.DoubleSide,
                transparent: true,
            });
            
            const panel = new THREE.Mesh(panelGeom, panelMat);
            panel.position.set(unitX, localY, localZ);
            panel.rotation.x = THREE.MathUtils.degToRad(-angle);
            
            // Add thin border
            const edges = new THREE.EdgesGeometry(panelGeom);
            const line = new THREE.LineSegments(
                edges,
                new THREE.LineBasicMaterial({ color: 0x444455, linewidth: 1 })
            );
            panel.add(line);
            
            scene.add(panel);
            panels.push({
                mesh: panel,
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
}

// =============================================================================
// WEBSOCKET CONNECTION
// =============================================================================

function connectWebSocket(ip) {
    const url = `ws://${ip}:${CONFIG.WS_PORT}`;
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
            console.log('WebSocket disconnected');
            // Attempt to reconnect
            setTimeout(() => connectWebSocket(ip), CONFIG.RECONNECT_DELAY);
        };
        
        wsConnection.onerror = (error) => {
            updateStatus('error', 'Connection Error');
            console.error('WebSocket error:', error);
        };
        
    } catch (e) {
        updateStatus('error', 'Failed to connect');
        console.error('WebSocket connection failed:', e);
        // Show connect dialog again
        setTimeout(showConnectDialog, 2000);
    }
}

function updateStatus(state, text) {
    const statusEl = document.getElementById('status-text');
    statusEl.textContent = text;
    statusEl.className = state;
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
    if (data.panels) {
        data.panels.forEach((brightness, index) => {
            if (panels[index]) {
                const normalizedBrightness = brightness / 50; // DMX max ~50
                const gray = 0.15 + normalizedBrightness * 0.85;
                panels[index].mesh.material.color.setRGB(gray, gray, gray * 0.95);
                panels[index].brightness = normalizedBrightness;
            }
        });
    }
    
    // Update tracked people
    if (data.people) {
        updateTrackedPeople(data.people);
    }
    
    // Update mode display
    if (data.mode) {
        updateModeDisplay(data.mode, data.status_text);
    }
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
        trackedPeople[person.id].position.set(person.x, person.y + 85, person.z);
    });
}

function createPersonMesh() {
    const group = new THREE.Group();
    
    // Simple cylinder for body
    const bodyGeom = new THREE.CylinderGeometry(15, 15, 150, 12);
    const bodyMat = new THREE.MeshBasicMaterial({ 
        color: 0x44aa66,
        transparent: true,
        opacity: 0.6,
    });
    const body = new THREE.Mesh(bodyGeom, bodyMat);
    group.add(body);
    
    // Sphere for head
    const headGeom = new THREE.SphereGeometry(15, 12, 12);
    const head = new THREE.Mesh(headGeom, bodyMat);
    head.position.y = 85;
    group.add(head);
    
    return group;
}

function updateModeDisplay(mode, statusText) {
    const modeLabel = document.getElementById('mode-label');
    modeLabel.textContent = mode.toUpperCase();
    modeLabel.className = `visible ${mode}`;
}

// =============================================================================
// ANIMATION
// =============================================================================

function animate() {
    requestAnimationFrame(animate);
    
    // Subtle camera sway for life
    const time = Date.now() * 0.0001;
    camera.position.x = 300 + Math.sin(time) * 5;
    camera.position.y = 200 + Math.sin(time * 0.7) * 3;
    camera.lookAt(120, 60, 0);
    
    // Pulse the light glow slightly
    if (lightGlow && currentState?.light) {
        const pulse = Math.sin(Date.now() * 0.003) * 0.1 + 1;
        lightGlow.scale.setScalar(pulse);
    }
    
    renderer.render(scene, camera);
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

// =============================================================================
// START
// =============================================================================

init();
