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
let trackedPeople = {};
let wsConnection = null;
let currentState = null;
let lastReportVersion = -1;  // Track report version to avoid redundant updates
let latestDailyReport = null;  // Store latest report for when panel opens
let latestRealtimeTrends = null;  // Store realtime trends

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
    
    // Position camera for a good view of panels and tracking area
    // Looking from front-right side, further back to see people
    // Units are at negative X (-30 to -270), so position camera to see that area
    camera.position.set(200, 250, 400);
    camera.lookAt(-150, 80, 50);
    
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
    controls.target.set(-150, 60, 50);  // Look at center of units (negative X)
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 100;
    controls.maxDistance = 1500;
    controls.update();
    
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
    
    // Trends panel toggle
    const trendsPanel = document.getElementById('trends-panel');
    const trendsBtn = document.getElementById('trends-btn');
    const trendsHeader = document.getElementById('trends-header');
    
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
    
    trendsHeader.addEventListener('click', () => {
        trendsPanel.classList.toggle('collapsed');
    });
    
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
    
    // Wander box wireframe (yellow) - where the light can wander
    const wb = CONFIG.WANDER_BOX;
    const wbWidth = wb.max_x - wb.min_x;
    const wbHeight = wb.max_y - wb.min_y;
    const wbDepth = wb.max_z - wb.min_z;
    const wbGeom = new THREE.BoxGeometry(wbWidth, wbHeight, wbDepth);
    const wbEdges = new THREE.EdgesGeometry(wbGeom);
    const wbLine = new THREE.LineSegments(
        wbEdges,
        new THREE.LineBasicMaterial({ color: 0xffff00, transparent: true, opacity: 0.4 })
    );
    // Position at center of wander box
    wbLine.position.set(
        (wb.min_x + wb.max_x) / 2,
        (wb.min_y + wb.max_y) / 2,
        (wb.min_z + wb.max_z) / 2
    );
    scene.add(wbLine);
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
    // DMX values from Python are in range 1-50 (DMX_MIN to DMX_MAX)
    // Python sends units in reverse order (unit 3,2,1,0), so reverse to match viewer order
    if (data.panels) {
        const reversedPanels = [...data.panels].reverse();
        reversedPanels.forEach((dmxValue, index) => {
            if (panels[index]) {
                const normalizedBrightness = (dmxValue - 1) / 49; // Map 1-50 to 0-1
                // Apply exponential curve for more dramatic effect (brights pop more)
                const curved = Math.pow(normalizedBrightness, 0.6);
                // Very dark minimum (0.03) to over-bright (1.2) with warm tint
                const intensity = 0.03 + curved * 1.17;
                panels[index].mesh.material.color.setRGB(
                    Math.min(intensity, 1.0), 
                    Math.min(intensity * 0.95, 1.0), 
                    Math.min(intensity * 0.85, 1.0)
                );
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
        updateModeDisplay(data.mode, data.status);
    }
    
    // Update behavior status text
    if (data.status !== undefined) {
        document.getElementById('behavior-status').textContent = data.status || '';
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
    
    // Update orbit controls
    controls.update();
    
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
// TRENDS DISPLAY
// =============================================================================

function updateTrendsDisplay(report, realtime) {
    // Update realtime section
    updateRealtimeSection(realtime);
    
    // Update daily section
    updateDailySection(report);
}

function updateRealtimeSection(realtime) {
    if (!realtime) {
        // No realtime data - show placeholders
        document.getElementById('stat-period').textContent = '--';
        document.getElementById('stat-1m-active').textContent = '-';
        document.getElementById('stat-1m-passive').textContent = '-';
        document.getElementById('stat-5m-active').textContent = '-';
        document.getElementById('stat-5m-passive').textContent = '-';
        document.getElementById('stat-15m-active').textContent = '-';
        document.getElementById('stat-15m-passive').textContent = '-';
        document.getElementById('stat-60m-active').textContent = '-';
        document.getElementById('stat-60m-passive').textContent = '-';
        return;
    }
    
    // Period
    const period = realtime.period || 'unknown';
    const periodDisplay = period.replace('_', ' ').toUpperCase();
    document.getElementById('stat-period').textContent = periodDisplay;
    
    // 1 minute (recent)
    if (realtime.recent?.available) {
        document.getElementById('stat-1m-active').textContent = realtime.recent.active || 0;
        document.getElementById('stat-1m-passive').textContent = realtime.recent.passive || 0;
    } else {
        document.getElementById('stat-1m-active').textContent = '-';
        document.getElementById('stat-1m-passive').textContent = '-';
    }
    
    // 5 minute (short)
    if (realtime.short?.available) {
        document.getElementById('stat-5m-active').textContent = realtime.short.active || 0;
        document.getElementById('stat-5m-passive').textContent = realtime.short.passive || 0;
    } else {
        document.getElementById('stat-5m-active').textContent = '-';
        document.getElementById('stat-5m-passive').textContent = '-';
    }
    
    // 15 minute (medium)
    if (realtime.medium?.available) {
        document.getElementById('stat-15m-active').textContent = realtime.medium.active || 0;
        document.getElementById('stat-15m-passive').textContent = realtime.medium.passive || 0;
    } else {
        document.getElementById('stat-15m-active').textContent = '-';
        document.getElementById('stat-15m-passive').textContent = '-';
    }
    
    // 60 minute (long)
    if (realtime.long?.available) {
        document.getElementById('stat-60m-active').textContent = realtime.long.active || 0;
        document.getElementById('stat-60m-passive').textContent = realtime.long.passive || 0;
    } else {
        document.getElementById('stat-60m-active').textContent = '-';
        document.getElementById('stat-60m-passive').textContent = '-';
    }
}

function updateDailySection(report) {
    if (!report) {
        // No report available - show placeholder
        document.getElementById('stat-total').textContent = '--';
        document.getElementById('stat-current').textContent = '--';
        document.getElementById('stat-peak').textContent = '--';
        
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
    document.getElementById('stat-total').textContent = summary.total_unique_people || 0;
    
    // Find current hour and peak hour from hourly data
    const hourlyData = report.hourly_trends || [];
    const currentHour = new Date().getHours();
    
    // Find current hour count
    const currentHourData = hourlyData.find(h => h.hour === currentHour);
    document.getElementById('stat-current').textContent = currentHourData ? currentHourData.total_people : 0;
    
    // Use peak_times from report
    const peakTimes = report.peak_times || {};
    if (peakTimes.peak_hour !== null && peakTimes.peak_hour !== undefined) {
        const peakHour = peakTimes.peak_hour;
        const label = peakHour === 0 ? '12a' : peakHour === 12 ? '12p' : peakHour < 12 ? `${peakHour}a` : `${peakHour-12}p`;
        document.getElementById('stat-peak').textContent = label;
    } else {
        document.getElementById('stat-peak').textContent = '--';
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
