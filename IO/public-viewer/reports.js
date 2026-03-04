// =============================================================================
// Drop Ceiling — Report Viewer
// =============================================================================

const REPORTS_BASE = '../reports/daily';
const HOUR_LABELS = Array.from({ length: 24 }, (_, i) =>
    i === 0 ? '12a' : i === 12 ? '12p' : i < 12 ? `${i}a` : `${i - 12}p`
);

const DAY_NAMES = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const DAY_NAMES_SHORT = ['S', 'M', 'T', 'W', 'T', 'F', 'S'];

// Design tokens (match public-viewer/style.css)
const COLORS = {
    accent: '#4a9eff',
    engaged: '#ffaa44',
    passing: '#66ee88',
    outputAvg: '#7cd3ff',
    outputBand: 'rgba(124, 211, 255, 0.25)',
    flowRTL: 'rgba(255, 140, 100, 0.85)',
    flowRTLBg: 'rgba(255, 140, 100, 0.15)',
    flowLTR: '#7cd3ff',
    flowLTRBg: 'rgba(124, 211, 255, 0.15)',
    muted: '#666666',
    gridLine: 'rgba(255, 255, 255, 0.06)',
    textFaint: 'rgba(255, 255, 255, 0.35)',
    panelBg: 'rgba(20, 20, 25, 0.9)',
    bg: '#0a0a0c',
    white: '#ffffff',

    // Mode colors
    idle: '#556688',
    flow: '#66ee88',
    engaged_mode: '#ffaa44',
    crowd: '#ff6b6b',
    pulse: '#c084fc',
    ambient: '#7cd3ff',
    unknown: '#444466',

    // Personality param colors (gradient spread)
    personality: [
        'rgba(100, 200, 255, 0.8)',   // responsiveness
        'rgba(80, 220, 240, 0.8)',    // energy
        'rgba(70, 230, 220, 0.8)',    // attention_span
        'rgba(80, 240, 200, 0.8)',    // sociability
        'rgba(100, 255, 180, 0.8)',   // exploration
        'rgba(130, 240, 160, 0.8)',   // memory
        'rgba(255, 200, 120, 0.7)',   // brightness_global
        'rgba(255, 180, 100, 0.7)',   // speed_global
        'rgba(255, 160, 100, 0.7)',   // pulse_global
        'rgba(255, 140, 100, 0.7)',   // follow_speed_global
        'rgba(255, 220, 130, 0.7)',   // dwell_influence
        'rgba(200, 180, 255, 0.7)',   // idle_trend_weight
    ],
};

// Personality parameter display names
const PARAM_LABELS = {
    responsiveness: 'Responsive',
    energy: 'Energy',
    attention_span: 'Attention',
    sociability: 'Social',
    exploration: 'Explore',
    memory: 'Memory',
    brightness_global: 'Brightness',
    speed_global: 'Speed',
    pulse_global: 'Pulse',
    follow_speed_global: 'Follow Spd',
    dwell_influence: 'Dwell',
    idle_trend_weight: 'Idle Wt',
};

// =============================================================================
// STATE
// =============================================================================

let reportIndex = null;
let reportCache = new Map();
let selectedDate = null;
let currentView = 'day'; // 'day' or 'multi'
let dayCharts = {};
let multiCharts = {};

// =============================================================================
// CHART.JS GLOBAL DEFAULTS
// =============================================================================

Chart.defaults.font.family = "'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif";
Chart.defaults.font.size = 10;
Chart.defaults.color = COLORS.muted;
Chart.defaults.plugins.legend.display = false;
Chart.defaults.plugins.tooltip.backgroundColor = COLORS.panelBg;
Chart.defaults.plugins.tooltip.borderColor = 'rgba(255, 255, 255, 0.1)';
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.cornerRadius = 8;
Chart.defaults.plugins.tooltip.titleFont = { family: "'Space Grotesk', sans-serif", size: 11, weight: '500' };
Chart.defaults.plugins.tooltip.bodyFont = { family: "'Space Grotesk', sans-serif", size: 10 };
Chart.defaults.plugins.tooltip.padding = 10;
Chart.defaults.plugins.tooltip.displayColors = true;
Chart.defaults.plugins.tooltip.boxWidth = 8;
Chart.defaults.plugins.tooltip.boxHeight = 8;
Chart.defaults.plugins.tooltip.boxPadding = 4;
Chart.defaults.elements.point.radius = 0;
Chart.defaults.elements.point.hoverRadius = 4;
Chart.defaults.elements.line.tension = 0.3;
Chart.defaults.elements.line.borderWidth = 1.5;
Chart.defaults.animation = false;
Chart.defaults.responsive = true;
Chart.defaults.maintainAspectRatio = false;

// Common scale config
const GRID_CONFIG = {
    color: COLORS.gridLine,
    drawBorder: false,
    tickLength: 0,
};

const TICK_CONFIG = {
    color: COLORS.textFaint,
    font: { size: 8, family: "'Space Grotesk', sans-serif" },
    padding: 6,
};

// =============================================================================
// INIT
// =============================================================================

async function init() {
    showLoading(true);
    try {
        reportIndex = await fetchJSON(`${REPORTS_BASE}/_index.json`);
        // Filter to only include reports from first reliable day onward
        const MIN_DATE = '2026-02-13';
        reportIndex.reports = reportIndex.reports.filter(r => r.date >= MIN_DATE);
        buildDateNav();
        setupEventListeners();
        // Default to the most recent available date
        const availableDates = reportIndex.reports.map(r => r.date);
        const initialDate = availableDates[availableDates.length - 1];
        await selectDate(initialDate);
    } catch (err) {
        console.error('Failed to load report index:', err);
    }
    showLoading(false);
}

// =============================================================================
// DATA FETCHING
// =============================================================================

async function fetchJSON(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${url}`);
    return resp.json();
}

async function loadReport(date) {
    if (reportCache.has(date)) return reportCache.get(date);
    const report = await fetchJSON(`${REPORTS_BASE}/${date}.json`);
    reportCache.set(date, report);
    return report;
}

async function loadAllReports() {
    const promises = reportIndex.reports.map(r => loadReport(r.date));
    return Promise.all(promises);
}

// =============================================================================
// DATE NAVIGATION
// =============================================================================

function buildDateNav() {
    const container = document.getElementById('date-pills');
    container.innerHTML = '';

    let lastWeek = null;
    reportIndex.reports.forEach(r => {
        const d = new Date(r.date + 'T12:00:00');
        const week = getISOWeek(d);

        // Week separator
        if (lastWeek !== null && week !== lastWeek) {
            const sep = document.createElement('div');
            sep.className = 'week-sep';
            container.appendChild(sep);
        }
        lastWeek = week;

        const btn = document.createElement('button');
        btn.className = 'date-pill';
        const dayOfWeek = d.getDay();
        if (dayOfWeek === 0 || dayOfWeek === 6) btn.classList.add('weekend');

        // Show day name + short date
        const month = d.getMonth() + 1;
        const day = d.getDate();
        btn.textContent = `${DAY_NAMES_SHORT[dayOfWeek]} ${month}/${day}`;
        btn.dataset.date = r.date;
        btn.addEventListener('click', () => selectDate(r.date));
        container.appendChild(btn);
    });
}

function getISOWeek(d) {
    const date = new Date(d.getTime());
    date.setHours(0, 0, 0, 0);
    date.setDate(date.getDate() + 3 - (date.getDay() + 6) % 7);
    const week1 = new Date(date.getFullYear(), 0, 4);
    return 1 + Math.round(((date.getTime() - week1.getTime()) / 86400000 - 3 + (week1.getDay() + 6) % 7) / 7);
}

function updateDatePills() {
    document.querySelectorAll('.date-pill').forEach(pill => {
        pill.classList.toggle('active', pill.dataset.date === selectedDate);
    });
    // Scroll active pill into view
    const active = document.querySelector('.date-pill.active');
    if (active) active.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
}

async function selectDate(date) {
    selectedDate = date;
    updateDatePills();

    // Always switch to day view when a date pill is clicked
    if (currentView !== 'day') {
        currentView = 'day';
        document.querySelectorAll('.toggle-btn').forEach(b => {
            b.classList.toggle('active', b.dataset.view === 'day');
        });
        document.getElementById('day-view').classList.remove('hidden');
        document.getElementById('multi-view').classList.add('hidden');
    }

    const report = await loadReport(date);
    renderDayView(report);
}

function navigateDate(delta) {
    const dates = reportIndex.reports.map(r => r.date);
    const idx = dates.indexOf(selectedDate);
    const newIdx = Math.max(0, Math.min(dates.length - 1, idx + delta));
    selectDate(dates[newIdx]);
}

// =============================================================================
// EVENT LISTENERS
// =============================================================================

function setupEventListeners() {
    // View toggle
    document.querySelectorAll('.toggle-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentView = btn.dataset.view;

            document.getElementById('day-view').classList.toggle('hidden', currentView !== 'day');
            document.getElementById('multi-view').classList.toggle('hidden', currentView !== 'multi');

            if (currentView === 'day') {
                const report = await loadReport(selectedDate);
                renderDayView(report);
            } else {
                await renderMultiView();
            }
        });
    });

    // Date arrows
    document.getElementById('date-prev').addEventListener('click', () => navigateDate(-1));
    document.getElementById('date-next').addEventListener('click', () => navigateDate(1));

    // Keyboard
    document.addEventListener('keydown', e => {
        if (e.key === 'ArrowLeft') navigateDate(-1);
        if (e.key === 'ArrowRight') navigateDate(1);
    });

    // Export page
    document.getElementById('export-page-btn').addEventListener('click', exportPage);

    // Per-chart export
    document.querySelectorAll('.export-chart-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const panel = btn.closest('.chart-panel');
            const canvas = panel.querySelector('canvas');
            const label = panel.querySelector('.panel-label')?.textContent || 'chart';
            if (canvas) exportChart(canvas, label);
        });
    });
}

// =============================================================================
// DAY VIEW RENDERING
// =============================================================================

function renderDayView(report) {
    renderSummaryCards(report);
    renderHourlyTraffic(report);
    renderHourlyPeople(report);
    renderFlowDirection(report);
    renderBrightness(report);
    renderBehaviorModes(report);
    renderAutoTuning(report);
}

function renderSummaryCards(report) {
    const s = report.summary || {};
    const pt = report.peak_times || {};

    document.getElementById('card-people').textContent = formatNum(s.total_unique_people);
    document.getElementById('card-active').textContent = formatNum(s.total_active_zone_visits);
    document.getElementById('card-passive').textContent = formatNum(s.total_passive_zone_count);

    // Peak hour
    if (pt.peak_hour != null) {
        document.getElementById('card-peak').textContent =
            `${HOUR_LABELS[pt.peak_hour]} (${formatNum(pt.peak_hour_count)})`;
    } else {
        document.getElementById('card-peak').textContent = '--';
    }

    // Brightness
    document.getElementById('card-brightness').textContent =
        s.avg_brightness > 0 ? s.avg_brightness.toFixed(1) : 'N/A';

    // Hours
    document.getElementById('card-hours').textContent =
        s.hours_with_data != null ? `${s.hours_with_data}/24` : '--';
}

// ------------------------------------
// Hourly Traffic — Stacked Area
// ------------------------------------
function renderHourlyTraffic(report) {
    const hourly = padHourly(report.hourly_trends || []);
    const active = hourly.map(h => h.active_count);
    const passive = hourly.map(h => h.passive_count);

    dayCharts.traffic = createOrUpdate(dayCharts.traffic, 'chart-hourly-traffic', {
        type: 'line',
        data: {
            labels: HOUR_LABELS,
            datasets: [
                {
                    label: 'Active',
                    data: active,
                    borderColor: COLORS.engaged,
                    backgroundColor: hexToAlpha(COLORS.engaged, 0.15),
                    fill: true,
                    order: 1,
                },
                {
                    label: 'Passive',
                    data: passive,
                    borderColor: COLORS.passing,
                    backgroundColor: hexToAlpha(COLORS.passing, 0.1),
                    fill: true,
                    order: 2,
                },
            ],
        },
        options: {
            scales: {
                x: { grid: GRID_CONFIG, ticks: { ...TICK_CONFIG, maxRotation: 0, autoSkip: true, maxTicksLimit: 12 } },
                y: { grid: GRID_CONFIG, ticks: TICK_CONFIG, beginAtZero: true },
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        title: (items) => `${HOUR_LABELS[items[0].dataIndex]} — Hour ${items[0].dataIndex}`,
                    },
                },
            },
        },
    });
}

// ------------------------------------
// Hourly People — Bar Chart
// ------------------------------------
function renderHourlyPeople(report) {
    const hourly = padHourly(report.hourly_trends || []);
    const people = hourly.map(h => h.total_people);

    dayCharts.people = createOrUpdate(dayCharts.people, 'chart-hourly-people', {
        type: 'bar',
        data: {
            labels: HOUR_LABELS,
            datasets: [{
                label: 'People',
                data: people,
                backgroundColor: people.map((v, i) => {
                    const max = Math.max(...people);
                    if (v === max && v > 0) return COLORS.accent;
                    if (v > 0) return 'rgba(74, 158, 255, 0.4)';
                    return 'rgba(34, 34, 51, 0.5)';
                }),
                borderRadius: 2,
                borderSkipped: false,
            }],
        },
        options: {
            scales: {
                x: { grid: { display: false }, ticks: { ...TICK_CONFIG, maxRotation: 0, autoSkip: true, maxTicksLimit: 12 } },
                y: { grid: GRID_CONFIG, ticks: TICK_CONFIG, beginAtZero: true },
            },
        },
    });
}

// ------------------------------------
// Flow Direction — Diverging Bar
// ------------------------------------
function renderFlowDirection(report) {
    const hourly = padHourly(report.hourly_trends || []);
    const ltr = hourly.map(h => h.flow_ltr);
    const rtl = hourly.map(h => -h.flow_rtl);

    dayCharts.flow = createOrUpdate(dayCharts.flow, 'chart-flow', {
        type: 'bar',
        data: {
            labels: HOUR_LABELS,
            datasets: [
                {
                    label: 'L→R',
                    data: ltr,
                    backgroundColor: COLORS.flowLTRBg,
                    borderColor: COLORS.flowLTR,
                    borderWidth: 1,
                    borderRadius: 2,
                    borderSkipped: false,
                },
                {
                    label: 'R→L',
                    data: rtl,
                    backgroundColor: COLORS.flowRTLBg,
                    borderColor: COLORS.flowRTL,
                    borderWidth: 1,
                    borderRadius: 2,
                    borderSkipped: false,
                },
            ],
        },
        options: {
            scales: {
                x: { grid: { display: false }, stacked: true, ticks: { ...TICK_CONFIG, maxRotation: 0, autoSkip: true, maxTicksLimit: 12 } },
                y: {
                    grid: GRID_CONFIG,
                    stacked: true,
                    ticks: {
                        ...TICK_CONFIG,
                        callback: v => Math.abs(v),
                    },
                },
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: ${formatNum(Math.abs(ctx.raw))}`,
                    },
                },
            },
        },
    });
}

// ------------------------------------
// Brightness — Line Chart
// ------------------------------------
function renderBrightness(report) {
    const hourly = padHourly(report.hourly_trends || []);
    const brightness = hourly.map(h => h.avg_brightness);
    const hasBrightness = brightness.some(b => b > 0);

    dayCharts.brightness = createOrUpdate(dayCharts.brightness, 'chart-brightness', {
        type: 'line',
        data: {
            labels: HOUR_LABELS,
            datasets: [{
                label: 'Avg Brightness',
                data: hasBrightness ? brightness : [],
                borderColor: COLORS.outputAvg,
                backgroundColor: COLORS.outputBand,
                fill: true,
                pointHoverBackgroundColor: COLORS.outputAvg,
            }],
        },
        options: {
            scales: {
                x: { grid: GRID_CONFIG, ticks: { ...TICK_CONFIG, maxRotation: 0, autoSkip: true, maxTicksLimit: 12 } },
                y: { grid: GRID_CONFIG, ticks: TICK_CONFIG, beginAtZero: true, suggestedMax: 50 },
            },
        },
    });
}

// ------------------------------------
// Behavior Modes — Donut (computed from hourly engagement data)
// ------------------------------------

/**
 * Compute mode distribution proportionally from actual visit counts.
 *
 * Instead of classifying each hour as a single mode (winner-take-all),
 * we weight by actual event counts so the donut accurately reflects
 * the ratio of engaged / flowing / idle activity throughout the day.
 *
 * Method:
 *   1. For each hour, split its events into engaged (active zone) and
 *      passing (passive zone) weighted contributions.
 *   2. Hours below a minimum traffic threshold count as "idle" time.
 *   3. The final distribution is the proportion of each category across
 *      the full 24-hour period.
 *
 * This guarantees: if a day has ANY active zone visits, engaged > 0%.
 */
function computeModeDistribution(report) {
    const hourly = report.hourly_trends || [];
    const TRAFFIC_FLOOR = 50; // minimum people to count as non-idle hour

    let engagedWeight = 0;
    let flowWeight = 0;
    let idleHours = 0;
    let activeHours = 0;

    for (const h of hourly) {
        const people = h.total_people || 0;
        const active = h.active_count || 0;
        const passive = h.passive_count || 0;

        if (people < TRAFFIC_FLOOR && active === 0) {
            idleHours++;
            continue;
        }

        activeHours++;
        engagedWeight += active;
        flowWeight += passive;
    }

    // If no meaningful traffic at all
    if (activeHours === 0) return { idle: 1.0 };

    const totalWeight = engagedWeight + flowWeight;
    const idleFrac = idleHours / 24;
    const activeFrac = 1 - idleFrac;

    const dist = {};

    // Engaged: proportion of active zone interactions within active hours
    if (engagedWeight > 0 && totalWeight > 0) {
        dist.engaged = (engagedWeight / totalWeight) * activeFrac;
    }

    // Flow: proportion of passive traffic within active hours
    if (flowWeight > 0 && totalWeight > 0) {
        dist.flow = (flowWeight / totalWeight) * activeFrac;
    }

    // Idle: fraction of quiet hours
    if (idleFrac > 0) {
        dist.idle = idleFrac;
    }

    return dist;
}

function renderBehaviorModes(report) {
    // Compute modes from actual hourly engagement data
    const dist = computeModeDistribution(report);
    const labels = Object.keys(dist);
    const values = Object.values(dist);
    const colors = labels.map(m => COLORS[m] || COLORS.unknown);
    const displayLabels = labels.map(l => l.charAt(0).toUpperCase() + l.slice(1));

    // Build the legend HTML
    const legendEl = document.getElementById('modes-legend');
    legendEl.innerHTML = '';
    labels.forEach((m, i) => {
        const pct = (values[i] * 100).toFixed(1);
        const div = document.createElement('div');
        div.className = 'legend-item';
        div.innerHTML = `<span class="legend-swatch" style="background:${colors[i]}"></span>
            <span class="legend-label">${displayLabels[i]}</span>
            <span class="legend-value">${pct}%</span>`;
        legendEl.appendChild(div);
    });

    // Dominant mode for center text
    const dominantIdx = values.indexOf(Math.max(...values));
    const dominantMode = displayLabels[dominantIdx] || '--';
    const dominantPct = values[dominantIdx] ? (values[dominantIdx] * 100).toFixed(0) + '%' : '';

    // Center text plugin (instance-scoped)
    const centerTextPlugin = {
        id: 'centerTextModes',
        afterDraw(chart) {
            const { ctx: c, chartArea: { left, right, top, bottom } } = chart;
            const cx = (left + right) / 2;
            const cy = (top + bottom) / 2;
            c.save();
            c.textAlign = 'center';
            c.textBaseline = 'middle';
            c.fillStyle = COLORS.white;
            c.font = "500 14px 'Space Grotesk', sans-serif";
            c.fillText(dominantMode, cx, cy - 8);
            c.fillStyle = COLORS.textFaint;
            c.font = "300 11px 'Space Grotesk', sans-serif";
            c.fillText(dominantPct, cx, cy + 10);
            c.restore();
        },
    };

    dayCharts.modes = createOrUpdate(dayCharts.modes, 'chart-modes', {
        type: 'doughnut',
        data: {
            labels: displayLabels,
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderColor: COLORS.bg,
                borderWidth: 2,
            }],
        },
        options: {
            cutout: '55%',
            plugins: {
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.label}: ${(ctx.raw * 100).toFixed(1)}%`,
                    },
                },
            },
        },
        plugins: [centerTextPlugin],
    });
}

// ------------------------------------
// Auto-Tuning — Horizontal Bars
// ------------------------------------
function renderAutoTuning(report) {
    const panel = document.getElementById('tuning-panel');
    const badge = document.getElementById('tuning-adjustments');
    const strategyEl = document.getElementById('tuning-strategy');

    const tuning = report.auto_tuning;
    if (!tuning || !tuning.optimal_values) {
        panel.classList.add('hidden');
        return;
    }
    panel.classList.remove('hidden');

    badge.textContent = `${formatNum(tuning.total_adjustments)} adjustments`;
    strategyEl.textContent = tuning.strategy_summary || '';

    const keys = Object.keys(tuning.optimal_values);
    const values = Object.values(tuning.optimal_values);
    const labels = keys.map(k => PARAM_LABELS[k] || k);

    dayCharts.tuning = createOrUpdate(dayCharts.tuning, 'chart-tuning', {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Value',
                data: values,
                backgroundColor: keys.map((_, i) => COLORS.personality[i % COLORS.personality.length]),
                borderRadius: 4,
                borderSkipped: false,
            }],
        },
        options: {
            indexAxis: 'y',
            scales: {
                x: {
                    grid: GRID_CONFIG,
                    ticks: TICK_CONFIG,
                    min: 0,
                    max: 1.1,
                },
                y: {
                    grid: { display: false },
                    ticks: { ...TICK_CONFIG, font: { size: 9 } },
                },
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.label}: ${ctx.raw.toFixed(3)}`,
                    },
                },
            },
        },
    });
}

// =============================================================================
// MULTI-DAY VIEW RENDERING
// =============================================================================

async function renderMultiView() {
    showLoading(true);
    const reports = await loadAllReports();
    showLoading(false);

    renderTotalsSummary(reports);
    renderMultiPeople(reports);
    renderMultiEngagement(reports);
    renderMultiFlow(reports);
    renderHeatmap(reports);
    renderMultiPeak(reports);
    renderMultiBrightness(reports);
    renderMultiPersonality(reports);
}

// ------------------------------------
// Totals Summary Cards
// ------------------------------------
function renderTotalsSummary(reports) {
    const totalPeople = reports.reduce((s, r) => s + (r.summary.total_unique_people || 0), 0);
    const totalEvents = reports.reduce((s, r) => s + (r.summary.total_events || 0), 0);
    const totalActive = reports.reduce((s, r) => s + (r.summary.total_active_zone_visits || 0), 0);
    const avgPeople = reports.length > 0 ? Math.round(totalPeople / reports.length) : 0;
    const brightVals = reports.map(r => r.summary.avg_brightness).filter(b => b && b > 0);
    const avgBrightness = brightVals.length > 0 ? (brightVals.reduce((a, b) => a + b, 0) / brightVals.length).toFixed(1) : 'N/A';

    document.getElementById('totals-people').textContent = formatNum(totalPeople);
    document.getElementById('totals-events').textContent = formatNum(totalEvents);
    document.getElementById('totals-avg').textContent = formatNum(avgPeople) + '/day';
    document.getElementById('totals-active').textContent = formatNum(totalActive);
    document.getElementById('totals-brightness').textContent = avgBrightness;
    document.getElementById('totals-days').textContent = reports.length;
}

// ------------------------------------
// Weekly Aggregation Helper
// ------------------------------------
function computeWeeklyAverages(reports, accessor) {
    // Group reports by ISO week and compute average per week
    const weeks = new Map();
    reports.forEach((r, i) => {
        const d = new Date(r.date + 'T12:00:00');
        const weekKey = getISOWeek(d);
        if (!weeks.has(weekKey)) weeks.set(weekKey, { indices: [], values: [] });
        const val = accessor(r);
        weeks.get(weekKey).indices.push(i);
        if (val != null) weeks.get(weekKey).values.push(val);
    });

    // Build a data array the same length as reports, with weekly avg at each week's midpoint
    const weeklyData = new Array(reports.length).fill(null);
    for (const [, group] of weeks) {
        if (group.values.length === 0) continue;
        const avg = group.values.reduce((a, b) => a + b, 0) / group.values.length;
        // Place the weekly average at the midpoint index of the week
        const midIdx = group.indices[Math.floor(group.indices.length / 2)];
        weeklyData[midIdx] = avg;
    }
    return weeklyData;
}

function getMultiLabels(reports) {
    return reports.map(r => {
        const d = new Date(r.date + 'T12:00:00');
        return `${DAY_NAMES_SHORT[d.getDay()]} ${d.getMonth() + 1}/${d.getDate()}`;
    });
}

// ------------------------------------
// Daily People Trend
// ------------------------------------
function renderMultiPeople(reports) {
    const labels = getMultiLabels(reports);
    const data = reports.map(r => r.summary.total_unique_people);
    const weeklyAvg = computeWeeklyAverages(reports, r => r.summary.total_unique_people);

    multiCharts.people = createOrUpdate(multiCharts.people, 'chart-multi-people', {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Unique People',
                    data,
                    borderColor: COLORS.accent,
                    backgroundColor: hexToAlpha(COLORS.accent, 0.1),
                    fill: true,
                    pointRadius: 3,
                    pointHoverRadius: 5,
                    pointBackgroundColor: COLORS.accent,
                },
                {
                    label: 'Weekly Avg',
                    data: weeklyAvg,
                    borderColor: COLORS.engaged,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    borderDash: [6, 3],
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    pointBackgroundColor: COLORS.engaged,
                    pointStyle: 'rectRot',
                    spanGaps: true,
                },
            ],
        },
        options: {
            scales: {
                x: { grid: GRID_CONFIG, ticks: { ...TICK_CONFIG, maxRotation: 45 } },
                y: { grid: GRID_CONFIG, ticks: TICK_CONFIG, beginAtZero: true },
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    align: 'end',
                    labels: {
                        color: COLORS.textFaint,
                        font: { size: 8, family: "'Space Grotesk', sans-serif" },
                        boxWidth: 10,
                        boxHeight: 2,
                        padding: 8,
                        usePointStyle: false,
                    },
                },
                tooltip: {
                    callbacks: {
                        title: (items) => reports[items[0].dataIndex].date,
                        label: (ctx) => ctx.dataset.label === 'Weekly Avg'
                            ? `Week Avg: ${formatNum(Math.round(ctx.raw))} people`
                            : `${formatNum(ctx.raw)} people`,
                    },
                },
            },
        },
    });
}

// ------------------------------------
// Engagement Rate
// ------------------------------------
function renderMultiEngagement(reports) {
    const labels = getMultiLabels(reports);
    const activeRates = reports.map(r => {
        const total = r.summary.total_events || 1;
        return (r.summary.total_active_zone_visits / total * 100);
    });
    const passiveRates = reports.map(r => {
        const total = r.summary.total_events || 1;
        return (r.summary.total_passive_zone_count / total * 100);
    });

    multiCharts.engagement = createOrUpdate(multiCharts.engagement, 'chart-multi-engagement', {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Active %',
                    data: activeRates,
                    borderColor: COLORS.engaged,
                    backgroundColor: 'transparent',
                    pointRadius: 3,
                    pointBackgroundColor: COLORS.engaged,
                },
                {
                    label: 'Passive %',
                    data: passiveRates,
                    borderColor: COLORS.passing,
                    backgroundColor: 'transparent',
                    pointRadius: 3,
                    pointBackgroundColor: COLORS.passing,
                },
            ],
        },
        options: {
            scales: {
                x: { grid: GRID_CONFIG, ticks: { ...TICK_CONFIG, maxRotation: 45 } },
                y: { grid: GRID_CONFIG, ticks: { ...TICK_CONFIG, callback: v => `${v.toFixed(0)}%` }, beginAtZero: true },
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        title: (items) => reports[items[0].dataIndex].date,
                        label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toFixed(1)}%`,
                    },
                },
            },
        },
    });
}

// ------------------------------------
// Flow Balance
// ------------------------------------
function renderMultiFlow(reports) {
    const labels = getMultiLabels(reports);
    const balance = reports.map(r => r.flow?.flow_balance || 0);

    multiCharts.flow = createOrUpdate(multiCharts.flow, 'chart-multi-flow', {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Flow Balance',
                data: balance,
                backgroundColor: balance.map(v => v >= 0 ? COLORS.flowLTRBg : COLORS.flowRTLBg),
                borderColor: balance.map(v => v >= 0 ? COLORS.flowLTR : COLORS.flowRTL),
                borderWidth: 1,
                borderRadius: 3,
                borderSkipped: false,
            }],
        },
        options: {
            scales: {
                x: { grid: { display: false }, ticks: { ...TICK_CONFIG, maxRotation: 45 } },
                y: {
                    grid: GRID_CONFIG,
                    ticks: { ...TICK_CONFIG, callback: v => v.toFixed(1) },
                    suggestedMin: -0.5,
                    suggestedMax: 0.5,
                },
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        title: (items) => reports[items[0].dataIndex].date,
                        label: (ctx) => {
                            const v = ctx.raw;
                            const dir = v > 0.05 ? 'L→R bias' : v < -0.05 ? 'R→L bias' : 'Balanced';
                            return `${v.toFixed(3)} (${dir})`;
                        },
                    },
                },
            },
        },
    });
}

// ------------------------------------
// Hourly Heatmap — Canvas drawn manually
// ------------------------------------
function renderHeatmap(reports) {
    const canvas = document.getElementById('chart-heatmap');
    const ctx = canvas.getContext('2d');
    const container = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;

    const rect = container.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';

    const width = rect.width;
    const height = rect.height;

    ctx.clearRect(0, 0, width, height);

    const margin = { top: 4, right: 10, bottom: 24, left: 56 };
    const chartW = width - margin.left - margin.right;
    const chartH = height - margin.top - margin.bottom;

    const rows = reports.length;
    const cols = 24;
    const cellW = chartW / cols;
    const cellH = Math.min(chartH / rows, 20);
    const totalH = cellH * rows;

    // Find global max
    let globalMax = 0;
    const grid = reports.map(r => {
        const hourly = padHourly(r.hourly_trends || []);
        return hourly.map(h => {
            if (h.total_people > globalMax) globalMax = h.total_people;
            return h.total_people;
        });
    });

    // Draw cells
    for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
            const val = grid[row][col];
            const intensity = globalMax > 0 ? val / globalMax : 0;
            const x = margin.left + col * cellW;
            const y = margin.top + row * cellH;

            ctx.fillStyle = heatColor(intensity);
            ctx.fillRect(x, y, cellW - 1, cellH - 1);
        }
    }

    // Y-axis labels (dates)
    ctx.fillStyle = COLORS.textFaint;
    ctx.font = "8px 'Space Grotesk', sans-serif";
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let row = 0; row < rows; row++) {
        const d = new Date(reports[row].date + 'T12:00:00');
        const label = `${DAY_NAMES_SHORT[d.getDay()]} ${d.getMonth() + 1}/${d.getDate()}`;
        const y = margin.top + row * cellH + cellH / 2;
        ctx.fillText(label, margin.left - 6, y);
    }

    // X-axis labels (hours)
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let col = 0; col < cols; col += 3) {
        const x = margin.left + col * cellW + cellW / 2;
        ctx.fillText(HOUR_LABELS[col], x, margin.top + totalH + 6);
    }
}

function heatColor(intensity) {
    if (intensity <= 0) return COLORS.bg;
    // Interpolate from dark blue to bright accent
    const r = Math.round(10 + intensity * 64);
    const g = Math.round(10 + intensity * 148);
    const b = Math.round(12 + intensity * 243);
    const a = 0.3 + intensity * 0.7;
    return `rgba(${r}, ${g}, ${b}, ${a})`;
}

// ------------------------------------
// Peak Hour Shift
// ------------------------------------
function renderMultiPeak(reports) {
    const labels = getMultiLabels(reports);
    const peaks = reports.map(r => r.peak_times?.peak_hour ?? null);

    multiCharts.peak = createOrUpdate(multiCharts.peak, 'chart-multi-peak', {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Peak Hour',
                data: peaks,
                borderColor: COLORS.accent,
                backgroundColor: COLORS.accent,
                pointRadius: 5,
                pointHoverRadius: 7,
                pointStyle: 'circle',
                showLine: true,
                borderDash: [4, 4],
            }],
        },
        options: {
            scales: {
                x: { grid: GRID_CONFIG, ticks: { ...TICK_CONFIG, maxRotation: 45 } },
                y: {
                    grid: GRID_CONFIG,
                    ticks: {
                        ...TICK_CONFIG,
                        callback: v => HOUR_LABELS[v] || v,
                        stepSize: 2,
                    },
                    min: 0,
                    max: 23,
                    reverse: false,
                },
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        title: (items) => reports[items[0].dataIndex].date,
                        label: (ctx) => `Peak at ${HOUR_LABELS[ctx.raw]} (${reports[ctx.dataIndex].peak_times?.peak_hour_count || 0} people)`,
                    },
                },
            },
        },
    });
}

// ------------------------------------
// Brightness Trend
// ------------------------------------
function renderMultiBrightness(reports) {
    const labels = getMultiLabels(reports);
    const brightness = reports.map(r => r.summary.avg_brightness || null);
    const hasBrightness = brightness.some(b => b && b > 0);
    const weeklyAvg = computeWeeklyAverages(reports, r => r.summary.avg_brightness || null);

    multiCharts.brightness = createOrUpdate(multiCharts.brightness, 'chart-multi-brightness', {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Avg Brightness',
                    data: hasBrightness ? brightness : [],
                    borderColor: COLORS.outputAvg,
                    backgroundColor: COLORS.outputBand,
                    fill: true,
                    pointRadius: 3,
                    pointBackgroundColor: COLORS.outputAvg,
                    spanGaps: true,
                },
                {
                    label: 'Weekly Avg',
                    data: weeklyAvg,
                    borderColor: COLORS.engaged,
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    borderDash: [6, 3],
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    pointBackgroundColor: COLORS.engaged,
                    pointStyle: 'rectRot',
                    spanGaps: true,
                },
            ],
        },
        options: {
            scales: {
                x: { grid: GRID_CONFIG, ticks: { ...TICK_CONFIG, maxRotation: 45 } },
                y: { grid: GRID_CONFIG, ticks: TICK_CONFIG, beginAtZero: true, suggestedMax: 40 },
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    align: 'end',
                    labels: {
                        color: COLORS.textFaint,
                        font: { size: 8, family: "'Space Grotesk', sans-serif" },
                        boxWidth: 10,
                        boxHeight: 2,
                        padding: 8,
                        usePointStyle: false,
                    },
                },
                tooltip: {
                    callbacks: {
                        title: (items) => reports[items[0].dataIndex].date,
                        label: (ctx) => ctx.raw != null ? `${ctx.dataset.label}: ${ctx.raw.toFixed(1)}` : 'N/A',
                    },
                },
            },
        },
    });
}

// ------------------------------------
// Personality Evolution — Multi-line
// ------------------------------------
function renderMultiPersonality(reports) {
    const panel = document.getElementById('personality-panel');

    // Filter reports with tuning data
    const tuningReports = reports.filter(r => r.auto_tuning?.optimal_values);
    if (tuningReports.length === 0) {
        panel.classList.add('hidden');
        return;
    }
    panel.classList.remove('hidden');

    const labels = tuningReports.map(r => {
        const d = new Date(r.date + 'T12:00:00');
        return `${DAY_NAMES_SHORT[d.getDay()]} ${d.getMonth() + 1}/${d.getDate()}`;
    });

    const paramKeys = Object.keys(tuningReports[0].auto_tuning.optimal_values);
    const datasets = paramKeys.map((key, i) => ({
        label: PARAM_LABELS[key] || key,
        data: tuningReports.map(r => r.auto_tuning.optimal_values[key] ?? null),
        borderColor: COLORS.personality[i % COLORS.personality.length],
        backgroundColor: 'transparent',
        pointRadius: 2,
        pointHoverRadius: 4,
        borderWidth: 1.5,
        spanGaps: true,
    }));

    multiCharts.personality = createOrUpdate(multiCharts.personality, 'chart-multi-personality', {
        type: 'line',
        data: { labels, datasets },
        options: {
            scales: {
                x: { grid: GRID_CONFIG, ticks: { ...TICK_CONFIG, maxRotation: 45 } },
                y: { grid: GRID_CONFIG, ticks: TICK_CONFIG, min: 0, max: 1.1 },
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom',
                    labels: {
                        color: COLORS.textFaint,
                        font: { size: 8, family: "'Space Grotesk', sans-serif" },
                        boxWidth: 8,
                        boxHeight: 8,
                        padding: 8,
                        usePointStyle: true,
                        pointStyle: 'circle',
                    },
                },
                tooltip: {
                    callbacks: {
                        title: (items) => tuningReports[items[0].dataIndex].date,
                    },
                },
            },
        },
    });
}

// =============================================================================
// CHART HELPERS
// =============================================================================

function createOrUpdate(existing, canvasId, config) {
    if (existing) existing.destroy();
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, config);
}

function padHourly(trends) {
    const map = new Map(trends.map(h => [h.hour, h]));
    return Array.from({ length: 24 }, (_, i) => map.get(i) || {
        hour: i, total_events: 0, total_people: 0, active_count: 0,
        passive_count: 0, avg_speed: 0, flow_ltr: 0, flow_rtl: 0,
        bloom_count: 0, dominant_mode: 'unknown', avg_brightness: 0,
    });
}

// =============================================================================
// EXPORT
// =============================================================================

function exportChart(canvas, title) {
    // Get the chart instance
    const chartInstance = Chart.getChart(canvas);

    // Create export canvas with title and padding
    const dpr = window.devicePixelRatio || 1;
    const sourceW = canvas.width;
    const sourceH = canvas.height;
    const padding = 40 * dpr;
    const titleH = 50 * dpr;

    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = sourceW + padding * 2;
    exportCanvas.height = sourceH + titleH + padding * 2;
    const ctx = exportCanvas.getContext('2d');

    // Background
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);

    // Title
    ctx.fillStyle = COLORS.white;
    ctx.font = `${12 * dpr}px 'Space Grotesk', sans-serif`;
    ctx.textAlign = 'left';
    ctx.letterSpacing = `${2 * dpr}px`;
    ctx.fillText(title.toUpperCase(), padding, padding + 16 * dpr);

    // Date
    ctx.fillStyle = COLORS.muted;
    ctx.font = `${10 * dpr}px 'Space Grotesk', sans-serif`;
    ctx.fillText(selectedDate, padding, padding + 32 * dpr);

    // Chart image
    if (chartInstance) {
        const img = new Image();
        img.onload = () => {
            ctx.drawImage(img, padding, titleH + padding, sourceW, sourceH);
            downloadCanvas(exportCanvas, `${title.toLowerCase().replace(/\s+/g, '-')}-${selectedDate}.png`);
        };
        img.src = chartInstance.toBase64Image();
    } else {
        // For heatmap (raw canvas, no Chart.js)
        ctx.drawImage(canvas, padding, titleH + padding);
        downloadCanvas(exportCanvas, `${title.toLowerCase().replace(/\s+/g, '-')}-${selectedDate}.png`);
    }
}

async function exportPage() {
    const btn = document.getElementById('export-page-btn');
    btn.disabled = true;
    btn.innerHTML = '<span>Exporting…</span>';

    try {
        const target = currentView === 'day'
            ? document.getElementById('day-view')
            : document.getElementById('multi-view');

        const canvas = await html2canvas(document.body, {
            backgroundColor: COLORS.bg,
            scale: 2,
            useCORS: true,
            logging: false,
            windowWidth: document.body.scrollWidth,
            windowHeight: document.body.scrollHeight,
        });

        const viewLabel = currentView === 'day' ? selectedDate : 'trends';
        downloadCanvas(canvas, `report-${viewLabel}.png`);
    } catch (err) {
        console.error('Export failed:', err);
    }

    btn.disabled = false;
    btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> Export Page`;
}

function downloadCanvas(canvas, filename) {
    const a = document.createElement('a');
    a.download = filename;
    a.href = canvas.toDataURL('image/png');
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// =============================================================================
// UTILITIES
// =============================================================================

function formatNum(n) {
    if (n == null) return '--';
    return n.toLocaleString();
}

function hexToAlpha(hex, alpha) {
    // Works with both hex (#RRGGBB) and CSS color names
    if (hex.startsWith('#')) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
    // For named/rgba colors, just return with alpha
    return hex.replace(/[\d.]+\)$/, `${alpha})`);
}

function showLoading(show) {
    document.getElementById('loading').classList.toggle('hidden', !show);
}

// =============================================================================
// START
// =============================================================================

init();
