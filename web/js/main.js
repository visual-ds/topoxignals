import { state } from './state.js?v=signed-metrics';
import { getTimeline, getGraphs, getMapper } from './api.js';
import { TimelineView } from './timeline.js?v=signed-metrics-3';
import { GraphView } from './graph.js?v=signed-metrics';
import { MapperView } from './mapper.js?v=signed-metrics';

// Initialize views
const rawTimelineView = new TimelineView('raw-timeline-container', 'raw');
const explTimelineView = new TimelineView('expl-timeline-container', 'expl');

const currGraphView = new GraphView('graph-curr-container', 'curr');
const nextGraphView = new GraphView('graph-next-container', 'next');
const currFilteredGraphView = new GraphView('graph-curr-filtered-container', 'curr-filtered');
const nextFilteredGraphView = new GraphView('graph-next-filtered-container', 'next-filtered');

// DOM Elements
const dimSelect = document.getElementById('dim-select');
const metricSelect = document.getElementById('metric-select');
const datasetSelect = document.getElementById('dataset-select');
const seriesSelect = document.getElementById('series-select');
const loadBtn = document.getElementById('load-btn');

const titleGraphCurr = document.getElementById('title-graph-curr');
const titleGraphNext = document.getElementById('title-graph-next');
const titleGraphCurrFiltered = document.getElementById('title-graph-curr-filtered');
const titleGraphNextFiltered = document.getElementById('title-graph-next-filtered');
const rawTimelineTitle = document.getElementById('raw-timeline-title');
const explTimelineTitle = document.getElementById('expl-timeline-title');

// Event Listeners for UI interaction
dimSelect.addEventListener('change', (e) => state.update({ dimension: e.target.value }));
metricSelect.addEventListener('change', (e) => state.update({ metric: e.target.value }));
datasetSelect.addEventListener('change', (e) => {
    state.update({ dataset: e.target.value });
    loadViewerData();
});
seriesSelect.addEventListener('change', (e) => {
    state.update({ series: e.target.value });
    loadViewerData();
});

let fetchingTimelines = false;
let fetchingGraphs = false;

async function loadViewerData() {
    // Reset interaction state
    state.update({
        activeTimestep: null, hoverTimestep: null,
        prevGraphData: null, currGraphData: null, nextGraphData: null
    });

    // 1. Fetch Timeline Data
    if (!fetchingTimelines) {
        fetchingTimelines = true;
        try {
            const current = state.get();
            const data = await getTimeline(current.dimension, current.metric, current.dataset, current.series);
            let rData = [];
            let eData = [];

            if (Array.isArray(data)) {
                rData = data.map(d => ({ timestep: d.timestep, value: d.raw ?? d.value0 ?? Object.values(d)[1] }));
                eData = data.map(d => ({ timestep: d.timestep, value: d.exp ?? d.value1 ?? Object.values(d)[2] }));
            } else if (data.timeline_0 && data.timeline_1) {
                rData = data.timeline_0;
                eData = data.timeline_1;
            } else if (data.raw && data.exp) {
                rData = data.raw;
                eData = data.exp;
            }

            state.update({
                rawTimelineData: rData,
                explTimelineData: eData,
                timelineDomain: data.domain ?? null,
                timelineZeroReference: data.zero_reference ?? null,
                activeTimestep: data.initial_timestep ?? rData.find(point => point.value >= 0)?.timestep ?? null
            });
            const suffix = current.series === 'topology'
                ? ` — H${current.dimension}, ${current.metric}`
                : '';
            rawTimelineTitle.textContent = `${data.labels?.raw ?? "Topological Consistency — TC(t)"}${suffix}`;
            explTimelineTitle.textContent = `${data.labels?.exp ?? "Topological Stability — TS(t)"}${suffix}`;
        } catch (error) {
            console.error("Error fetching timelines:", error);
        } finally {
            fetchingTimelines = false;
        }
    }
}

loadBtn.addEventListener('click', loadViewerData);

// React to state changes
state.subscribe(async (current, changedKeys) => {
    // Render timelines
    if (changedKeys.includes('rawTimelineData') || changedKeys.includes('explTimelineData') ||
        changedKeys.includes('hoverTimestep') || changedKeys.includes('activeTimestep') ||
        changedKeys.includes('hoverNodeId') || changedKeys.includes('hoverNodeElements')) {
        rawTimelineView.render();
        explTimelineView.render();
    }

    // Fetch graphs if a timestep is selected
    if (changedKeys.includes('activeTimestep') && current.activeTimestep !== null && !fetchingGraphs) {
        fetchingGraphs = true;
        try {
            const graphPayload = await getGraphs(current.activeTimestep, current.dataset);
            
            titleGraphCurr.textContent = `Timestep: ${current.activeTimestep}`;
            titleGraphNext.textContent = graphPayload.next ? `Timestep: ${current.activeTimestep + 1}` : "No next snapshot";
            titleGraphCurrFiltered.textContent = `Timestep: ${current.activeTimestep} (Explanation)`;
            titleGraphNextFiltered.textContent = graphPayload.next ? `Timestep: ${current.activeTimestep + 1} (Explanation)` : "No next snapshot";

            state.update({
                prevGraphData: graphPayload.prev ?? graphPayload['t-1'] ?? graphPayload[0] ?? null,
                currGraphData: graphPayload.curr ?? graphPayload['t'] ?? graphPayload.graph_0 ?? graphPayload[1] ?? null,
                nextGraphData: graphPayload.next ?? graphPayload['t+1'] ?? graphPayload[2] ?? null
            });
        } catch (err) {
            console.error("Error fetching graphs:", err);
            titleGraphCurr.textContent = "Error loading";
            state.update({ prevGraphData: null, currGraphData: null, nextGraphData: null });
        } finally {
            fetchingGraphs = false;
        }
    }

    // Render graphs
    if (changedKeys.includes('currGraphData') ||
        changedKeys.includes('nextGraphData') || changedKeys.includes('hoverGraphNodeId') ||
        changedKeys.includes('hoverEdgeId') || changedKeys.includes('graphTransform')) {
        currGraphView.render();
        nextGraphView.render();
        currFilteredGraphView.render();
        nextFilteredGraphView.render();
    }
});

console.log("App Started");
loadViewerData();
