class State {
    constructor() {
        this.listeners = [];
        this.data = {
            dataset: 'hsnet',
            series: 'topology',
            dimension: '0',
            metric: 'bottleneck',
            activeTimestep: null,
            hoverTimestep: null,
            hoverNodeId: null,      // For highlighting node across graphs and mappers
            hoverGraphNodeId: null, // Same
            hoverEdgeId: null,      // For highlighting edge across graphs
            rawTimelineData: null,
            explTimelineData: null,
            timelineDomain: null,
            timelineZeroReference: null,
            rawMapperData: null,
            explMapperData: null,
            prevGraphData: null,
            currGraphData: null,
            nextGraphData: null,
            graphTransform: null
        };
    }

    subscribe(listener) {
        this.listeners.push(listener);
        return () => {
            this.listeners = this.listeners.filter(l => l !== listener);
        };
    }

    get() {
        return this.data;
    }

    update(updates) {
        const oldData = { ...this.data };
        this.data = { ...this.data, ...updates };

        const changedKeys = Object.keys(updates).filter(key => oldData[key] !== this.data[key]);

        if (changedKeys.length > 0) {
            this.listeners.forEach(listener => listener(this.data, changedKeys, oldData));
        }
    }
}

export const state = new State();
