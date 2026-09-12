import { state } from './state.js?v=signed-metrics';

const tooltip = d3.select("#d3-tooltip");

export class TimelineView {
    constructor(containerId, type) {
        this.containerId = containerId;
        this.type = type; // 'raw' or 'expl'

        this.margin = { top: 20, right: 30, bottom: 30, left: 50 };
        this.initDOM();

        window.addEventListener('resize', () => {
            clearTimeout(this.resizeTimer);
            this.resizeTimer = setTimeout(() => this.render(), 100);
        });
    }

    initDOM() {
        const container = d3.select(`#${this.containerId}`);
        container.selectAll("*").remove();

        this.svg = container.append("svg")
            .style("width", "100%")
            .style("height", "100%")
            .style("overflow", "visible"); // Changed to visible for axes

        this.g = this.svg.append("g");

        this.overlay = this.g.append("rect")
            .attr("fill", "transparent")
            .on("mousemove", (event) => this.handleMouseMove(event))
            .on("mouseout", () => this.handleMouseOut());

        this.xAxisGroup = this.g.append("g").attr("class", "x-axis");
        this.yAxisGroup = this.g.append("g").attr("class", "y-axis");

        // Axis Titles
        this.xAxisTitle = this.g.append("text")
            .attr("text-anchor", "middle")
            .attr("font-size", "10px")
            .attr("fill", "#64748b")
            .text("Time");

        this.yAxisTitle = this.svg.append("text")
            .attr("text-anchor", "middle")
            .attr("font-size", "10px")
            .attr("fill", "#64748b")
            .attr("transform", "rotate(-90)")
            .text("Distance");

        this.path = this.g.append("path")
            .attr("fill", "none")
            .attr("stroke", this.type === 'raw' ? '#3b82f6' : '#10b981')
            .attr("stroke-width", 2);

        this.hoverLine = this.g.append("line")
            .attr("stroke", "#94a3b8")
            .attr("stroke-width", 1.5)
            .attr("stroke-dasharray", "4,4")
            .style("opacity", 0)
            .style("pointer-events", "none");

        this.zeroLine = this.g.append("line")
            .attr("stroke", "#64748b")
            .attr("stroke-width", 1)
            .attr("stroke-dasharray", "3,3")
            .style("pointer-events", "none");

        this.dotsGroup = this.g.append("g");
    }

    handleMouseMove(event) {
        if (!this.data || this.data.length === 0 || !this.xScale) return;

        const [mx] = d3.pointer(event);
        const xValue = this.xScale.invert(mx);
        const bisect = d3.bisector(d => d.timestep).center;
        const i = bisect(this.data, xValue);

        if (i >= 0 && i < this.data.length) {
            const nearestPoint = this.data[i];
            state.update({ hoverTimestep: nearestPoint.timestep });
        }
    }

    handleMouseOut() {
        state.update({ hoverTimestep: null });
        tooltip.style("opacity", 0);
    }

    handleDotMouseOver(event, d) {
        event.stopPropagation();
        state.update({ hoverTimestep: d.timestep });

        tooltip.transition().duration(100).style("opacity", 0.9);
        const timeLabel = d.start_time !== undefined
            ? `transition: <strong>${d.start_time} → ${d.end_time}</strong>`
            : `timestep: <strong>${d.timestep}</strong>`;
        tooltip.html(`${timeLabel}<br/>value: <strong>${d.value.toFixed(4)}</strong>`)
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 28) + "px");
    }

    handleDotMouseOut(event, d) {
        event.stopPropagation();
        this.handleMouseOut();
    }

    handleDotClick(event, d) {
        event.stopPropagation();
        state.update({ activeTimestep: d.start_time ?? d.timestep });
        tooltip.style("opacity", 0);
    }

    render() {
        const appState = state.get();
        this.data = this.type === 'raw' ? appState.rawTimelineData : appState.explTimelineData;

        if (!this.data || this.data.length === 0) {
            this.g.selectAll("*").style("display", "none");
            this.yAxisTitle.style("display", "none");
            return;
        }

        this.g.selectAll("*").style("display", "");
        this.yAxisTitle.style("display", "");

        const containerNode = document.getElementById(this.containerId);
        const width = containerNode.clientWidth - this.margin.left - this.margin.right;
        const height = containerNode.clientHeight - this.margin.top - this.margin.bottom;

        if (width <= 0 || height <= 0) return;

        this.g.attr("transform", `translate(${this.margin.left},${this.margin.top})`);
        this.overlay.attr("width", width).attr("height", height);

        // Titles placement
        this.xAxisTitle
            .attr("x", width / 2)
            .attr("y", height + this.margin.bottom - 5);

        this.yAxisTitle
            .attr("y", 15)
            .attr("x", -(height / 2) - this.margin.top);

        const domain = appState.timelineDomain ?? d3.extent(this.data, d => d.timestep);
        const [domainStart, domainEnd] = domain;
        this.xScale = d3.scaleLinear()
            .domain(domainStart === domainEnd ? [domainStart - 0.5, domainEnd + 0.5] : domain)
            .range([0, width]);

        // The lower panel is the signed comparison (TS or J_G − J_E).
        // Its zero reference separates changes in the graph from changes in its explanation.
        const showZeroReference = this.type === 'expl';
        const isMissing = d => d.value === -1 && !showZeroReference;
        const validData = this.data.filter(d => Number.isFinite(d.value) && !isMissing(d));
        let yMin = d3.min(validData, d => d.value) !== undefined ? d3.min(validData, d => d.value) : 0;
        let yMax = d3.max(validData, d => d.value) !== undefined ? d3.max(validData, d => d.value) : 1;
        if (showZeroReference) {
            yMin = Math.min(yMin, 0);
            yMax = Math.max(yMax, 0);
        }
        const yPadding = (yMax - yMin) * 0.1 || 0.1;

        let diffThreshold = Infinity;
        const diffMap = new Map();
        if (appState.rawTimelineData && appState.explTimelineData) {
            let diffValues = [];
            const rawMap = new Map(appState.rawTimelineData.filter(d => d.value >= 0).map(d => [d.timestep, d.value]));
            const explMap = new Map(appState.explTimelineData.filter(d => d.value >= 0).map(d => [d.timestep, d.value]));

            for (const [ts, rawVal] of rawMap.entries()) {
                if (explMap.has(ts)) {
                    const diff = Math.abs(rawVal - explMap.get(ts));
                    diffMap.set(ts, diff);
                    diffValues.push(diff);
                }
            }
            if (diffValues.length > 1) {
                const mean = diffValues.reduce((a, b) => a + b, 0) / diffValues.length;
                const variance = diffValues.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (diffValues.length - 1);
                const std = Math.sqrt(variance);
                diffThreshold = 2 * variance;
            }
        }

        const yLowerBound = showZeroReference ? yMin - yPadding : Math.max(0, yMin - yPadding);
        this.yScale = d3.scaleLinear()
            .domain([yLowerBound, yMax + yPadding])
            .range([height, 0]);

        const line = d3.line()
            .x(d => this.xScale(d.timestep))
            .y(d => this.yScale(d.value));

        this.path.datum(validData)
            .attr("d", line);

        this.zeroLine
            .attr("x1", 0).attr("x2", width)
            .attr("y1", this.yScale(0)).attr("y2", this.yScale(0))
            .style("display", showZeroReference ? "" : "none");

        const timeTicks = d3.range(Math.ceil(domainStart), Math.floor(domainEnd) + 1);
        this.xAxisGroup
            .attr("transform", `translate(0,${height})`)
            .call(d3.axisBottom(this.xScale).tickValues(timeTicks).tickFormat(d3.format("d")));

        const yTicks = showZeroReference ? Array.from(new Set([yMin, 0, yMax])).sort((a, b) => a - b) : [yMin, yMax];
        this.yAxisGroup
            .call(d3.axisLeft(this.yScale).tickValues(yTicks).tickFormat(d3.format(".3f")));

        const hoverTs = appState.hoverTimestep;
        if (hoverTs !== null) {
            const hx = this.xScale(hoverTs);
            this.hoverLine
                .attr("x1", hx).attr("x2", hx)
                .attr("y1", 0).attr("y2", height)
                .style("opacity", 1);
        } else {
            this.hoverLine.style("opacity", 0);
        }

        // Mapper highlight matching
        const mapperHoverTsSet = new Set();
        if (appState.hoverNodeType === this.type && Array.isArray(appState.hoverNodeElements)) {
            appState.hoverNodeElements.forEach(item => mapperHoverTsSet.add(Number(item)));
        }

        const activeTs = appState.activeTimestep;

        const dots = this.dotsGroup.selectAll("circle")
            .data(this.data, d => d.timestep);

        dots.enter()
            .append("circle")
            .attr("cursor", "pointer")
            .on("mouseover", (e, d) => this.handleDotMouseOver(e, d))
            .on("mouseout", (e, d) => this.handleDotMouseOut(e, d))
            .on("click", (e, d) => this.handleDotClick(e, d))
            .merge(dots)
            .attr("cx", d => this.xScale(d.timestep))
            .attr("cy", d => isMissing(d) ? this.yScale(yMin) : this.yScale(d.value))
            .attr("r", d => {
                if (d.timestep === hoverTs) return 6;
                if (d.timestep === activeTs) return 5;
                if (mapperHoverTsSet.has(d.timestep)) return 5;
                return 3.5;
            })
            .attr("fill", d => {
                if (isMissing(d)) return '#94a3b8';
                if (diffMap.has(d.timestep) && diffMap.get(d.timestep) > diffThreshold) return 'red';
                if (mapperHoverTsSet.has(d.timestep)) return '#f59e0b'; // amber-500 for mapper highlight
                return this.type === 'raw' ? '#1e3a8a' : '#064e3b';
            })
            .attr("stroke", d => {
                if (d.timestep === activeTs) return '#ef4444'; // Red rim for active
                if (d.timestep === hoverTs) return '#f59e0b';
                return 'none';
            })
            .attr("stroke-width", d => (d.timestep === activeTs || d.timestep === hoverTs) ? 2 : 0)
            .style("z-index", d => (d.timestep === hoverTs || d.timestep === activeTs || mapperHoverTsSet.has(d.timestep)) ? 10 : 1);

        dots.exit().remove();
    }
}
