import { state } from './state.js?v=signed-metrics';

const tooltip = d3.select("#d3-tooltip");
const colorScale = d3.scaleOrdinal(d3.schemeSet3);

export class GraphView {
    constructor(containerId, type) {
        this.containerId = containerId;
        this.type = type; // 'selected', 'prev', 'next'
        this.margin = { top: 20, right: 20, bottom: 20, left: 20 };
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
            .style("height", "100%");

        this.zoomG = this.svg.append("g");

        this.zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (e) => {
                // Determine if this event was triggered by user interaction or programmatic sync
                if (e.sourceEvent) {
                    state.update({ graphTransform: e.transform });
                } else {
                    this.applyTransform(e.transform);
                }
            });

        this.svg.call(this.zoom);

        this.linkGroup = this.zoomG.append("g").attr("class", "links");
        this.nodeGroup = this.zoomG.append("g").attr("class", "nodes");
    }

    applyTransform(transform) {
        if (!transform) return;
        this.zoomG.attr("transform", transform);
        const invScale = 1 / transform.k;
        this.nodeGroup.selectAll("circle").attr("r", 6 * invScale);
        this.linkGroup.selectAll("line").attr("stroke-width", d => {
            const imp = d.importance || 0;
            const w = 1 + (imp * 3); // Scale 1 to 4
            return w * invScale;
        });
    }

    render() {
        const appState = state.get();
        let graphData = null;
        if (this.type === 'curr' || this.type === 'curr-filtered') graphData = appState.currGraphData;
        else if (this.type === 'next' || this.type === 'next-filtered') graphData = appState.nextGraphData;

        if (!graphData || !graphData.nodes) {
            this.linkGroup.selectAll("*").remove();
            this.nodeGroup.selectAll("*").remove();
            if (this.type === 'curr') {
                const legendEl = document.getElementById("graph-legend");
                if (legendEl) legendEl.innerHTML = "";
            }
            return;
        }

        const containerNode = document.getElementById(this.containerId);
        const width = containerNode.clientWidth || 350;
        const height = containerNode.clientHeight || 350;

        const nodes = Array.isArray(graphData.nodes) ? graphData.nodes : Object.values(graphData.nodes);
        let links = Array.isArray(graphData.edges) ? graphData.edges : Object.values(graphData.edges);

        if (this.type.includes('filtered')) {
            links = links.filter(edge => (edge.importance || 0) > 0);
        }

        // Apply external transform if changed
        if (appState.graphTransform) {
            // Apply silently without triggering the zoom event that writes back to state
            this.svg.call(this.zoom.transform, appState.graphTransform);
            this.applyTransform(appState.graphTransform);
        }

        // Compute fixed positions based on selected/curr graph, or just use their raw coordinates
        // The instructions say "node positions... must be the same and fixed, but scaled differently"
        if (nodes.length > 0) {
            const getX = n => n.pos?.[0] ?? n.x ?? 0;
            const getY = n => n.pos?.[1] ?? n.y ?? 0;

            const xExtent = d3.extent(nodes, getX);
            const yExtent = d3.extent(nodes, getY);

            let xScale = d => d;
            let yScale = d => d;

            if (xExtent[0] !== undefined && xExtent[1] !== undefined) {
                const xSpan = Math.max(xExtent[1] - xExtent[0], 0.01);
                const ySpan = Math.max(yExtent[1] - yExtent[0], 0.01);
                const padding = 20;

                xScale = d3.scaleLinear().domain(xExtent).range([padding, width - padding]);
                yScale = d3.scaleLinear().domain(yExtent).range([padding, height - padding]);

                nodes.forEach(n => {
                    n.sx = xScale(getX(n));
                    n.sy = yScale(getY(n));
                });
            }
        }

        const nodeMap = new Map(nodes.map(n => [String(n.id), n]));

        // Generate Legend in 'curr' view
        if (this.type === 'curr') {
            const classes = new Set(nodes.map(n => n.class !== undefined ? n.class : n.label));
            const legendEl = document.getElementById("graph-legend");
            if (legendEl) {
                legendEl.innerHTML = Array.from(classes).sort().map(cls => {
                    const color = colorScale(String(cls));
                    return `<div class="flex items-center gap-1"><span class="w-2 h-2 rounded-full inline-block" style="background-color: ${color}"></span>Class ${cls}</div>`;
                }).join("");
            }
        }

        const edgeId = (d) => {
            const sourceId = typeof d.source === 'object' ? d.source.id : d.source;
            const targetId = typeof d.target === 'object' ? d.target.id : d.target;
            // Sorting to handle undirected edge ID matching safely
            const [min, max] = [String(sourceId), String(targetId)].sort();
            return `${min}-${max}`;
        };

        const linkSel = this.linkGroup.selectAll("line")
            .data(links, d => edgeId(d));

        linkSel.enter()
            .append("line")
            .merge(linkSel)
            .attr("x1", d => {
                const sId = typeof d.source === 'object' ? d.source.id : d.source;
                return nodeMap.get(String(sId))?.sx || 0;
            })
            .attr("y1", d => {
                const sId = typeof d.source === 'object' ? d.source.id : d.source;
                return nodeMap.get(String(sId))?.sy || 0;
            })
            .attr("x2", d => {
                const tId = typeof d.target === 'object' ? d.target.id : d.target;
                return nodeMap.get(String(tId))?.sx || 0;
            })
            .attr("y2", d => {
                const tId = typeof d.target === 'object' ? d.target.id : d.target;
                return nodeMap.get(String(tId))?.sy || 0;
            })
            .attr("stroke", d => {
                const eId = edgeId(d);
                if (appState.hoverEdgeId === eId) return "#f59e0b"; // Highlight edge across graphs (amber)
                const imp = d.importance || 0;
                return "#94a3b8"; //imp >= 0.95 ? "#064e3b" : "#94a3b8"; // dark-green if >= 0.9, grey if < 0.9
            })
            .attr("stroke-opacity", 0.85)
            .attr("stroke-width", d => {
                const scale = d3.zoomTransform(this.svg.node()).k || 1;
                const invScale = 1 / scale;
                const imp = d.importance || 0;
                const baseW = 1 + (imp * 3); // scale [0,1] importance to [1,4] width
                return 2;//(appState.hoverEdgeId === edgeId(d) ? baseW * 1.5 : baseW) * invScale;
            })
            .on("mouseover", (event, d) => {
                state.update({ hoverEdgeId: edgeId(d) });
                const imp = d.importance || 0;
                tooltip.transition().duration(100).style("opacity", 0.9);
                tooltip.html(`importance: <strong>${imp.toFixed(3)}</strong>`)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            })
            .on("mouseout", () => {
                state.update({ hoverEdgeId: null });
                tooltip.style("opacity", 0);
            });

        linkSel.exit().remove();

        const nodeSel = this.nodeGroup.selectAll("circle")
            .data(nodes, d => d.id);

        nodeSel.enter()
            .append("circle")
            .merge(nodeSel)
            .attr("cx", d => d.sx)
            .attr("cy", d => d.sy)
            .attr("r", () => {
                const scale = d3.zoomTransform(this.svg.node()).k || 1;
                return 4 / scale;
            })
            .attr("fill", d => {
                const cls = d.class !== undefined ? d.class : d.label;
                return colorScale(String(cls));
            })
            .attr("fill-opacity", 0.85)
            .attr("stroke", d => {
                return (appState.hoverGraphNodeId === String(d.id)) ? '#ef4444' : '#475569';
            })
            .attr("stroke-width", d => {
                const scale = d3.zoomTransform(this.svg.node()).k || 1;
                const invScale = 1 / scale;
                return (appState.hoverGraphNodeId === String(d.id) ? 3 : 1) * invScale;
            })
            .on("mouseover", (event, d) => {
                state.update({ hoverGraphNodeId: String(d.id) });
                const cls = d.class !== undefined ? d.class : d.label;
                tooltip.transition().duration(100).style("opacity", 0.9);
                tooltip.html(`id: <strong>${d.id}</strong><br/>class: <strong>${cls}</strong>`)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            })
            .on("mouseout", () => {
                state.update({ hoverGraphNodeId: null });
                tooltip.style("opacity", 0);
            })
            .on("click", (event, d) => {
                console.log(`Node clicked: id=${d.id}`, d);
            });

        nodeSel.exit().remove();
    }
}
