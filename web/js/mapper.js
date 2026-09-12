import { state } from './state.js?v=signed-metrics';

export class MapperView {
    constructor(containerId, type) {
        this.containerId = containerId;
        this.type = type; // 'raw' or 'expl'
        this.margin = { top: 10, right: 10, bottom: 10, left: 10 };
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

        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (e) => {
                this.zoomG.attr("transform", e.transform);
                const invScale = 1 / e.transform.k;
                this.nodeGroup.selectAll("circle").attr("r", d => (d && d.baseR ? d.baseR : 5) * invScale);
                this.linkGroup.selectAll("line").attr("stroke-width", 1 * invScale);
            });

        this.svg.call(zoom);

        this.linkGroup = this.zoomG.append("g").attr("class", "links");
        this.nodeGroup = this.zoomG.append("g").attr("class", "nodes");
    }

    render() {
        const appState = state.get();
        const data = this.type === 'raw' ? appState.rawMapperData : appState.explMapperData;

        if (!data || !data.nodes) {
            this.linkGroup.selectAll("*").remove();
            this.nodeGroup.selectAll("*").remove();
            return;
        }

        const containerNode = document.getElementById(this.containerId);
        const width = containerNode.clientWidth || 300;
        const height = containerNode.clientHeight || 200;

        const nodes = Array.isArray(data.nodes) ? data.nodes : Object.values(data.nodes);
        const links = Array.isArray(data.edges) ? data.edges : (data.edges ? Object.values(data.edges) : []);

        if (nodes.length > 0) {
            const getElements = n => n.timesteps || n.elements || n.points || n.nodes || [];
            const maxElements = d3.max(nodes, n => getElements(n).length) || 1;
            const radiusScale = d3.scaleSqrt().domain([1, maxElements]).range([3, 15]);
            // Mapper layout scaling. Assuming they come with some x,y coordinates
            // or pos attribute
            const getX = n => n.pos?.[0] ?? n.x ?? 0;
            const getY = n => n.pos?.[1] ?? n.y ?? 0;

            const xExtent = d3.extent(nodes, getX);
            const yExtent = d3.extent(nodes, getY);

            let xScale = d => d;
            let yScale = d => d;

            if (xExtent[0] !== undefined && xExtent[1] !== undefined) {
                // scale graph to fit
                const padding = 20;
                xScale = d3.scaleLinear().domain(xExtent).range([padding, width - padding]);
                yScale = d3.scaleLinear().domain(yExtent).range([padding, height - padding]);

                nodes.forEach(n => {
                    n.sx = xScale(getX(n));
                    n.sy = yScale(getY(n));
                    n.baseR = radiusScale(getElements(n).length || 1);
                });
            } else {
                nodes.forEach(n => {
                    n.sx = width / 2;
                    n.sy = height / 2;
                    n.baseR = radiusScale(getElements(n).length || 1);
                });
            }
        }

        const nodeMap = new Map(nodes.map(n => [String(n.id), n]));

        const linkSel = this.linkGroup.selectAll("line")
            .data(links, d => `${d.source}-${d.target}`);

        linkSel.enter()
            .append("line")
            .merge(linkSel)
            .attr("x1", d => nodeMap.get(String(d.source))?.sx || 0)
            .attr("y1", d => nodeMap.get(String(d.source))?.sy || 0)
            .attr("x2", d => nodeMap.get(String(d.target))?.sx || 0)
            .attr("y2", d => nodeMap.get(String(d.target))?.sy || 0)
            .attr("stroke", "#94a3b8")
            .attr("stroke-width", () => {
                const k = d3.zoomTransform(this.svg.node()).k || 1;
                return 1 / k;
            });

        linkSel.exit().remove();

        const nodeSel = this.nodeGroup.selectAll("circle")
            .data(nodes, d => d.id);

        nodeSel.enter()
            .append("circle")
            .merge(nodeSel)
            .attr("cx", d => d.sx)
            .attr("cy", d => d.sy)
            .attr("r", d => {
                const k = d3.zoomTransform(this.svg.node()).k || 1;
                return (d.baseR || 5) / k;
            })
            // If the node is currently hovered
            .attr("fill", d => (appState.hoverNodeId && String(appState.hoverNodeId) === String(d.id) && appState.hoverNodeType === this.type) ? '#f59e0b' : '#3b82f6')
            .attr("stroke", "#1e3a8a")
            .on("mouseover", (event, d) => {
                // Find timesteps from common field names used by Mapper algorithms
                const elements = d.timesteps || d.elements || d.points || d.nodes || [];
                state.update({ hoverNodeId: String(d.id), hoverNodeType: this.type, hoverNodeElements: elements });
            })
            .on("mouseout", () => {
                state.update({ hoverNodeId: null, hoverNodeType: null, hoverNodeElements: null });
            });

        nodeSel.exit().remove();
    }
}
