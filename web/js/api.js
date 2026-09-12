export async function getTimeline(dim, metric, dataset, series) {
    const query = new URLSearchParams({ dataset, view: series });
    const res = await fetch(`/api/timelines/${dim}/${metric}?${query}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

export async function getGraphs(timestep, dataset) {
    const res = await fetch(`/api/graphs/${timestep}?dataset=${encodeURIComponent(dataset)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

export async function getMapper(dim) {
    const res = await fetch(`/api/mapper/${dim}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}
