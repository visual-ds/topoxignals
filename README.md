# TopoXignals

Local web viewer for the bundled temporal-graph datasets. It provides the
HSNet, Email, and Toy selections with aligned graph and timeline panels.

## Data

The repository includes the example datasets under `data/`:

```text
data/
├── toy/syn_dataset.pkl
├── email/
│   ├── monthly_adjacency_lists.pkl
│   ├── monthly_global_edge_scores.pkl
│   ├── persistence_distances.json
│   └── persistence_distances_scored.json
└── hsnet/latest/
    ├── monday1_dfs.pkl
    ├── edge_importance_monday1_attention.csv
    └── results/{b,w}{0,1}_timeseries_monday1.json
```

## Configuration

The repository provides `config.toml.example`. To customize paths or the
initial dataset, create your own local copy:

```bash
cp config.toml.example config.toml
```

`config.toml` is intentionally ignored by Git, so local settings are not
published. Edit it when the data lives somewhere else or when you want to
change the initial dataset.

- `[paths].dataset_root` is the directory that contains `toy/`, `email/`, and
  `hsnet/`. It is `data` by default.
- `[paths]` and `[edge_scores]` contain the Email filenames, relative to
  `dataset_root`.
- `[viewer].default_dataset` selects `hsnet`, `email`, or `toy` at startup.

## Launch the web viewer

Install the project and run the server:

```bash
python -m pip install -e .
xplain-server
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).
