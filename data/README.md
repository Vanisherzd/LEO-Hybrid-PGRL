# Data Registry

`data/` is a **lightweight registry / catalog**, not the full dataset. It documents
how TLE and SGP4-derived data are organized for the PGRL pipeline, and what is (and
is not) committed to the repository.

Only **schemas, manifests, and tiny examples** live here. Raw and processed datasets
remain **local-only**.

## Data flow

```
TLE records  ──▶  SGP4/SDP4 propagation  ──▶  normalized SGP4-derived state/elements
(orbital source)   (generated features)        + query time
                                                      │
                                                      ▼
                                                 PGRL predictor
```

- **TLE records** are the orbital *source* (external input; not generated here).
- **SGP4/SDP4-derived states** are *generated features* (Cartesian state + Doppler,
  geodetic position, etc.).
- **PGRL** consumes normalized SGP4-derived state/elements and a query time; it does
  not consume raw TLE text directly.
- **Raw data stays local-only** (`data_raw/`, `data_processed/`, or `local_archive/`).
  Hardware IQ stays under `hardware/captures/` or `local_archive/raw_iq/`.

## What is committed vs. local-only

| Class            | Committed? | Location                                  |
|------------------|-----------|--------------------------------------------|
| Schemas          | yes       | `data/schemas/`                            |
| Manifests        | yes       | `data/manifests/`                          |
| Tiny examples    | yes       | `data/examples/`                           |
| Registry index   | yes       | `data/registry.yaml`                       |
| Raw TLE archives | no        | `data_raw/tle/` (git-ignored)              |
| SGP4 features    | no        | `data_processed/sgp4_states/` (git-ignored)|
| Pass windows     | no        | `data_processed/pass_windows/` (git-ignored)|
| Hardware IQ      | no        | `hardware/captures/`, `local_archive/raw_iq/`|

**Do not commit** large datasets, raw SP3/BRDC/IQ, or real large TLE archives.

## Conventions for scripts

Scripts should write processed outputs to `data_processed/` or `validation_runs/`
(both git-ignored), **never** to a committed path. Example records under
`data/examples/` are illustrative only and must not be relied on for real results.

## Registry

See [`registry.yaml`](registry.yaml) for the logical dataset catalog
(`tle_catalog`, `sgp4_states`, `pass_windows`, `pgrl_features`,
`hardware_iq_summaries`). Each entry records its schema, whether it is committed,
its storage location, what generates it, what consumes it, and its claim scope.

## Note on the `data` Python package

`data/__init__.py` and `data/dataset.py` are the existing dataset-loading code and
are unrelated to this registry metadata. The registry adds catalog/schema files only.
