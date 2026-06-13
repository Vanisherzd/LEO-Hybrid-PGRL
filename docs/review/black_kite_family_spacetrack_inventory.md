# BLACK KITE Family Space-Track Data Inventory
This inventory records locally fetched raw Space-Track artifacts for BLACK KITE family satellites. Raw data remain under `dataraw/` and are not committed.

## Summary
| Object | NORAD | History TLE bytes | History JSON bytes | Manifest |
|---|---:|---:|---:|---|
| BLACK KITE-1 | 66741 | 58930 | 468458 | `dataraw/spacetrack/black_kite_1_66741/fetch_manifest.json` |
| BLACK KITE-2 | 68474 | 26128 | 207976 | `dataraw/spacetrack/black_kite_2_68474/fetch_manifest.json` |

## Claims and limitations
- These artifacts support real Space-Track TLE provenance for BLACK KITE family satellites.
- They support software-only model-derived Doppler/replay experiments.
- `reference_is_measured_truth=false`: these are not measured Doppler truth.
- They do not support live satellite, RF, PER/BER/CRC, or gateway validation claims.
- BLACK KITE-2 should remain the evaluation target; BLACK KITE-1 can serve as auxiliary train/pretrain data.
