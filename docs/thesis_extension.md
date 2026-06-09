# Thesis / Journal Extension Roadmap

This document lists modules that exist in the broader repository but are **outside the clean workshop-submission scope** for the first paper.

The first paper should remain focused on:

> calibrated PGRL uncertainty → risk-aware LR-FHSS uplink control, with IQ-level signal detection only.

## Reserved for Thesis / Journal

| Module | Current Location | Status | Why Excluded from Workshop Paper |
|---|---|---:|---|
| TDMA / MAC scheduler | `protocols/`, `scripts/benchmark_tdma.py`, `tests/test_tdma.py` | Implemented simulation | Adds a second protocol/MAC contribution and distracts from LR-FHSS uncertainty-control story |
| PPO / GRPO online learning | `models/grpo_agent.py`, `physics_ml/grpo_agent.py`, `scripts/train_online.py`, `configs/online_grpo.yaml` | Implemented / legacy | Requires a full online-learning evaluation story; not needed for current claim |
| ISAC RF-driven self-healing | `src/sdr/isac_rf_orbit_healing.py` | Prototype | Powerful but separate closed-loop RF/orbit-correction paper |
| Packet validation / PER harness | `hardware/packet_validation/`, `docs/*packet*`, `docs/*per*` | Harness only | No standards-compliant decoded receiver; PER is unavailable and should not be claimed |
| Full LR-FHSS gateway | external / planned | Not available | Requires receiver-side decoder, CRC, and packet-delivery logs |
| Multi-node scheduling | concept / legacy TDMA code | Concept | Network-level problem beyond single-terminal uplink control |
| Semantic communication | docs only | Concept | Separate research thread |
| Long-term autonomous operation | concept | Not available | Requires field deployment and long-run logs |

## How to Treat These Files in the Repo

For a clean submission branch, either:

1. keep the files but mark them as **legacy / extension / non-paper artifacts**, or
2. move them later under `extensions/` or `legacy/` in a dedicated cleanup commit.

Do not cite these modules as evidence for the current workshop paper unless the corresponding experiments are completed and claim boundaries are rewritten.

## Relationship to the First Paper

The first paper establishes the core cross-layer idea:

1. Use SGP4/PGRL to produce a corrected state estimate and calibrated position-domain uncertainty.
2. Convert that uncertainty into a risk-aware LR-FHSS guard-control proxy.
3. Evaluate the control tradeoff with simulation/proxy metrics.
4. Provide conducted IQ-level RF evidence using LR1121 and USRP B210, without claiming decoder/PER.

A later thesis/journal version may add MAC scheduling, GRPO online learning, ISAC feedback, or PER only after the missing receiver-side evidence exists.
