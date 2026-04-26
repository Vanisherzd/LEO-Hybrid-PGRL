"""Tests for MAC TDMA protocol."""
import pytest
from datetime import datetime
import numpy as np
from protocols.mac_tdma import (
    MACTDMAProtocol, TDMAConfig, SlotType, CommEvent, R_EARTH,
)


def test_tdma_config_defaults():
    config = TDMAConfig()
    assert config.num_iot_slots == 32
    assert config.num_priority_slots == 2
    assert config.frame_duration_s == 1.0
    assert config.min_snr_db == 3.0


def test_build_frame_structure():
    config = TDMAConfig()
    mac = MACTDMAProtocol(config)
    frame = mac.build_frame(0, datetime.utcnow())
    assert frame.frame_id == 0
    assert len(frame.slots) > 0
    last_time = 0.0
    for slot in frame.slots:
        assert slot.start_time_s >= last_time
        last_time = slot.start_time_s + slot.duration_s


def test_propagation_delay():
    config = TDMAConfig()
    mac = MACTDMAProtocol(config)
    sat_pos = np.array([6778.0, 0.0, 0.0])
    ground_pos = np.array([6378.0, 0.0, 0.0])
    delay = mac.compute_propagation_delay(sat_pos, ground_pos)
    assert 0.002 < delay < 0.050


def test_snr_computation():
    config = TDMAConfig()
    mac = MACTDMAProtocol(config)
    sat_pos = np.array([6778.0, 0.0, 0.0])
    ground_pos = np.array([6378.0, 0.0, 0.0])
    snr = mac.compute_snr(sat_pos, ground_pos, elevation_deg=90.0)
    assert -50 < snr < 100


def test_ground_station_generation():
    config = TDMAConfig()
    mac = MACTDMAProtocol(config)
    stations = mac.generate_ground_stations(32)
    assert len(stations) == 32
    for pos in stations.values():
        assert pos.shape == (3,)
        r = np.linalg.norm(pos)
        assert 6300 < r < 6500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
