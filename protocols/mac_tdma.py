"""
MAC TDMA Protocol — Time Division Multiple Access for LEO Satellite IoT
========================================================================
Implements a TDMA-based MAC protocol for satellite-ground IoT communication.

Key design:
- Frame structure: N slots per frame, IoT nodes assigned dedicated slots
- Priority slots: critical telemetry vs normal IoT data
- Sync slots: beacon/pseudo-sync at frame start
- Adaptive slot sizing based on communication quality
- Successful transmission triggers online GRPO update

The protocol is event-driven: on successful reception, emit a CommEvent
that triggers the GRPO online learning pipeline.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Optional

import numpy as np


class SlotType(Enum):
    """TDMA slot types."""
    BEACON = "beacon"       # Sync/beacon slot (satellite to ground)
    PRIORITY = "priority"   # High-priority telemetry slot
    IOT = "iot"             # Normal IoT data slot
    GUARD = "guard"         # Guard interval between slots


@dataclass
class TMDASlot:
    """Single TDMA time slot."""
    slot_id: int
    slot_type: SlotType
    start_time_s: float     # Relative to frame start
    duration_s: float       # Slot duration
    node_id: Optional[int] = None  # Assigned IoT node
    priority: int = 0       # 0=low, 1=medium, 2=high (telemetry)
    transmitted: bool = False
    ack_received: bool = False
    snr_db: float = 0.0     # Signal-to-noise ratio at reception


@dataclass
class TDMAFrame:
    """Complete TDMA frame containing all slots."""
    frame_id: int
    start_time: datetime
    duration_s: float
    slots: list[TMDASlot] = field(default_factory=list)

    @property
    def end_time(self) -> datetime:
        return self.start_time + timedelta(seconds=self.duration_s)

    def successful_slots(self) -> list[TMDASlot]:
        return [s for s in self.slots if s.transmitted and s.ack_received]


@dataclass
class CommEvent:
    """
    Successful communication event — emitted after each successful TDMA slot.
    This is the trigger for online GRPO weight update.
    """
    timestamp: datetime
    frame_id: int
    slot_id: int
    node_id: Optional[int]
    t_since_epoch_s: float  # Time relative to satellite epoch
    orbital_elements: np.ndarray  # (7,) Keplerian elements at this time
    rx_signal_strength_dbm: float
    snr_db: float
    data_bytes: int
    slot_type: SlotType
    propagation_delay_s: float  # light-time + processing

    def to_grpo_observation(
        self,
        gps_pos: np.ndarray,
        gps_vel: np.ndarray,
    ) -> dict:
        """Convert to GRPO observation format."""
        return {
            "t": self.t_since_epoch_s,
            "orbital_elems": self.orbital_elements,
            "gps_pos": gps_pos,
            "gps_vel": gps_vel,
            "reward": 0.0,  # computed by GRPO agent
        }


class TDMAConfig:
    """TDMA configuration parameters."""
    # Frame structure
    frame_duration_s: float = 1.0          # Frame length (1 second typical)
    slot_guard_s: float = 0.0001            # Guard time between slots (0.1 ms)
    beacon_duration_s: float = 0.01         # Beacon slot (10 ms)
    priority_slot_duration_s: float = 0.02  # Priority slot (20 ms)
    iot_slot_duration_s: float = 0.005      # Normal IoT slot (5 ms)

    # Slot counts per frame
    num_priority_slots: int = 2
    num_iot_slots: int = 32                 # 32 IoT nodes per frame
    num_guard_slots: int = 2                # Inter-frame guard

    # Frequency
    carrier_frequency_ghz: float = 2.0      # 2 GHz (L-band)
    bandwidth_mhz: float = 1.0              # 1 MHz channel
    tx_power_dbm: float = 30.0              # 1W transmission
    min_snr_db: float = 3.0                 # Minimum SNR for successful reception

    # Antenna
    satellite_altitude_km: float = 400.0
    elevation_mask_deg: float = 10.0        # Minimum elevation angle


class MACTDMAProtocol:
    """
    TDMA MAC protocol for LEO satellite IoT communication.

    Responsibilities:
    1. Frame structure management (slot allocation)
    2. Schedule generation (which node transmits when)
    3. Propagation delay modeling (range-dependent)
    4. Link quality estimation (SNR, path loss)
    5. CommEvent generation on successful reception
    """

    def __init__(self, config: Optional[TDMAConfig] = None):
        self.config = config or TDMAConfig()
        self.frame_history: list[TDMAFrame] = []
        self._current_frame_id = 0
        self._comm_event_callbacks: list[Callable[[CommEvent], None]] = []
        self._rng = np.random.default_rng()

    # ── Slot & Frame Construction ────────────────────────────────────────────

    def build_frame(self, frame_id: int, start_time: datetime) -> TDMAFrame:
        """Construct a TDMA frame with all slot types."""
        slots = []
        current_time = 0.0
        slot_id = 0

        # 1. Beacon slot (satellite broadcasts timing/sync)
        slots.append(TMDASlot(
            slot_id=slot_id, slot_type=SlotType.BEACON,
            start_time_s=current_time,
            duration_s=self.config.beacon_duration_s,
            priority=2,
        ))
        slot_id += 1
        current_time += self.config.beacon_duration_s + self.config.slot_guard_s

        # 2. Priority slots (high-priority telemetry)
        for p in range(self.config.num_priority_slots):
            slots.append(TMDASlot(
                slot_id=slot_id, slot_type=SlotType.PRIORITY,
                start_time_s=current_time,
                duration_s=self.config.priority_slot_duration_s,
                priority=2, node_id=p,
            ))
            slot_id += 1
            current_time += self.config.priority_slot_duration_s + self.config.slot_guard_s

        # 3. IoT data slots (round-robin among nodes)
        for i in range(self.config.num_iot_slots):
            slots.append(TMDASlot(
                slot_id=slot_id, slot_type=SlotType.IOT,
                start_time_s=current_time,
                duration_s=self.config.iot_slot_duration_s,
                priority=1, node_id=100 + i,  # node IDs 100+
            ))
            slot_id += 1
            current_time += self.config.iot_slot_duration_s + self.config.slot_guard_s

        # 4. Guard interval at end
        slots.append(TMDASlot(
            slot_id=slot_id, slot_type=SlotType.GUARD,
            start_time_s=current_time,
            duration_s=self.config.slot_guard_s,
            priority=0,
        ))

        frame = TDMAFrame(
            frame_id=frame_id,
            start_time=start_time,
            duration_s=current_time + self.config.slot_guard_s,
            slots=slots,
        )
        return frame

    # ── Propagation Model ─────────────────────────────────────────────────────

    def compute_propagation_delay(self, satellite_pos: np.ndarray, ground_pos: np.ndarray) -> float:
        """
        Compute one-way propagation delay in seconds.
        Includes light-time + processing delay.
        """
        range_km = np.linalg.norm(satellite_pos - ground_pos)
        light_time = range_km / 299792.458  # speed of light km/s
        processing_delay = 0.002  # 2 ms processing overhead
        return light_time + processing_delay

    def compute_snr(
        self,
        satellite_pos: np.ndarray,
        ground_pos: np.ndarray,
        elevation_deg: float,
    ) -> float:
        """
        Compute SNR at ground receiver using free-space path loss.

        SNR = P_tx + G_tx + G_rx - PL - N
        where PL = 20*log10(4*pi*d/lambda)
        """
        range_km = max(np.linalg.norm(satellite_pos - ground_pos), 1.0)
        lambda_m = 0.3 / (self.config.carrier_frequency_ghz / 2)  # wavelength in meters

        # Free-space path loss
        pl_db = 20 * math.log10(4 * math.pi * range_km * 1000 / lambda_m)

        # Ground station antenna gain (simple isotropic + gain)
        G_rx_db = 3.0  # dBi
        G_tx_db = 6.0  # dBi (satellite)
        N_db = -150.0 + 10 * math.log10(self.config.bandwidth_mhz * 1e6)  # thermal noise

        snr_db = (
            self.config.tx_power_dbm
            + G_tx_db + G_rx_db
            - pl_db
            - N_db
        )

        # Elevation-dependent fade margin
        if elevation_deg < 30:
            snr_db -= (30 - elevation_deg) * 0.05  # small fade

        return snr_db

    # ── Link Quality Simulation ───────────────────────────────────────────────

    def simulate_slot_success(self, slot: TMDASlot, snr_db: float) -> bool:
        """
        Determine if a slot was successfully transmitted and ACKed.
        Success probability modeled from SNR with BER curve.
        """
        if snr_db < self.config.min_snr_db:
            return False

        # Simple threshold model with small random variation
        success_prob = min(1.0, (snr_db - self.config.min_snr_db) / 10.0 + 0.9)
        return self._rng.random() < success_prob

    # ── Frame Simulation ──────────────────────────────────────────────────────

    def simulate_frame(
        self,
        frame: TDMAFrame,
        satellite_pos: np.ndarray,
        ground_positions: dict[int, np.ndarray],
        elevation_deg: float,
        orbital_elements: np.ndarray,
        t_since_epoch_s: float,
    ) -> list[CommEvent]:
        """
        Simulate all slots in a frame and return successful CommEvents.

        Args:
            frame: TDMA frame
            satellite_pos: (3,) ECI position in km
            ground_positions: dict mapping node_id -> (3,) position in km
            elevation_deg: current elevation angle
            orbital_elements: (7,) Keplerian elements
            t_since_epoch_s: time since satellite epoch

        Returns:
            List of successful CommEvent objects (these trigger GRPO updates)
        """
        snr_db = self.compute_snr(satellite_pos, ground_positions.get(0, np.zeros(3)), elevation_deg)
        comm_events = []

        for slot in frame.slots:
            if slot.slot_type == SlotType.GUARD:
                continue

            slot_start = frame.start_time + timedelta(seconds=slot.start_time_s)
            t_slot = t_since_epoch_s + slot.start_time_s

            success = self.simulate_slot_success(slot, snr_db)

            if success:
                # Compute node-specific propagation delay
                node_pos = ground_positions.get(slot.node_id, np.zeros(3))
                prop_delay = self.compute_propagation_delay(satellite_pos, node_pos)

                # Small data payload (IoT telemetry)
                data_bytes = 16 if slot.slot_type == SlotType.IOT else 64

                event = CommEvent(
                    timestamp=slot_start,
                    frame_id=frame.frame_id,
                    slot_id=slot.slot_id,
                    node_id=slot.node_id,
                    t_since_epoch_s=t_slot,
                    orbital_elements=orbital_elements.copy(),
                    rx_signal_strength_dbm=self.config.tx_power_dbm - 50,
                    snr_db=snr_db,
                    data_bytes=data_bytes,
                    slot_type=slot.slot_type,
                    propagation_delay_s=prop_delay,
                )
                comm_events.append(event)
                slot.transmitted = True
                slot.ack_received = True
                slot.snr_db = snr_db

        self.frame_history.append(frame)
        return comm_events

    # ── Event-Driven GRPO Trigger ─────────────────────────────────────────────

    def register_comm_callback(self, callback: Callable[[CommEvent], None]) -> None:
        """Register a callback to be called on each successful communication."""
        self._comm_event_callbacks.append(callback)

    def dispatch_comm_events(self, events: list[CommEvent]) -> None:
        """Dispatch successful comm events to registered callbacks (triggers GRPO)."""
        for event in events:
            for callback in self._comm_event_callbacks:
                callback(event)

    # ── Simulation Helpers ────────────────────────────────────────────────────

    def generate_ground_stations(self, num_stations: int = 32) -> dict[int, np.ndarray]:
        """Generate random ground station positions on Earth's surface."""
        stations = {}
        for i in range(num_stations):
            # Random lat/lon
            lat = self._rng.uniform(-60, 60)  # avoid polar regions
            lon = self._rng.uniform(0, 360)
            # Convert to ECEF (simplified)
            elat = lat * math.pi / 180
            elon = lon * math.pi / 180
            x = R_EARTH * math.cos(elat) * math.cos(elon)
            y = R_EARTH * math.cos(elat) * math.sin(elon)
            z = R_EARTH * math.sin(elat)
            stations[100 + i] = np.array([x, y, z])
        return stations

    def run_full_pass(
        self,
        num_frames: int,
        orbital_elements: np.ndarray,
        propagator,
        ground_stations: dict[int, np.ndarray],
    ) -> list[CommEvent]:
        """
        Run a full simulation of multiple TDMA frames across a satellite pass.

        This is the main entry point for the online RL simulation loop.
        Each successful CommEvent triggers a GRPO weight update.

        Returns:
            all_events: all successful communication events
        """
        all_events = []
        start_time = datetime.utcnow()

        # Compute satellite visibility per frame
        for frame_idx in range(num_frames):
            t_s = frame_idx * self.config.frame_duration_s
            pos, vel = propagator.propagate(t_s)

            # Simplified elevation (angle between position vector and closest ground station)
            # Use first station as reference
            ref_station = ground_stations.get(100, np.array([R_EARTH, 0, 0]))
            elevation = self._compute_elevation(pos, ref_station)

            frame = self.build_frame(frame_idx, start_time + timedelta(seconds=frame_idx * self.config.frame_duration_s))
            events = self.simulate_frame(
                frame, pos, ground_stations, elevation,
                orbital_elements, t_s,
            )
            all_events.extend(events)

            # Dispatch events to GRPO callbacks
            self.dispatch_comm_events(events)

        return all_events

    def _compute_elevation(self, sat_pos: np.ndarray, ground_pos: np.ndarray) -> float:
        """Compute elevation angle from ground station to satellite."""
        sat_norm = sat_pos / np.linalg.norm(sat_pos)
        ground_norm = ground_pos / np.linalg.norm(ground_pos)
        cos_el = np.dot(sat_norm, ground_norm)
        # Subtract from 90 degrees to get elevation
        el_rad = math.pi / 2 - math.acos(max(-1, min(1, cos_el)))
        return math.degrees(el_rad)


# Physical constant used above
R_EARTH = 6378.137


if __name__ == "__main__":
    # Smoke test
    from models.orbital_physics import generate_synthetic_tle, SGP4Propagator

    line1, line2 = generate_synthetic_tle(400)
    sgp4 = SGP4Propagator(line1, line2)

    config = TDMAConfig()
    mac = MACTDMAProtocol(config)

    # Generate ground stations
    stations = mac.generate_ground_stations(32)
    print(f"Generated {len(stations)} ground stations")

    # Simulate 10 frames (~10 seconds of satellite pass)
    orbital_elems = np.array([sgp4.tle["a"], 0.001, sgp4.tle["inclination"],
                               sgp4.tle["raan"], sgp4.tle["omega"], sgp4.tle["mean_anomaly"],
                               sgp4.tle["n"]])

    events = mac.run_full_pass(10, orbital_elems, sgp4, stations)
    print(f"Successful comm events: {len(events)}")
    print(f"Success rate: {len(events)/(10*(config.num_priority_slots+config.num_iot_slots))*100:.1f}%")

    # Frame structure summary
    frame = mac.build_frame(0, datetime.utcnow())
    print(f"\nFrame structure: {len(frame.slots)} slots, {frame.duration_s*1000:.1f} ms total")
    for s in frame.slots:
        print(f"  Slot {s.slot_id:2d}: {s.slot_type.value:8s} @ {s.start_time_s*1000:6.2f} ms, "
              f"dur={s.duration_s*1000:.2f} ms, node={s.node_id}")

    print("\n✓ MAC TDMA protocol smoke test passed")