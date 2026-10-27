import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import drone_multilink as drone


def test_simulator_records_first_mission_completion_time():
    config = drone.SimConfig(
        physics_dt=0.1,
        radio_dt=0.1,
        start_pos=(5.0, 0.0, 0.0),
        waypoints=[drone.Waypoint((5.0, 0.0, 0.0))],
    )
    sim = drone.DroneMultiLinkSim(config)

    assert sim.mission_completion_s is None
    sim.step()

    assert sim.follower.done
    assert sim.mission_completion_s == 0.1
    assert sim.summary()["mission_completion_s"] == 0.1
