#!/usr/bin/env python3
"""ROS 2 wrapper around radioenv.  *** Not run against a live ROS 2 install (none in the build sandbox);
only syntax-checked. Treat as a starting point. ***

Subscribes  /drone/pose            geometry_msgs/PoseStamped   position in the map's local ENU frame (z = m MSL)
Publishes   /radioenv/link_quality std_msgs/String             JSON {link_id: {rssi_dbm, snr_db, sinr_db, ...}}
            /radioenv/diagnostics  diagnostic_msgs/DiagnosticArray   one status per link

Parameters  preset (str, default "demo"), seed (int), weather (str), config_path (str, overrides preset),
            rate_hz (float, re-publish rate when the pose is static), fading (str: full|shadow|none)

Time: seconds since node start (or ROS sim time if use_sim_time is true) is the environment time `t`.
"""
import json
import os
import sys

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from radioenv import RadioEnvConfig, RadioEnvironment, presets   # noqa: E402

_SCALARS = ("rssi_dbm", "snr_db", "sinr_db", "noise_dbm", "interference_dbm", "pathloss_db", "per", "throughput_mbps",
            "latency_ms", "latency_p99_ms", "available", "ctrl_reliable", "video_usable", "quality", "outage_prob", "margin_db")


class RadioEnvNode(Node):
    def __init__(self):
        super().__init__("radioenv")
        self.declare_parameter("preset", "demo")
        self.declare_parameter("seed", 1)
        self.declare_parameter("weather", "clear")
        self.declare_parameter("config_path", "")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("fading", "full")
        cp = self.get_parameter("config_path").value
        cfg = RadioEnvConfig.from_json(cp) if cp else presets.get(self.get_parameter("preset").value,
                                                                  seed=int(self.get_parameter("seed").value),
                                                                  weather=self.get_parameter("weather").value)
        self.env = RadioEnvironment(cfg)
        self.fading = self.get_parameter("fading").value
        self.t0 = self.get_clock().now()
        self.pos = None
        self.create_subscription(PoseStamped, "/drone/pose", self.on_pose, 10)
        self.pub_json = self.create_publisher(String, "/radioenv/link_quality", 10)
        self.pub_diag = self.create_publisher(DiagnosticArray, "/radioenv/diagnostics", 10)
        self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self.tick)
        self.get_logger().info(f"radioenv ready: links={self.env.link_ids}")

    def on_pose(self, msg: PoseStamped):
        p = msg.pose.position
        self.pos = np.array([p.x, p.y, p.z], dtype=float)

    def tick(self):
        if self.pos is None:
            return
        t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        res = self.env.query_all(self.pos, t, fading=self.fading)
        out, diag = {}, DiagnosticArray()
        diag.header.stamp = self.get_clock().now().to_msg()
        for lid, q in res.items():
            d = {k: float(getattr(q, k)) for k in _SCALARS if getattr(q, k) is not None}
            out[lid] = d
            st = DiagnosticStatus(name=f"radioenv/{lid}", hardware_id=q.kind)
            st.level = DiagnosticStatus.OK if d["ctrl_reliable"] or d["video_usable"] else (
                DiagnosticStatus.WARN if d["available"] else DiagnosticStatus.ERROR)
            st.message = f"SINR {d['sinr_db']:.1f} dB, {d['throughput_mbps']:.1f} Mbps, {d['latency_ms']:.0f} ms"
            st.values = [KeyValue(key=k, value=f"{v:.3f}") for k, v in d.items()]
            diag.status.append(st)
        self.pub_json.publish(String(data=json.dumps(out)))
        self.pub_diag.publish(diag)


def main():
    rclpy.init()
    node = RadioEnvNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
