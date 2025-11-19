"""Lightweight SLAM scaffold combining visual odometry, IMU fusion, and mapping."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class IMUMeasurement:
    timestamp: float
    accel: np.ndarray
    gyro: np.ndarray


@dataclass
class CameraFrame:
    timestamp: float
    image: np.ndarray
    keypoints: Optional[np.ndarray] = None
    descriptors: Optional[np.ndarray] = None


@dataclass
class Pose:
    position: np.ndarray
    orientation: np.ndarray


@dataclass
class SLAMConfig:
    feature_detector: str = "ORB"
    max_keypoints: int = 500
    map_resolution: float = 0.1


class SLAMPipeline:
    """Skeleton SLAM system that can run inside simulation notebooks."""

    def __init__(self, config: SLAMConfig | None = None) -> None:
        import cv2

        self.config = config or SLAMConfig()
        self.detector = cv2.ORB_create(self.config.max_keypoints)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.poses: List[Pose] = []
        self.map_points: List[np.ndarray] = []
        self.latest_frame: Optional[CameraFrame] = None

    def _extract_features(self, frame: CameraFrame) -> CameraFrame:
        keypoints, descriptors = self.detector.detectAndCompute(frame.image, None)
        frame.keypoints = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        frame.descriptors = descriptors
        return frame

    def _estimate_motion(self, frame_a: CameraFrame, frame_b: CameraFrame) -> Tuple[np.ndarray, np.ndarray]:
        import cv2

        matches = self.matcher.match(frame_a.descriptors, frame_b.descriptors)
        pts_a = np.float32([frame_a.keypoints[m.queryIdx] for m in matches])
        pts_b = np.float32([frame_b.keypoints[m.trainIdx] for m in matches])
        E, mask = cv2.findEssentialMat(pts_a, pts_b, method=cv2.RANSAC, prob=0.999)
        _, R, t, _ = cv2.recoverPose(E, pts_a, pts_b)
        return R, t.squeeze()

    def _integrate_imu(self, imu: IMUMeasurement, dt: float) -> Pose:
        accel_world = imu.accel * dt
        gyro_delta = imu.gyro * dt
        position = accel_world
        orientation = gyro_delta
        return Pose(position=position, orientation=orientation)

    def track(self, frame: CameraFrame, imu: Optional[IMUMeasurement] = None) -> Pose:
        frame = self._extract_features(frame)
        if self.latest_frame is None:
            pose = Pose(position=np.zeros(3), orientation=np.array([1, 0, 0, 0]))
            self.poses.append(pose)
            self.latest_frame = frame
            return pose

        R, t = self._estimate_motion(self.latest_frame, frame)
        imu_pose = None
        if imu is not None:
            dt = frame.timestamp - self.latest_frame.timestamp
            imu_pose = self._integrate_imu(imu, dt)

        position = self.poses[-1].position + t.flatten()
        orientation = R.flatten()
        if imu_pose is not None:
            position = 0.8 * position + 0.2 * imu_pose.position
            orientation = 0.8 * orientation + 0.2 * imu_pose.orientation

        pose = Pose(position=position, orientation=orientation)
        self.poses.append(pose)
        self.latest_frame = frame
        return pose

    def get_trajectory(self) -> np.ndarray:
        return np.array([pose.position for pose in self.poses])

    def insert_map_point(self, point: np.ndarray) -> None:
        self.map_points.append(point)

    def export_map(self) -> np.ndarray:
        if not self.map_points:
            return np.empty((0, 3))
        return np.vstack(self.map_points)


__all__ = ["SLAMPipeline", "SLAMConfig", "IMUMeasurement", "CameraFrame", "Pose"]
