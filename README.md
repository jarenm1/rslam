# MDE Visual SLAM


![GitHub last commit](https://img.shields.io/github/last-commit/jarenm1/rslam)
[![Crates.io](https://img.shields.io/crates/v/rslam.svg)](https://crates.io/crates/rslam)

---

## Current Features

| Feature | Status | Notes |
|---|---|---|
| Core SLAM System | 🚧 WIP | Basic SLAM structure & modules |
| Feature Detection | ✅ Complete | ORB feature detection |
| Feature Matching | ✅ Complete | BFMatching |
| Keyframe Management | 🚧 WIP | Management of keyframes is being developed |
| ROS Integration | 🚧 WIP | A ROS node is created to subscribe to raw camera feed |

## Planned Features

- ROS integration for publishing the map and camera pose
- Real-time performance
- Loop closure
- Relocalization
- MDE for dense points
- Bevy Gaussian splatting visulization w/ Bevy plugin
