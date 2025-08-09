use depth_estimate::{DepthEstimate, DepthEstimateConfig};
use opencv::imgcodecs::{self, IMREAD_ANYCOLOR};
use std::path::PathBuf;
//
//
// REQUIRES MIDAS-SMALL ONNX MODEL & OPENCV TO WORK
//
//
fn main() {
    let image = imgcodecs::imread("image1.jpg", IMREAD_ANYCOLOR).unwrap();

    let config = DepthEstimateConfig::new(
        ort::session::builder::GraphOptimizationLevel::Level3,
        1,
        PathBuf::from("model-small.onnx"),
    );
    let mut estimate = DepthEstimate::new(config).unwrap();
    let outputs = estimate.estimate(image).unwrap();
}
