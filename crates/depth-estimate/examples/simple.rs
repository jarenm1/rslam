use depth_estimate::{DepthEstimate, DepthEstimateConfig, midas::transforms::MidasTransform};
use opencv::imgcodecs::{self, IMREAD_ANYCOLOR};
use std::path::PathBuf;
//
//
// REQUIRES MIDAS-SMALL ONNX MODEL & OPENCV TO WORK
//
//
fn main() {
    // Takes anycolor ?
    let image = imgcodecs::imread("image1.jpg", IMREAD_ANYCOLOR).unwrap();

    let config = DepthEstimateConfig::new(
        ort::session::builder::GraphOptimizationLevel::Level3,
        1,
        PathBuf::from("model-small.onnx"),
    );
    let mut estimate = DepthEstimate::new(
        config,
        Box::new(MidasTransform {
            target_height: 384,
            target_width: 384,
            mean: opencv::core::Scalar::from_array([0.485, 0.456, 0.406, 0.0]),
            std: opencv::core::Scalar::from_array([0.299, 0.224, 0.225, 0.0]),
        }),
    )
    .unwrap();
    let outputs = estimate.estimate(image).unwrap();
}
