use std::path::{Path, PathBuf};

use opencv::core::Mat;
use ort::{
    execution_providers::{CUDAExecutionProvider, TensorRTExecutionProvider},
    session::{Session, builder::GraphOptimizationLevel},
};

#[derive(Debug, thiserror::Error)]
pub enum EstimateError {
    #[error("ORT error")]
    OrtError(#[from] ort::Error),
}


pub struct DepthEstimate {
    model: Session,
}

impl DepthEstimate {
    pub fn new<P: AsRef<Path>>(config: DepthEstimateConfig) -> Result<Self, EstimateError> {
        let mut model = Session::builder()?
            .with_optimization_level(config.optimization_level)?
            .with_intra_threads(config.intra_threads)?
            .commit_from_file(config.file_path.clone())?;
        Ok(Self { model })
    }
    pub fn estimate(&mut self, image: Mat) {
        let input = image;
        // resize to 256x256 for midas small

        let mut resized = Mat::default();
        todo!()
    }
}

pub struct DepthEstimateConfig {
    optimization_level: GraphOptimizationLevel,
    intra_threads: usize,
    file_path: PathBuf,
}

impl DepthEstimateConfig {
    pub fn new<P: AsRef<Path>>(
        optimization_level: GraphOptimizationLevel,
        intra_threads: usize,
        file_path: P,
    ) -> Self {
        Self {
            optimization_level,
            intra_threads,
            file_path: file_path.as_ref().to_path_buf(),
        }
    }
}
