use super::{BinaryDescriptors, GrayscaleImage, MatchingError};
use opencv::{
    core::{Vector, no_array},
    prelude::*,
};

pub trait EdgeDetector {
    fn detect_edges(&mut self, img: &GrayscaleImage) -> Result<BinaryDescriptors, MatchingError>;
}

pub struct OrbDetector {
    detector: opencv::features2d::ORB,
}

impl EdgeDetector for OrbDetector {
    #[inline]
    fn detect_edges(&mut self, img: &GrayscaleImage) -> Result<BinaryDescriptors, MatchingError> {
        let mut keypoints: Vector<opencv::core::KeyPoint> = Vector::new();
        let mut descriptors = Mat::default();
        let mat = Mat::try_from(img)?;
        self.detector.detect_and_compute(
            &mat,
            &no_array(),
            &mut keypoints,
            &mut descriptors,
            false,
        )?;

        let result = BinaryDescriptors::try_from(&descriptors)?;

        Ok(result)
    }
}
