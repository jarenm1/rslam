use super::{BinaryDescriptors, GrayscaleImage, MatchingError};
use opencv::{
    core::{KeyPoint, Vector, no_array},
    prelude::*,
};

#[derive(Debug)]
pub struct Features {
    pub keypoints: Vector<KeyPoint>,
    pub descriptors: BinaryDescriptors<'static>,
}

pub trait FeatureDetector {
    fn detect_features(&mut self, img: &GrayscaleImage) -> Result<Features, MatchingError>;
}

pub struct OrbDetector {
    detector: opencv::features2d::ORB,
}

impl FeatureDetector for OrbDetector {
    #[inline]
    fn detect_features(&mut self, img: &GrayscaleImage) -> Result<Features, MatchingError> {
        let mut keypoints: Vector<opencv::core::KeyPoint> = Vector::new();
        let mut descriptors = Mat::default();
        let mat = &*img.as_mat_view()?;
        self.detector.detect_and_compute(
            &mat,
            &no_array(),
            &mut keypoints,
            &mut descriptors,
            false,
        )?;

        Ok(Features {
            keypoints,
            descriptors: BinaryDescriptors::from_mat_borrowed(&descriptors)?.into_owned(),
        })
    }
}

pub enum FeatureDetectorModel {
    OrbDetector(OrbDetector),
}

impl FeatureDetector for FeatureDetectorModel {
    #[inline]
    fn detect_features(&mut self, img: &GrayscaleImage) -> Result<Features, MatchingError> {
        match self {
            FeatureDetectorModel::OrbDetector(detector) => detector.detect_features(img),
        }
    }
}
