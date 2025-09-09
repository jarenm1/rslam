use super::{BinaryDescriptors, GrayscaleImage, MatchingError};
use opencv::{
    core::{KeyPoint, Ptr, Vector, no_array},
    prelude::*,
};
use tracing::{debug, info, trace};

#[derive(Debug)]
pub struct Features {
    pub keypoints: Vector<KeyPoint>,
    pub descriptors: BinaryDescriptors<'static>,
}

pub trait FeatureDetector {
    fn detect_features(&mut self, img: &GrayscaleImage) -> Result<Features, MatchingError>;
}

pub struct OrbDetector {
    detector: Ptr<opencv::features2d::ORB>,
}

impl FeatureDetector for OrbDetector {
    fn detect_features(&mut self, img: &GrayscaleImage) -> Result<Features, MatchingError> {
        trace!("Starting ORB feature detection");

        let mut keypoints: Vector<opencv::core::KeyPoint> = Vector::new();
        let mut descriptors = Mat::default();
        let mat = &*img.as_mat_view()?;

        debug!("Mat view created successfully for feature detection");
        debug!("Image dimensions: {}x{}", mat.rows(), mat.cols());

        self.detector.detect_and_compute(
            &mat,
            &no_array(),
            &mut keypoints,
            &mut descriptors,
            false,
        )?;

        info!("ORB detected {} keypoints", keypoints.len());
        debug!(
            "Descriptor matrix: {}x{}",
            descriptors.rows(),
            descriptors.cols()
        );

        let binary_descriptors = BinaryDescriptors::from_mat_borrowed(&descriptors)?.into_owned();
        debug!("Successfully converted descriptors to owned binary descriptors");

        Ok(Features {
            keypoints,
            descriptors: binary_descriptors,
        })
    }
}

pub enum FeatureDetectorModel {
    OrbDetector(OrbDetector),
}

impl FeatureDetectorModel {
    pub fn new_orb() -> Result<Self, MatchingError> {
        debug!("Creating new ORB detector with default parameters");
        debug!("ORB params: n_features=500, scale_factor=1.2, n_levels=8, edge_threshold=31");

        let orb = opencv::features2d::ORB::create(
            500,    // n_features: maximum number of features to retain
            1.2f32, // scale_factor: pyramid decimation ratio
            8,      // n_levels: number of pyramid levels
            31,     // edge_threshold: size of border where features are not detected
            0,      // first_level: level of pyramid to put source image to
            2, // wta_k: number of points that produce each element of oriented BRIEF descriptor
            opencv::features2d::ORB_ScoreType::HARRIS_SCORE, // score_type
            31, // patch_size: size of patch used by oriented BRIEF descriptor
            20, // fast_threshold: fast threshold
        )?;

        info!("Successfully created ORB detector");
        Ok(Self::OrbDetector(OrbDetector { detector: orb }))
    }
}

impl FeatureDetector for FeatureDetectorModel {
    fn detect_features(&mut self, img: &GrayscaleImage) -> Result<Features, MatchingError> {
        trace!("FeatureDetectorModel dispatching detection");
        match self {
            FeatureDetectorModel::OrbDetector(detector) => {
                debug!("Using ORB detector");
                detector.detect_features(img)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init_tracing() -> tracing::subscriber::DefaultGuard {
        let subscriber = tracing_subscriber::FmtSubscriber::builder()
            .with_max_level(tracing::Level::DEBUG)
            .with_test_writer()
            .finish();
        tracing::subscriber::set_default(subscriber)
    }

    #[test]
    fn test_orb_detector_creation() -> Result<(), Box<dyn std::error::Error>> {
        let _guard = init_tracing();
        info!("Starting test_orb_detector_creation");

        let _detector = FeatureDetectorModel::new_orb()?;
        info!("ORB detector created successfully");
        Ok(())
    }
}
