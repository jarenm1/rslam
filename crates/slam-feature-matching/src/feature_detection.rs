use super::{BinaryDescriptors, GrayscaleImage, MatchingError};
pub use opencv::features2d::ORB_ScoreType;
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

pub struct OrbConfig {
    n_features: i32,
    scale_factor: f32,
    nlevels: i32,
    edge_threshold: i32,
    first_level: i32,
    wta_k: i32,
    score_type: ORB_ScoreType,
    patch_size: i32,
    fast_threshold: i32,
}

impl OrbConfig {
    pub fn new(
        n_features: i32,
        scale_factor: f32,
        nlevels: i32,
        edge_threshold: i32,
        first_level: i32,
        wta_k: i32,
        score_type: ORB_ScoreType,
        patch_size: i32,
        fast_threshold: i32,
    ) -> Self {
        OrbConfig {
            n_features,
            scale_factor,
            nlevels,
            edge_threshold,
            first_level,
            wta_k,
            score_type,
            patch_size,
            fast_threshold,
        }
    }
}

impl Default for OrbConfig {
    fn default() -> Self {
        OrbConfig {
            n_features: 8,
            scale_factor: 1.2,
            nlevels: 8,
            edge_threshold: 31,
            first_level: 0,
            wta_k: 2,
            score_type: ORB_ScoreType::HARRIS_SCORE,
            patch_size: 31,
            fast_threshold: 20,
        }
    }
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
    pub fn new_orb(config: OrbConfig) -> Result<Self, MatchingError> {
        debug!("Creating new ORB detector with default parameters");
        debug!("ORB params: n_features=500, scale_factor=1.2, n_levels=8, edge_threshold=31");

        let orb = opencv::features2d::ORB::create(
            config.n_features,
            config.scale_factor,
            config.nlevels,
            config.edge_threshold,
            config.first_level,
            config.wta_k,
            config.score_type,
            config.patch_size,
            config.fast_threshold,
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

        let _detector = FeatureDetectorModel::new_orb(OrbConfig::default())?;
        info!("ORB detector created successfully");
        Ok(())
    }
}
