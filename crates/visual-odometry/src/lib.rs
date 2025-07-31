use camera::Camera;
use frame::Frame;
use opencv::calib3d::{RANSAC, find_essential_mat, recover_pose};
use opencv::core::{DMatch, KeyPointTraitConst, Mat, Point2d, Point2f, Ptr, Vector, no_array};
use opencv::features2d::{BFMatcher, FlannBasedMatcher, ORB, ORB_ScoreType, draw_matches_def};
use opencv::prelude::*;

pub mod camera;
mod frame;

#[derive(thiserror::Error, Debug)]
pub enum OdometryError {
    #[error("OpenCV error: {0}")]
    OpenCv(#[from] opencv::Error),
    #[error("Not enough points to estimate pose")]
    NotEnoughPoints,
}

pub struct VisualOdometry {
    config: ORBConfig,
    orb: Ptr<ORB>,
    matcher: Ptr<BFMatcher>,
    frame_id: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct ORBConfig {
    pub nfeatures: i32,
    pub scale_factor: f32,
    pub nlevels: i32,
    pub edge_threshold: i32,
    pub first_level: i32,
    pub wta_k: i32,
    pub score_type: ORB_ScoreType,
    pub patch_size: i32,
    pub fast_threshold: i32,
}

impl Default for ORBConfig {
    /// Creates a default ORB configuration.
    ///
    /// These are the default values:
    /// - `nfeatures`: 1000
    /// - `scale_factor`: 1.2
    /// - `nlevels`: 8
    /// - `edge_threshold`: 31
    /// - `first_level`: 0
    /// - `wta_k`: 2
    /// - `score_type`: `ORB_ScoreType::HARRIS_SCORE`
    /// - `patch_size`: 31
    /// - `fast_threshold`: 20
    fn default() -> Self {
        Self {
            nfeatures: 500,
            scale_factor: 1.2,
            nlevels: 8,
            edge_threshold: 31,
            first_level: 0,
            wta_k: 2,
            score_type: ORB_ScoreType::FAST_SCORE,
            patch_size: 31,
            fast_threshold: 20,
        }
    }
}

impl VisualOdometry {
    pub fn new(config: ORBConfig) -> Result<Self, OdometryError> {
        let orb = ORB::create(
            config.nfeatures,
            config.scale_factor,
            config.nlevels,
            config.edge_threshold,
            config.first_level,
            config.wta_k,
            config.score_type,
            config.patch_size,
            config.fast_threshold,
        )?;
        let matcher = BFMatcher::create(opencv::core::NORM_HAMMING, true)?;

        Ok(Self {
            config,
            orb,
            matcher,
            frame_id: 0,
        })
    }

    #[inline]
    pub fn process_frame(&mut self, image: Mat) -> Result<Frame, OdometryError> {
        let mut keypoints: Vector<opencv::core::KeyPoint> = Vector::new();
        let mut descriptors = Mat::default();
        self.orb
            .detect_and_compute(&image, &no_array(), &mut keypoints, &mut descriptors, false)
            .unwrap();

        Ok(Frame::new(self.frame_id, image, keypoints, descriptors))
    }

    #[inline]
    pub fn frame_match(
        self,
        frame1: Frame,
        frame2: Frame,
    ) -> Result<Vector<DMatch>, OdometryError> {
        let mut matches = Vector::new();
        // YOU FUCKING HAVE TO USE TRAIN MATCH BTW. FOUND THIS SHIT IN A GITHUB ISSUE.
        self.matcher
            .train_match(
                &frame1.descriptors.clone(),
                &frame2.descriptors.clone(),
                &mut matches,
                &no_array(),
            )
            .unwrap();

        println!("match len: {}", matches.len());

        Ok(matches)
    }
}
