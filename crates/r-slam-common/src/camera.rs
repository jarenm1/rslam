use opencv::{
    Error as CvError,
    core::{CV_64F, Mat, Point2f, Size, Vector, no_array},
    prelude::*,
};

#[derive(thiserror::Error, Debug)]
pub enum CameraError {
    #[error("Invalid camera instrinsic paramater: {0}")]
    InvalidIntrinsics(String),
    #[error("Invalid image dimensions: width={width}, height={height}")]
    InvalidDimensions { width: i32, height: i32 },
    #[error("Failed to create opencv Matrix for instrinsics or distortion coefficients.")]
    OpenCvMatCreationError(#[from] CvError),
    #[error("Camera initilization failed: {0}")]
    InitializationFailed(String),
}

#[derive(Debug, Clone)]
pub struct Camera {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub camera_matrix: Mat,
    pub distortion_coeffs: Vector<f64>,
    window_size: Size,
}

impl Camera {
    // unsafe, look for alternatives, but should be fine.
    pub fn new(
        camera_matrix: Mat,
        distortion_coeffs: Vector<f64>,
        width: i32,
        height: i32,
    ) -> Result<Self, CameraError> {
        if camera_matrix.typ() != CV_64F {
            return Err(CameraError::InvalidIntrinsics(
                "Expected CV_64F".to_string(),
            ));
        }
        if camera_matrix.size()? != Size::new(3, 3) {
            return Err(CameraError::InvalidIntrinsics(
                "Camera Matrix is invalid size. Expected 3x3.".to_string(),
            ));
        }
        Ok(Self {
            fx: *camera_matrix.at_2d(0, 0)?,
            fy: *camera_matrix.at_2d(1, 1)?,
            cx: *camera_matrix.at_2d(0, 2)?,
            cy: *camera_matrix.at_2d(1, 2)?,
            distortion_coeffs,
            camera_matrix,
            window_size: Size::new(width, height),
        })
    }

    pub fn undistort_points(&self, points: &Vector<Point2f>) -> Result<Vector<Point2f>, CvError> {
        let mut undistorted_points = Vector::new();
        opencv::calib3d::undistort_points(
            points,
            &mut undistorted_points,
            &self.camera_matrix,
            &self.distortion_coeffs,
            &no_array(),
            &no_array(),
        )?;
        Ok(undistorted_points)
    }
}
