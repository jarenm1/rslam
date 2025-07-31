use opencv::{
    Error as CvError,
    core::{Mat, Point2f, Size, TermCriteria, TermCriteria_Type, Vector},
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
    pub distortion_coeffs: Mat,
    window_size: Size,
}

impl Camera {
    pub fn new(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        width: i32,
        height: i32,
    ) -> Result<Self, CameraError> {
        let camera_matrix = Mat::from_slice_2d(&[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])?;
        Ok(Self {
            fx,
            fy,
            cx,
            cy,
            camera_matrix,
            distortion_coeffs: Mat::default(),
            window_size: Size { width, height },
        })
    }

    pub fn undistort_points(&self, points: &Vector<Point2f>) -> Result<Vector<Point2f>, CvError> {
        let mut undistorted_points = Vector::new();
        opencv::calib3d::undistort_points(
            points,
            &mut undistorted_points,
            &self.camera_matrix,
            &self.distortion_coeffs,
            &Mat::default(),
            &Mat::default(),
        )?;
        Ok(undistorted_points)
    }
}

impl Default for Camera {
    /// Creates a default camera configuration for testing.
    ///
    /// These are the default values:
    /// - `fx`: 554.3827
    /// - `fy`: 554.3827
    /// - `cx`: 320.0
    /// - `cy`: 240.0
    /// - `width`: 640
    /// - `height`: 480
    fn default() -> Self {
        let (fx, fy, cx, cy, width, height) = (554.3827, 554.3827, 320.0, 240.0, 640, 480);
        let camera_matrix =
            Mat::from_slice_2d(&[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]).unwrap();
        Self {
            fx,
            fy,
            cx,
            cy,
            camera_matrix,
            distortion_coeffs: Mat::default(),
            window_size: Size { width, height },
        }
    }
}
