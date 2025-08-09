use opencv::{
    core::{CV_32FC1, CV_64F, Mat, Mat_, Vector},
    imgcodecs, imgproc,
};
use r_slam_common::camera::Camera;
use visual_odometry::{ORBConfig, VisualOdometry};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut vo = VisualOdometry::new(ORBConfig::default(), get_camera()?).unwrap();

    // updtae: images are being loaded properly1
    let image1 = imgcodecs::imread("image1.jpg", imgcodecs::IMREAD_ANYCOLOR).unwrap();
    let mut gimage1 = Mat::default();
    imgproc::cvt_color(&image1, &mut gimage1, imgproc::COLOR_BGR2GRAY, 0).unwrap();

    let image2 = imgcodecs::imread("image2.jpg", imgcodecs::IMREAD_ANYCOLOR).unwrap();
    let mut gimage2 = Mat::default();
    imgproc::cvt_color(&image2, &mut gimage2, imgproc::COLOR_BGR2GRAY, 0).unwrap();

    // process frame -> match frame -> matches -> camera pose
    //
    let frame1 = vo.process_frame(gimage1)?;
    let frame2 = vo.process_frame(gimage2)?;

    let results = vo.frame_match(frame1, frame2, 0.7, 50.0)?; // DMatch

    Ok(())
}

// My camera instrinsics
fn get_camera() -> Result<Camera, Box<dyn std::error::Error>> {
    let width = 1920;
    let height = 1080;

    let matrix = unsafe { Mat::new_rows_cols(3, 3, CV_64F)? };

    let mut camera_matrix: Mat_<f64> = matrix.try_into()?;
    *camera_matrix.at_2d_mut(0, 0)? = 1.1;
    *camera_matrix.at_2d_mut(1, 1)? = 1.1;
    *camera_matrix.at_2d_mut(0, 2)? = 1.1;
    *camera_matrix.at_2d_mut(2, 2)? = 1.1;

    let distortion_coeffs = Vector::from_slice(&[]);

    Ok(Camera::new(
        camera_matrix.into_untyped(),
        distortion_coeffs,
        width,
        height,
    )?)
}
