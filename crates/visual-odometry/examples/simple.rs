use opencv::{core::Mat, features2d::draw_matches, imgcodecs, imgproc};
use visual_odometry::{ORBConfig, VisualOdometry};

fn main() {
    let mut vo = VisualOdometry::new(ORBConfig::default()).unwrap();

    // updtae: images are being loaded properly1
    let image1 = imgcodecs::imread("image1.jpg", imgcodecs::IMREAD_ANYCOLOR).unwrap();
    let mut gimage1 = Mat::default();
    imgproc::cvt_color(&image1, &mut gimage1, imgproc::COLOR_BGR2GRAY, 0).unwrap();

    let image2 = imgcodecs::imread("image2.jpg", imgcodecs::IMREAD_ANYCOLOR).unwrap();
    let mut gimage2 = Mat::default();
    imgproc::cvt_color(&image2, &mut gimage2, imgproc::COLOR_BGR2GRAY, 0).unwrap();
    
    // process frame -> match frame -> matches -> camera pose
}
