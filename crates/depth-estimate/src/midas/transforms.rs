use opencv::core::{Mat, MatExprTraitConst, MatTraitConst};

#[derive(Debug, thiserror::Error)]
pub enum TransformError {
    #[error("Failed to Pad or Resize: {0}")]
    PadAndResize(String),
    #[error("Failed to normalize image")]
    Normalize(),
    #[error(transparent)]
    OpenCV(#[from] opencv::Error),
}

pub fn resize_image(image: Mat, new_height: i32, new_width: i32) -> Result<Mat, TransformError> {
    use opencv::core::Size;
    use opencv::imgproc::{INTER_AREA, resize};

    let mut resized_image = Mat::default();
    resize(
        &image,
        &mut resized_image,
        Size::new(new_width, new_height),
        0.0,
        0.0,
        INTER_AREA,
    )?;
    Ok(resized_image)
}

pub fn pad_image(
    resized_image: Mat,
    target_height: i32,
    target_width: i32,
) -> Result<Mat, TransformError> {
    use opencv::core::{Mat, Rect};

    let mut padded_image =
        Mat::zeros(target_height, target_width, resized_image.typ())?.to_mat()?;
    let top = (target_height - resized_image.rows()) / 2;
    let left = (target_width - resized_image.cols()) / 2;
    let roi = Rect::new(left, top, resized_image.cols(), resized_image.rows());
    let mut roi_mat = Mat::roi_mut(&mut padded_image, roi)?;
    resized_image.copy_to(&mut roi_mat)?;
    Ok(padded_image)
}

pub fn normalize(
    image: Mat,
    mean: opencv::core::Scalar,
    std: opencv::core::Scalar,
) -> Result<Mat, TransformError> {
    use opencv::core::divide2;
    use opencv::core::subtract;
    use opencv::core::{CV_32F, Mat};

    let mut float_image = Mat::default();
    image.convert_to(&mut float_image, CV_32F, 1.0 / 255.0, 0.0)?;
    let mut normalized_image = Mat::default();
    subtract(
        &float_image,
        &mean,
        &mut normalized_image,
        &Mat::default(),
        -1,
    )?;
    let mut processed_image = Mat::default();
    divide2(&normalized_image, &std, &mut processed_image, 1.0, -1)?;
    Ok(processed_image)
}

pub trait ImageTransform {
    fn apply(&self, image: Mat) -> Result<Mat, TransformError>;
}

pub struct MidasTransform {
    pub target_height: i32,
    pub target_width: i32,
    pub mean: opencv::core::Scalar,
    pub std: opencv::core::Scalar,
}

impl ImageTransform for MidasTransform {
    fn apply(&self, image: Mat) -> Result<Mat, TransformError> {
        let resized = resize_image(image, self.target_height, self.target_width)?;
        let padded = pad_image(resized, self.target_height, self.target_width)?;
        normalize(padded, self.mean, self.std)
    }
}
