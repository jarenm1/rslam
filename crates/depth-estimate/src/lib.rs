use std::path::PathBuf;

use opencv::{
    core::{
        CV_32F, CV_32FC1, Mat, Mat_, MatExprTraitConst, MatTraitConst, Scalar, Size, Vector,
        no_array,
    },
    imgcodecs,
    imgproc::{self, COLOR_BGR2RGB, COLOR_BGRA2RGB, INTER_AREA},
};
use ort::{
    session::{Session, SessionOutputs, builder::GraphOptimizationLevel},
    value::Value,
};

#[derive(Debug, thiserror::Error)]
pub enum EstimateError {
    #[error("ORT error")]
    OrtError(#[from] ort::Error),
    #[error("Resize error")]
    OpenCVError(#[from] opencv::Error),
    #[error("Conversion error")]
    ConversionError,
}

pub struct DepthEstimate {
    model: Session,
}

impl DepthEstimate {
    pub fn new(config: DepthEstimateConfig) -> Result<Self, EstimateError> {
        let mut model = Session::builder()?
            .with_optimization_level(config.optimization_level)?
            .with_intra_threads(config.intra_threads)?
            .commit_from_file(config.file_path.clone())?;
        Ok(Self { model })
    }
    // should be model agnostic in future
    #[inline]
    pub fn estimate(&mut self, image: Mat) -> Result<Mat, EstimateError> {
        // TODO: move resizing logic
        let input_tensor_values = preprocess_mat_to_ort_tensor(&image, 384, 384)?;

        let shape = [1, 3, 384, 384];
        let input_tensor = Value::from_array((shape, input_tensor_values))?;

        let mut outputs = self.model.run(ort::inputs![input_tensor])?;
        save_depth_map(&mut outputs)
    }
}

pub struct DepthEstimateConfig {
    optimization_level: GraphOptimizationLevel,
    intra_threads: usize,
    file_path: PathBuf,
}

impl DepthEstimateConfig {
    pub fn new(
        optimization_level: GraphOptimizationLevel,
        intra_threads: usize,
        file_path: PathBuf,
    ) -> Self {
        Self {
            optimization_level,
            intra_threads,
            file_path: file_path.to_path_buf(),
        }
    }
}

fn resize_and_pad(
    input_image: &Mat,
    target_height: i32,
    target_width: i32,
) -> Result<Mat, EstimateError> {
    let mut rgb_image = Mat::default();
    match input_image.channels() {
        4 => {
            imgproc::cvt_color(&input_image, &mut rgb_image, COLOR_BGRA2RGB, 0)?;
        }
        3 => {
            imgproc::cvt_color(&input_image, &mut rgb_image, COLOR_BGR2RGB, 0)?;
        }
        _ => return Err(EstimateError::ConversionError),
    }

    let original_height = rgb_image.rows();
    let original_width = rgb_image.cols();

    let aspect_ratio = original_width as f32 / original_height as f32;

    let (new_width, new_height) = if aspect_ratio > 1.0 {
        (target_width, (target_width as f32 / aspect_ratio) as i32)
    } else {
        ((target_height as f32 * aspect_ratio) as i32, target_height)
    };

    let mut resized_image = Mat::default();
    imgproc::resize(
        &rgb_image,
        &mut resized_image,
        Size::new(new_width, new_height),
        0.0,
        0.0,
        INTER_AREA,
    )?;

    let mut padded_image = Mat::zeros(target_height, target_width, rgb_image.typ())?.to_mat()?;

    let top = (target_height - new_height) / 2;
    let left = (target_width - new_width) / 2;
    let roi = opencv::core::Rect::new(left, top, new_width, new_height);
    let mut roi_mat = Mat::roi_mut(&mut padded_image, roi)?;
    resized_image.copy_to(&mut roi_mat)?;

    Ok(padded_image)
}

fn preprocess_mat_to_ort_tensor(
    input_image: &Mat,
    target_height: i32,
    target_width: i32,
) -> Result<Vec<f32>, EstimateError> {
    let rgb_image = resize_and_pad(input_image, target_height, target_width)?;

    let mut float_image = Mat::default();
    rgb_image.convert_to(&mut float_image, CV_32F, 1.0 / 255.0, 0.0)?;

    let mut normalized_image = Mat::default();
    let mean = opencv::core::Scalar::from_array([0.485, 0.456, 0.406, 0.0]);
    let std = opencv::core::Scalar::from_array([0.299, 0.224, 0.225, 0.0]);
    opencv::core::subtract(&float_image, &mean, &mut normalized_image, &no_array(), -1)?;
    let mut processed_image = Mat::default();
    opencv::core::divide2(&normalized_image, &std, &mut processed_image, 1.0, -1)?;

    let mut input_tensor_values =
        vec![0.0f32; 1 * 3 * target_height as usize * target_width as usize];
    let mut channels = Vector::<Mat>::new();
    opencv::core::split(&processed_image, &mut channels)?;

    let mut index = 0;
    for c in 0..3 {
        for h in 0..target_height {
            for w in 0..target_width {
                input_tensor_values[index] = *channels.get(c as usize)?.at_2d(h, w)?;
                index += 1;
            }
        }
    }

    Ok(input_tensor_values)
}

fn save_depth_map(outputs: &mut SessionOutputs) -> Result<Mat, EstimateError> {
    // Process output (example: save depth map)
    let output_tensor = outputs[0].try_extract_array_mut::<f32>()?;
    let output_shape = output_tensor.shape();
    println!("Output tensor shape: {:?}", output_shape);

    // Convert output to depth map and save
    let output_data = output_tensor.view();
    let mut depth_mat = Mat::new_rows_cols_with_default(386, 386, CV_32FC1, Scalar::all(0.0))?;
    let mut depth_mat_ = Mat_::try_from(depth_mat)?;
    for h in 0..386 {
        for w in 0..386 {
            depth_mat_.at_row_mut(h)?[w as usize] = output_tensor[[0, h as usize, w as usize]];
        }
    }
    let depth_map: Mat = depth_mat_.into();
    let mut normalized_depth = Mat::default();
    opencv::core::normalize(
        &depth_map,
        &mut normalized_depth,
        0.0,
        255.0,
        opencv::core::NORM_MINMAX,
        -1,
        &no_array(),
    )?;
    imgcodecs::imwrite("depth_map.png", &normalized_depth, &Vector::new())?;
    Ok(normalized_depth)
}
