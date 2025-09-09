use std::{borrow::Cow, marker::PhantomData, ops::Deref};

use opencv::core::{CV_8UC1, Mat, MatTraitConst};
use opencv::prelude::*;
use tracing::{debug, error, info, trace};

pub mod feature_detection;
pub mod keyframe;
pub mod matching;

pub type GrayscaleImage<'a> = MatrixWrapper<'a, GrayscaleImageData>;
pub type BinaryDescriptors<'a> = MatrixWrapper<'a, BinaryDescriptorsData>;

#[derive(Debug, thiserror::Error)]
pub enum MatchingError {
    #[error("OpenCV error: {0}")]
    OpenCv(#[from] opencv::Error),
    #[error("The Mat is not continuous")]
    MatNotContinuous,
    #[error("Invalid Mat type")]
    InvalidMatType,
}

pub struct MatView<'a> {
    mat: Mat,
    _phantom: PhantomData<&'a ()>,
}

impl<'a> Deref for MatView<'a> {
    type Target = Mat;
    fn deref(&self) -> &Self::Target {
        &self.mat
    }
}

#[derive(Debug)]
pub struct MatrixWrapper<'a, T> {
    rows: usize,
    cols: usize,
    data: Cow<'a, [u8]>,
    _marker: PhantomData<T>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrayscaleImageData;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinaryDescriptorsData;

impl<'a, T> MatrixWrapper<'a, T> {
    pub fn from_mat_borrowed(mat: &'a Mat) -> Result<Self, MatchingError> {
        trace!("Starting from_mat_borrowed conversion");

        // Basic Mat validation
        if mat.empty() {
            error!("Mat is empty");
            return Err(MatchingError::InvalidMatType);
        }
        debug!("Mat is not empty");

        if !mat.is_continuous() {
            error!("Mat is not continuous");
            return Err(MatchingError::MatNotContinuous);
        }
        debug!("Mat is continuous");

        if mat.typ() != CV_8UC1 {
            error!("Mat type is {} but expected {}", mat.typ(), CV_8UC1);
            return Err(MatchingError::InvalidMatType);
        }
        debug!("Mat type is correct: CV_8UC1");

        let rows = mat.rows();
        let cols = mat.cols();
        debug!("Mat dimensions: {}x{}", rows, cols);

        // Validate dimensions
        if rows <= 0 || cols <= 0 {
            error!("Invalid dimensions: {}x{}", rows, cols);
            return Err(MatchingError::InvalidMatType);
        }

        let rows = rows as usize;
        let cols = cols as usize;
        let total_bytes = rows * cols;
        debug!("Total bytes to allocate: {}", total_bytes);

        // Safety checks before creating slice
        if total_bytes > isize::MAX as usize {
            error!("Total bytes {} exceeds isize::MAX", total_bytes);
            return Err(MatchingError::InvalidMatType);
        }

        if total_bytes == 0 {
            error!("Total bytes is zero");
            return Err(MatchingError::InvalidMatType);
        }

        let data_slice = unsafe {
            let data_ptr = mat.data();

            // Check for null pointer
            if data_ptr.is_null() {
                error!("Mat data pointer is null");
                return Err(MatchingError::InvalidMatType);
            }
            trace!("Data pointer is valid: {:?}", data_ptr);

            // Verify the Mat's step size matches our expectations
            let step = mat.step1(0).unwrap_or(0);
            debug!("Mat step size: {}, expected: {}", step, cols);
            if step != cols || step == 0 {
                error!("Step size mismatch: {} != {} or step is 0", step, cols);
                return Err(MatchingError::MatNotContinuous);
            }

            std::slice::from_raw_parts(data_ptr, total_bytes)
        };

        info!(
            "Successfully created MatrixWrapper with {}x{} dimensions",
            rows, cols
        );
        Ok(Self {
            rows,
            cols,
            data: Cow::Borrowed(data_slice),
            _marker: PhantomData,
        })
    }

    pub fn into_owned(self) -> MatrixWrapper<'static, T>
    where
        T: 'static,
    {
        MatrixWrapper {
            rows: self.rows,
            cols: self.cols,
            data: Cow::Owned(self.data.into_owned()),
            _marker: PhantomData,
        }
    }

    pub fn as_mat_view<'s>(&'s self) -> Result<MatView<'s>, MatchingError> {
        trace!("Creating MatView from MatrixWrapper");
        debug!("MatrixWrapper dimensions: {}x{}", self.rows, self.cols);

        // Create a new Mat and copy data instead of sharing memory
        let mut mat = Mat::new_rows_cols_with_default(
            self.rows as i32,
            self.cols as i32,
            CV_8UC1,
            opencv::core::Scalar::all(0.0),
        )
        .map_err(|e| MatchingError::OpenCv(e))?;

        // Copy data from our buffer to the Mat
        let mat_data =
            unsafe { std::slice::from_raw_parts_mut(mat.data_mut(), self.rows * self.cols) };
        mat_data.copy_from_slice(&self.data);

        debug!("Successfully created MatView with copied data");
        Ok(MatView {
            mat,
            _phantom: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{CV_8UC1, CV_8UC3, Mat, Size};

    fn init_tracing() -> tracing::subscriber::DefaultGuard {
        let subscriber = tracing_subscriber::FmtSubscriber::builder()
            .with_max_level(tracing::Level::DEBUG)
            .with_test_writer()
            .finish();
        tracing::subscriber::set_default(subscriber)
    }

    fn create_test_image(rows: i32, cols: i32) -> Result<Mat, opencv::Error> {
        debug!("Creating test image with dimensions: {}x{}", rows, cols);

        // Validate input dimensions
        if rows <= 0 || cols <= 0 {
            error!("Invalid test image dimensions: {}x{}", rows, cols);
            return Err(opencv::Error::new(
                opencv::core::StsError,
                "Invalid dimensions",
            ));
        }

        // Create mat with direct initialization - much simpler approach
        let mut mat =
            Mat::new_rows_cols_with_default(rows, cols, CV_8UC1, opencv::core::Scalar::all(0.0))?;

        debug!("Created base mat with new_rows_cols_with_default");

        // Fill with test pattern
        for i in 0..rows {
            for j in 0..cols {
                let value = ((i * cols + j) % 256) as u8;
                *mat.at_2d_mut::<u8>(i, j)? = value;
            }
        }

        // Basic validation
        if mat.empty() {
            error!("Created mat is empty");
            return Err(opencv::Error::new(
                opencv::core::StsError,
                "Created mat is empty",
            ));
        }

        info!("Successfully created test image {}x{}", rows, cols);
        Ok(mat)
    }

    #[test]
    fn test_round_trip_conversion() -> Result<(), Box<dyn std::error::Error>> {
        let _guard = init_tracing();
        info!("Starting test_round_trip_conversion");

        let original_mat = create_test_image(10, 8)?;

        // Convert to wrapper
        let wrapper = MatrixWrapper::<GrayscaleImageData>::from_mat_borrowed(&original_mat)?;

        // Convert back to mat
        let mat_view = wrapper.as_mat_view()?;

        // Verify data integrity
        assert_eq!(original_mat.rows(), mat_view.rows());
        assert_eq!(original_mat.cols(), mat_view.cols());
        assert_eq!(original_mat.typ(), mat_view.typ());

        // Check pixel values
        for i in 0..original_mat.rows() {
            for j in 0..original_mat.cols() {
                let original_val = *original_mat.at_2d::<u8>(i, j)?;
                let converted_val = *mat_view.at_2d::<u8>(i, j)?;
                assert_eq!(
                    original_val, converted_val,
                    "Pixel mismatch at ({}, {}): {} != {}",
                    i, j, original_val, converted_val
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_borrowed_to_owned_conversion() -> Result<(), Box<dyn std::error::Error>> {
        let _guard = init_tracing();
        info!("Starting test_borrowed_to_owned_conversion");

        let original_mat = create_test_image(5, 4)?;

        // Create wrapper from borrowed mat
        let borrowed_wrapper =
            MatrixWrapper::<GrayscaleImageData>::from_mat_borrowed(&original_mat)?;

        // Convert to owned
        let owned_wrapper = borrowed_wrapper.into_owned();

        // Original mat can be dropped here, but owned_wrapper should still be valid
        drop(original_mat);

        // Verify owned wrapper still works
        let mat_view = owned_wrapper.as_mat_view()?;
        assert_eq!(mat_view.rows(), 5);
        assert_eq!(mat_view.cols(), 4);

        Ok(())
    }

    #[test]
    fn test_data_integrity_across_multiple_conversions() -> Result<(), Box<dyn std::error::Error>> {
        let _guard = init_tracing();
        info!("Starting test_data_integrity_across_multiple_conversions");

        let mut original_mat = create_test_image(6, 6)?;

        // Set specific test pattern
        let test_values = [1u8, 42, 128, 200, 255, 17];
        for (idx, &val) in test_values.iter().enumerate() {
            *original_mat.at_2d_mut::<u8>(0, idx as i32)? = val;
        }

        // Multiple round trips
        for iteration in 0..3 {
            let wrapper = MatrixWrapper::<GrayscaleImageData>::from_mat_borrowed(&original_mat)?;
            let owned_wrapper = wrapper.into_owned();
            let mat_view = owned_wrapper.as_mat_view()?;

            // Verify test pattern is preserved
            for (idx, &expected_val) in test_values.iter().enumerate() {
                let actual_val = *mat_view.at_2d::<u8>(0, idx as i32)?;
                assert_eq!(
                    actual_val, expected_val,
                    "Iteration {}: Value mismatch at index {}",
                    iteration, idx
                );
            }

            // Create new mat from view for next iteration
            original_mat = mat_view.clone();
        }

        Ok(())
    }

    #[test]
    fn test_error_handling_scenarios() -> Result<(), Box<dyn std::error::Error>> {
        let _guard = init_tracing();
        info!("Starting test_error_handling_scenarios");

        // Test that continuous Mats work properly
        let continuous_mat = create_test_image(10, 10)?;

        // This should succeed since create_test_image creates continuous mats
        let result = MatrixWrapper::<GrayscaleImageData>::from_mat_borrowed(&continuous_mat);
        assert!(result.is_ok(), "Continuous mat should not fail");

        // Test that our wrapper correctly identifies continuous mats
        assert!(
            continuous_mat.is_continuous(),
            "Test mat should be continuous"
        );

        // Test round-trip conversion works
        let wrapper = result?;
        let mat_view = wrapper.as_mat_view()?;
        assert_eq!(mat_view.rows(), 10);
        assert_eq!(mat_view.cols(), 10);

        Ok(())
    }

    #[test]
    fn test_error_handling_invalid_mat_type() -> Result<(), Box<dyn std::error::Error>> {
        // Create a 3-channel color image (not grayscale)
        let color_mat = unsafe { Mat::new_size(Size::new(10, 10), CV_8UC3)? };

        // Should fail with InvalidMatType error
        let result = MatrixWrapper::<GrayscaleImageData>::from_mat_borrowed(&color_mat);

        match result {
            Err(MatchingError::InvalidMatType) => {
                // Expected error
            }
            _ => panic!("Expected InvalidMatType error"),
        }

        Ok(())
    }

    #[test]
    fn test_memory_safety_after_original_mat_dropped() -> Result<(), Box<dyn std::error::Error>> {
        let _guard = init_tracing();
        info!("Starting test_memory_safety_after_original_mat_dropped");

        let owned_wrapper = {
            let original_mat = create_test_image(4, 4)?;
            let wrapper = MatrixWrapper::<GrayscaleImageData>::from_mat_borrowed(&original_mat)?;
            wrapper.into_owned()
            // original_mat is dropped here
        };

        // Should still be able to use the owned wrapper
        let mat_view = owned_wrapper.as_mat_view()?;
        assert_eq!(mat_view.rows(), 4);
        assert_eq!(mat_view.cols(), 4);

        // Should be able to read data
        let _val = *mat_view.at_2d::<u8>(0, 0)?;

        Ok(())
    }

    #[test]
    fn test_large_image_data_integrity() -> Result<(), Box<dyn std::error::Error>> {
        // Test with larger image to stress test memory handling
        let rows = 100;
        let cols = 100;
        let original_mat = create_test_image(rows, cols)?;

        let wrapper = MatrixWrapper::<GrayscaleImageData>::from_mat_borrowed(&original_mat)?;
        let owned_wrapper = wrapper.into_owned();
        let mat_view = owned_wrapper.as_mat_view()?;

        // Verify a sampling of pixels across the image
        let test_positions = [(0, 0), (50, 50), (99, 99), (25, 75), (75, 25)];

        for &(row, col) in &test_positions {
            let original_val = *original_mat.at_2d::<u8>(row, col)?;
            let converted_val = *mat_view.at_2d::<u8>(row, col)?;
            assert_eq!(
                original_val, converted_val,
                "Large image pixel mismatch at ({}, {})",
                row, col
            );
        }

        Ok(())
    }

    #[test]
    fn test_binary_descriptors_wrapper() -> Result<(), Box<dyn std::error::Error>> {
        // Create a mat that could represent binary descriptors
        let descriptor_mat = create_test_image(10, 32)?; // 10 features, 32 bytes each

        let wrapper = MatrixWrapper::<BinaryDescriptorsData>::from_mat_borrowed(&descriptor_mat)?;
        let mat_view = wrapper.as_mat_view()?;

        assert_eq!(mat_view.rows(), 10);
        assert_eq!(mat_view.cols(), 32);

        // Verify some descriptor data
        let original_val = *descriptor_mat.at_2d::<u8>(5, 16)?;
        let wrapper_val = *mat_view.at_2d::<u8>(5, 16)?;
        assert_eq!(original_val, wrapper_val);

        Ok(())
    }
}
