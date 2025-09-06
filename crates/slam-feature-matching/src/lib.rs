use std::{borrow::Cow, marker::PhantomData, ops::Deref};

use opencv::core::{CV_8UC1, Mat, Mat_AUTO_STEP, MatTraitConst};
pub mod edge_detection;
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
        if !mat.is_continuous() {
            return Err(MatchingError::MatNotContinuous);
        }
        if mat.typ() != CV_8UC1 {
            return Err(MatchingError::InvalidMatType);
        }

        let rows = mat.rows() as usize;
        let cols = mat.cols() as usize;
        let total_bytes = rows * cols;

        let data_slice = unsafe {
            let data_ptr = mat.data();
            std::slice::from_raw_parts(data_ptr, total_bytes)
        };

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
        let mat_header = unsafe {
            let data_ptr = self.data.as_ptr() as *mut std::ffi::c_void;

            Mat::new_rows_cols_with_data_unsafe(
                self.rows as i32,
                self.cols as i32,
                CV_8UC1,
                data_ptr,
                Mat_AUTO_STEP,
            )?
        };

        Ok(MatView {
            mat: mat_header,
            _phantom: PhantomData,
        })
    }
}