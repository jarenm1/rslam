use opencv::core::{CV_8UC1, Mat, Mat_AUTO_STEP, MatTraitConst};

use crate::matching::Match;

pub mod edge_detection;
pub mod matching;

#[derive(Debug, thiserror::Error)]
pub enum MatchingError {
    #[error("OpenCV error: {0}")]
    OpenCv(#[from] opencv::Error),
    #[error("The Mat is not continuous")]
    MatNotContinuous,
    #[error("Invalid Mat type")]
    InvalidMatType,
}

pub struct BinaryDescriptors {
    pub matrix: nalgebra::DMatrix<u8>,
}

impl<'a> TryFrom<&'a BinaryDescriptors> for Mat {
    type Error = MatchingError;
    fn try_from(value: &'a BinaryDescriptors) -> Result<Self, Self::Error> {
        unsafe {
            let rows = value.matrix.nrows() as i32;
            let cols = value.matrix.ncols() as i32;

            let data_ptr = value.matrix.as_slice().as_ptr() as *mut std::ffi::c_void;

            let mat =
                Mat::new_rows_cols_with_data_unsafe(rows, cols, CV_8UC1, data_ptr, Mat_AUTO_STEP)?;

            Ok(mat)
        }
    }
}

impl<'a> TryFrom<&'a Mat> for BinaryDescriptors {
    type Error = MatchingError;

    fn try_from(mat: &'a Mat) -> Result<Self, Self::Error> {
        if !mat.is_continuous() {
            return Err(MatchingError::MatNotContinuous);
        }

        if mat.typ() != CV_8UC1 {
            return Err(MatchingError::InvalidMatType);
        }

        let rows = mat.rows() as usize;
        let cols = mat.cols() as usize;
        let total_bytes = mat.total() * mat.elem_size()?;

        let data_vec: Vec<u8> = unsafe {
            let data_ptr = mat.data();
            let slice = std::slice::from_raw_parts(data_ptr, total_bytes);
            slice.to_vec()
        };

        let matrix = nalgebra::DMatrix::from_vec(rows, cols, data_vec);

        Ok(BinaryDescriptors { matrix })
    }
}

pub struct GrayscaleImage {
    pub matrix: nalgebra::DMatrix<u8>,
}

impl<'a> TryFrom<&'a GrayscaleImage> for Mat {
    type Error = MatchingError;
    fn try_from(value: &'a GrayscaleImage) -> Result<Self, Self::Error> {
        unsafe {
            let rows = value.matrix.nrows() as i32;
            let cols = value.matrix.ncols() as i32;

            let data_ptr = value.matrix.as_slice().as_ptr() as *mut std::ffi::c_void;

            let mat =
                Mat::new_rows_cols_with_data_unsafe(rows, cols, CV_8UC1, data_ptr, Mat_AUTO_STEP)?;

            Ok(mat)
        }
    }
}

pub fn filter_matches(matches: Vec<Vec<Match>>, ratio: f32) -> Vec<Vec<Match>> {

    matches
        .iter()
        .filter_map(|m_group| {
            
            if let [m, n, ..] = m_group.as_slice() {
                if m.distance < ratio * n
            }
        })

}
