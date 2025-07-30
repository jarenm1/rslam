use opencv::core::{KeyPoint, Mat, Vector};

pub struct Frame {
    pub id: usize,
    pub image: Mat,
    pub keypoints: Vector<KeyPoint>,
    pub descriptors: Mat,
}

impl Frame {
    pub fn new(id: usize, image: Mat, keypoints: Vector<KeyPoint>, descriptors: Mat) -> Self {
        Self {
            id,
            image,
            keypoints,
            descriptors,
        }
    }
}
