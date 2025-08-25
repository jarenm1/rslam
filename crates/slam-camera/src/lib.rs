use opencv::prelude::*;

pub struct Camera {
    device: i32,
    intrinsics: Option<Mat>,
}

impl Camera {
    fn new(device: i32) -> Self {
        Self {
            device,
            intrinsics: None,
        }
    }
    fn change_device(&mut self, new_device: i32) {
        self.device = new_device;
    }
    fn change_intrinsics(&mut self, new_intrinsics: Mat) {
        self.intrinsics = Some(new_intrinsics);
    }
}
