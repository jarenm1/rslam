use super::camera::Camera;
use opencv::{
    Error as CvError,
    core::{CV_32FC1, Mat, Point3f, Vector},
    prelude::*,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BackprojectionError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error(transparent)]
    OpenCv(#[from] CvError),
}

#[derive(Debug, Clone, Copy)]
pub struct BackprojectionConfig {
    pub depth_min: f32,
    pub depth_max: Option<f32>,
    pub stride: usize,
}

// TODO : Consider removing builder pattern for simplier just new()
impl BackprojectionConfig {
    pub fn new(depth_min: f32) -> Self {
        Self {
            depth_min,
            depth_max: None,
            stride: 1,
        }
    }

    pub fn with_depth_max(mut self, depth_max: f32) -> Self {
        self.depth_max = Some(depth_max);
        self
    }

    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride.max(1);
        self
    }
}

// TODO : Replace with sensible defaults + comment about defaults..
impl Default for BackprojectionConfig {
    fn default() -> Self {
        Self {
            depth_min: 1.0,
            depth_max: Some(10.0),
            stride: 1,
        }
    }
}

/// Convert a CV_32FC1 depth map (meters) to a point cloud in the camera frame.
///
/// - Depth map layout: `v` = row (y), `u` = col (x)
/// - Output point coordinates follow the pinhole model:
///   x = (u - cx) / fx * z, y = (v - cy) / fy * z, z = depth
pub fn depth_map_to_point_cloud(
    depth_map: &Mat,
    intrinsics: &Camera,
    config: Option<BackprojectionConfig>,
) -> Result<Vector<Point3f>, BackprojectionError> {
    if depth_map.empty() {
        return Err(BackprojectionError::InvalidInput(
            "Depth map must be non-empty".to_string(),
        ));
    }
    if depth_map.typ() != CV_32FC1 as i32 {
        return Err(BackprojectionError::InvalidInput(
            "Depth map must be CV_32FC1 (f32, single channel)".to_string(),
        ));
    }

    let cfg = config.unwrap_or_else(|| BackprojectionConfig::new(0.0));
    let rows = depth_map.rows();
    let cols = depth_map.cols();

    if rows <= 0 || cols <= 0 {
        return Err(BackprojectionError::InvalidInput(format!(
            "Invalid dimensions rows={} cols={}",
            rows, cols
        )));
    }

    let rows_usize = rows as usize;
    let cols_usize = cols as usize;

    let mut point_cloud = Vector::<Point3f>::new();

    // Stride will panic if less than 1.
    let stride = cfg.stride.max(1);

    for v in (0..rows_usize).step_by(stride) {
        for u in (0..cols_usize).step_by(stride) {
            let depth_value = *depth_map.at_2d::<f32>(v as i32, u as i32)?;

            if !depth_value.is_finite() {
                continue;
            }
            if depth_value <= cfg.depth_min {
                continue;
            }
            if let Some(max_d) = cfg.depth_max {
                if depth_value > max_d {
                    continue;
                }
            }

            let fx = intrinsics.fx as f32;
            let fy = intrinsics.fy as f32;
            let cx = intrinsics.cx as f32;
            let cy = intrinsics.cy as f32;

            // Pinhole back-projection TODO : FIGURE OUT WHAT THIS MEANS>???
            let z = depth_value;
            let x = ((u as f32 - cx) / fx) * z;
            let y = ((v as f32 - cy) / fy) * z;

            point_cloud.push(Point3f::new(x, y, z));
        }
    }

    Ok(point_cloud)
}

/// Apply a 4x4 homogeneous transform to a point cloud.
/// Accepts CV_32F or CV_64F matrices. Returns points in the target frame.
pub fn transform_point_cloud(
    points: &Vector<Point3f>,
    transform: &Mat,
) -> Result<Vector<Point3f>, BackprojectionError> {
    if transform.rows() != 4 || transform.cols() != 4 {
        return Err(BackprojectionError::InvalidInput(
            "Transform must be 4x4".to_string(),
        ));
    }

    let mut m: [f32; 16] = [0.0; 16];
    match transform.typ() {
        t if t == opencv::core::CV_32F => {
            for r in 0..4 {
                for c in 0..4 {
                    m[r * 4 + c] = *transform.at_2d::<f32>(r as i32, c as i32)?;
                }
            }
        }
        t if t == opencv::core::CV_64F => {
            for r in 0..4 {
                for c in 0..4 {
                    m[r * 4 + c] = *transform.at_2d::<f64>(r as i32, c as i32)? as f32;
                }
            }
        }
        _ => {
            return Err(BackprojectionError::InvalidInput(
                "Transform must be CV_32F or CV_64F".to_string(),
            ));
        }
    }

    let mut out = Vector::<Point3f>::new();
    out.reserve(points.len());
    for i in 0..points.len() {
        let p = points.get(i)?;
        let x = p.x;
        let y = p.y;
        let z = p.z;

        let xp = m[0] * x + m[1] * y + m[2] * z + m[3];
        let yp = m[4] * x + m[5] * y + m[6] * z + m[7];
        let zp = m[8] * x + m[9] * y + m[10] * z + m[11];
        let wp = m[12] * x + m[13] * y + m[14] * z + m[15];

        if wp != 0.0 {
            out.push(Point3f::new(xp / wp, yp / wp, zp / wp));
        } else {
            out.push(Point3f::new(xp, yp, zp));
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{CV_8UC1, CV_64F, Scalar};

    fn make_camera(fx: f64, fy: f64, cx: f64, cy: f64, width: i32, height: i32) -> Camera {
        let mut k = Mat::new_rows_cols_with_default(3, 3, CV_64F, Scalar::all(0.0)).unwrap();
        *k.at_2d_mut::<f64>(0, 0).unwrap() = fx;
        *k.at_2d_mut::<f64>(1, 1).unwrap() = fy;
        *k.at_2d_mut::<f64>(0, 2).unwrap() = cx;
        *k.at_2d_mut::<f64>(1, 2).unwrap() = cy;

        let dist = Vector::<f64>::new();
        Camera::new(k, dist, width, height).unwrap()
    }

    #[test]
    fn test_backprojection_basic() {
        // Depth map 2x2 with values in meters
        let mut depth = Mat::new_rows_cols_with_default(2, 2, CV_32FC1, Scalar::all(0.0)).unwrap();
        *depth.at_2d_mut::<f32>(0, 0).unwrap() = 1.0; // (u=0,v=0)
        *depth.at_2d_mut::<f32>(0, 1).unwrap() = 2.0; // (u=1,v=0)
        *depth.at_2d_mut::<f32>(1, 0).unwrap() = 3.0; // (u=0,v=1)
        *depth.at_2d_mut::<f32>(1, 1).unwrap() = 4.0; // (u=1,v=1)

        // Simple intrinsics so x=u*z, y=v*z
        let cam = make_camera(1.0, 1.0, 0.0, 0.0, 2, 2);
        let cfg = BackprojectionConfig::new(0.0);
        let cloud = depth_map_to_point_cloud(&depth, &cam, Some(cfg)).unwrap();

        assert_eq!(cloud.len(), 4);
        let p00 = cloud.get(0).unwrap();
        assert!(
            (p00.x - 0.0).abs() < 1e-6 && (p00.y - 0.0).abs() < 1e-6 && (p00.z - 1.0).abs() < 1e-6
        );
        let p10 = cloud.get(1).unwrap();
        assert!(
            (p10.x - 2.0).abs() < 1e-6 && (p10.y - 0.0).abs() < 1e-6 && (p10.z - 2.0).abs() < 1e-6
        );
        let p01 = cloud.get(2).unwrap();
        assert!(
            (p01.x - 0.0).abs() < 1e-6 && (p01.y - 3.0).abs() < 1e-6 && (p01.z - 3.0).abs() < 1e-6
        );
        let p11 = cloud.get(3).unwrap();
        assert!(
            (p11.x - 4.0).abs() < 1e-6 && (p11.y - 4.0).abs() < 1e-6 && (p11.z - 4.0).abs() < 1e-6
        );
    }

    #[test]
    fn test_backprojection_min_max_and_stride() {
        // 3x3 grid with increasing depths
        let mut depth = Mat::new_rows_cols_with_default(3, 3, CV_32FC1, Scalar::all(0.0)).unwrap();
        let vals = [[0.0f32, 0.6, 5.0], [1.0, 2.0, 3.0], [f32::NAN, 10.0, 0.4]];
        for v in 0..3 {
            for u in 0..3 {
                *depth.at_2d_mut::<f32>(v, u).unwrap() = vals[v as usize][u as usize];
            }
        }

        let cam = make_camera(1.0, 1.0, 0.0, 0.0, 3, 3);
        // Keep values in (0.5, 5.0], stride 2 should pick (0,0),(0,2),(2,0),(2,2)
        let cfg = BackprojectionConfig::new(0.5)
            .with_depth_max(5.0)
            .with_stride(2);
        let cloud = depth_map_to_point_cloud(&depth, &cam, Some(cfg)).unwrap();
        // From selected positions, only (0,2)=5.0 and (2,2)=0.4(too small) remain; (0,0)=0.0 filtered, (2,0)=NaN filtered
        assert_eq!(cloud.len(), 1);
        let p = cloud.get(0).unwrap(); // (u=2,v=0,z=5)
        assert!((p.x - 10.0).abs() < 1e-6 && (p.y - 0.0).abs() < 1e-6 && (p.z - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_invalid_type_rejected() {
        let depth = Mat::new_rows_cols_with_default(2, 2, CV_8UC1, Scalar::all(0.0)).unwrap();
        let cam = make_camera(1.0, 1.0, 0.0, 0.0, 2, 2);
        let err = depth_map_to_point_cloud(&depth, &cam, None).unwrap_err();
        match err {
            BackprojectionError::InvalidInput(msg) => assert!(msg.contains("CV_32FC1")),
            _ => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn test_transform_point_cloud_identity_and_translation() {
        let mut pts = Vector::<Point3f>::new();
        pts.push(Point3f::new(1.0, 2.0, 3.0));

        // Identity 4x4 (CV_32F)
        let t_id = Mat::eye(4, 4, opencv::core::CV_32F)
            .unwrap()
            .to_mat()
            .unwrap();
        let out_id = transform_point_cloud(&pts, &t_id).unwrap();
        let p = out_id.get(0).unwrap();
        assert!((p.x - 1.0).abs() < 1e-6 && (p.y - 2.0).abs() < 1e-6 && (p.z - 3.0).abs() < 1e-6);

        // Translation by +1 in x (CV_64F)
        let mut t = Mat::eye(4, 4, CV_64F).unwrap().to_mat().unwrap();
        *t.at_2d_mut::<f64>(0, 3).unwrap() = 1.0;
        let out = transform_point_cloud(&pts, &t).unwrap();
        let p2 = out.get(0).unwrap();
        assert!(
            (p2.x - 2.0).abs() < 1e-6 && (p2.y - 2.0).abs() < 1e-6 && (p2.z - 3.0).abs() < 1e-6
        );
    }
}
