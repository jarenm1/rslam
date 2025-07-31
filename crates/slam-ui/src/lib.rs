use bevy::prelude::*;
use nokhwa::NokhwaError;
use nokhwa::utils::CameraInfo;
use opencv::prelude::*;
use opencv::videoio::VideoCapture;
use std::sync::{Arc, Mutex};
use std::time::Duration;

mod init;

/// Platform Notes:
///
/// Linux: Ensure your user has permission to access /dev/video* (add user to the video group).
/// Windows: nokhwa uses UVC or DirectShow; ensure drivers are installed.
/// macOS: nokhwa uses AVFoundation; you may need to grant camera permissions.
pub struct CameraFeedPlugin {
    device_index: i32,
    resolution: (f64, f64),
    fps: f64,
}
impl Plugin for CameraFeedPlugin {
    fn build(&self, app: &mut App) {
        nokhwa::nokhwa_initialize(|_| {});

        let mut capture =
            opencv::videoio::VideoCapture::new(self.device_index, opencv::videoio::CAP_ANY)
                .unwrap();

        if !opencv::videoio::VideoCapture::is_opened(&capture).expect("Failed to check camera.") {
            panic!("Unable to open camera device {}", self.device_index);
        }

        capture
            .set(opencv::videoio::CAP_PROP_FRAME_WIDTH, self.resolution.0)
            .expect("Failed to set width");

        capture
            .set(opencv::videoio::CAP_PROP_FRAME_HEIGHT, self.resolution.1)
            .expect("Failed to set height");

        capture
            .set(opencv::videoio::CAP_PROP_FPS, self.fps)
            .expect("failed to set FPS");

        let delay = Duration::from_secs_f64(1000.0 / self.fps / 1000.0);

        app.insert_resource(CameraResource {
            capture: Arc::new(Mutex::new(capture)),
            delay,
        })
        .add_systems(Startup, setup_camera)
        .add_systems(Update, update_camera_feed);
    }
}

impl CameraFeedPlugin {
    // We want the app to hang until a camera is chosen.
    pub fn select_camera(&mut self, selection: i32) {
        self.device_index = selection;
    }
    pub fn list_cameras() -> Result<Vec<CameraInfo>, NokhwaError> {
        let cameras = nokhwa::query(nokhwa::utils::ApiBackend::Auto)?;
        Ok(cameras)
    }
}

#[derive(Debug, Resource)]
struct CameraResource {
    capture: Arc<Mutex<VideoCapture>>,
    delay: Duration,
}

#[derive(Debug, Component)]
struct CameraSprite;

#[derive(Debug, Clone)]
struct Camera {}

fn setup_camera(mut commands: Commands) {
    commands.spawn((
        Sprite { ..default() },
        Transform::from_scale(Vec3::new(1.0, 1.0, 1.0)),
        CameraSprite,
    ));
}

fn update_camera_feed(
    camera: Res<CameraResource>,
    mut images: ResMut<Assets<Image>>,
    mut query: Query<&mut Sprite, With<CameraSprite>>,
    time: Res<Time>,
) {
}
