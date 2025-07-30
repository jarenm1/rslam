use bevy::prelude::*;
use opencv::prelude::*;
use opencv::videoio::VideoCapture;
use std::sync::{Arc, Mutex};
use std::time::Duration;

struct CameraFeedPlugin {
    device_index: i32,
    resolution: (f64, f64),
    fps: f64,
}
impl Plugin for CameraFeedPlugin {
    fn build(&self, app: &mut App) {
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
