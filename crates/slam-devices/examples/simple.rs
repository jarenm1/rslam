use bevy::prelude::*;
use slam_devices;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = App::new()
        .add_plugins(slam_devices::SlamCameraPlugin)
        .add_plugins(DefaultPlugins)
        .run();
    Ok(())
}
