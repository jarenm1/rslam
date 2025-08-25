use bevy::prelude::*;
use nokhwa::{query, utils::CameraInfo};

mod ui;

pub struct SlamCameraPlugin;

impl Plugin for SlamCameraPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(CameraDeviceState {
            devices: Vec::new(),
            selected_device: None,
        })
        .add_event::<RefreshDevices>()
        .add_systems(Startup, (init_device_query, ui::draw_device_list))
        .add_systems(Update, handle_device_refresh);
    }
}

#[derive(Event)]
struct RefreshDevices;

#[derive(Resource)]
struct CameraDeviceState {
    pub devices: Vec<CameraInfo>,
    pub selected_device: Option<usize>,
}

fn init_device_query(mut state: ResMut<CameraDeviceState>) {
    state.devices = query_devices();
}

fn handle_device_refresh(
    mut refresh_events: EventReader<RefreshDevices>,
    mut state: ResMut<CameraDeviceState>,
) {
    if refresh_events.is_empty() {
        return;
    }

    refresh_events.clear();

    state.devices = query_devices();
}

fn query_devices() -> Vec<CameraInfo> {
    query(nokhwa::utils::ApiBackend::Auto)
        .unwrap_or_default()
        .into_iter()
        .collect()
}
