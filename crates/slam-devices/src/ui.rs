use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};

use crate::CameraDeviceState;

pub fn draw_device_list(
    mut contexts: EguiContexts,
    mut refresh_event_writer: EventWriter<super::RefreshDevices>,
    mut state: ResMut<CameraDeviceState>,
) {
    todo!()
}
