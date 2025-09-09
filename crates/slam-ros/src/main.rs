use rclrs::{vendor::example_interfaces, *};

fn main() -> eyre::Result<()> {
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();

    let node = executor.create_node("test node")?;

    let subscription =
        node.create_subscription("topic_name", |msg: example_interfaces::msg::String| {
            println!("Recieved message: {}", msg);
        })?;
}

// subscribe to raw camera feed -> public slam map
