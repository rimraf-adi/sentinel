use sentinel_agent::client::{AgentEvent, EventBuffer};
use std::io::Write;

#[test]
fn test_agent_buffer_high_throughput() {
    let temp_dir = std::env::temp_dir().join(format!("agent_stress_{}", uuid::Uuid::new_v4()));
    let buf_path = temp_dir.join("high_throughput.jsonl");
    let buffer = EventBuffer::new(&buf_path);

    let event_count = 1000;
    for i in 0..event_count {
        let event = AgentEvent::telemetry(
            "gpu-box-heavy",
            serde_json::json!({
                "step": i,
                "gpu_temp": 50.0 + (i as f32 * 0.01),
                "vram_used": 1024 * 1024 * i
            }),
        );
        buffer.push(&event).expect("push event");
    }

    assert_eq!(buffer.count(), event_count);

    let drained = buffer.drain().expect("drain buffer");
    assert_eq!(drained.len(), event_count);
    assert_eq!(drained[0].payload["step"], 0);
    assert_eq!(drained[event_count - 1].payload["step"], event_count - 1);

    // Buffer cleared after drain
    assert_eq!(buffer.count(), 0);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_agent_buffer_corrupted_line_recovery() {
    let temp_dir = std::env::temp_dir().join(format!("agent_corrupt_{}", uuid::Uuid::new_v4()));
    let buf_path = temp_dir.join("corrupted.jsonl");
    let buffer = EventBuffer::new(&buf_path);

    // Write a valid event
    let event1 = AgentEvent::telemetry("node-1", serde_json::json!({"step": 1}));
    buffer.push(&event1).expect("push event 1");

    // Manually inject a corrupt / truncated JSON line into the file
    {
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&buf_path)
            .expect("open file");
        writeln!(file, "{{corrupt-partial-json-line...").expect("write corrupt line");
        writeln!(file, "").expect("write blank line");
    }

    // Write another valid event
    let event2 = AgentEvent::telemetry("node-1", serde_json::json!({"step": 2}));
    buffer.push(&event2).expect("push event 2");

    // Total lines in file is 4 (2 valid + 1 corrupt + 1 empty)
    // Drain should gracefully skip the corrupt line and recover both valid events!
    let drained = buffer.drain().expect("drain buffer");
    assert_eq!(drained.len(), 2, "Must recover both valid events despite corrupted line!");
    assert_eq!(drained[0].payload["step"], 1);
    assert_eq!(drained[1].payload["step"], 2);

    let _ = std::fs::remove_dir_all(&temp_dir);
}
