//! Outbound Remote Agent for Sentinel.
//! Connects out to daemon over WebSocket, buffers telemetry during disconnects, and replays on reconnection.

use futures::SinkExt;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::Message;

#[derive(Error, Debug)]
pub enum AgentError {
    #[error("WebSocket connection error: {0}")]
    WsError(#[from] tokio_tungstenite::tungstenite::Error),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerdeError(#[from] serde_json::Error),
}

/// Event transmitted by agent to daemon.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentEvent {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub machine_id: String,
    pub event_type: String,
    pub payload: serde_json::Value,
}

impl AgentEvent {
    pub fn telemetry(machine_id: impl Into<String>, payload: serde_json::Value) -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            machine_id: machine_id.into(),
            event_type: "telemetry".to_string(),
            payload,
        }
    }
}

/// Disk buffer for storing events when connection drops.
#[derive(Clone)]
pub struct EventBuffer {
    buffer_file: PathBuf,
}

impl EventBuffer {
    pub fn new(buffer_file: impl Into<PathBuf>) -> Self {
        let path = buffer_file.into();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        Self { buffer_file: path }
    }

    /// Append event to local buffer file.
    pub fn push(&self, event: &AgentEvent) -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.buffer_file)?;

        let line = serde_json::to_string(event)?;
        writeln!(file, "{line}")?;
        Ok(())
    }

    /// Read and drain all buffered events.
    pub fn drain(&self) -> std::io::Result<Vec<AgentEvent>> {
        if !self.buffer_file.exists() {
            return Ok(Vec::new());
        }

        let file = OpenOptions::new().read(true).open(&self.buffer_file)?;
        let reader = BufReader::new(file);
        let mut events = Vec::new();

        for line in reader.lines() {
            if let Ok(line_str) = line {
                if let Ok(event) = serde_json::from_str::<AgentEvent>(&line_str) {
                    events.push(event);
                }
            }
        }

        // Clear buffer file after reading
        let _ = std::fs::remove_file(&self.buffer_file);
        Ok(events)
    }

    pub fn count(&self) -> usize {
        if !self.buffer_file.exists() {
            return 0;
        }
        if let Ok(file) = OpenOptions::new().read(true).open(&self.buffer_file) {
            BufReader::new(file).lines().count()
        } else {
            0
        }
    }
}

pub struct AgentClient {
    pub machine_id: String,
    pub daemon_ws_url: String,
    pub buffer: EventBuffer,
    pub is_connected: Arc<Mutex<bool>>,
}

impl AgentClient {
    pub fn new(
        machine_id: impl Into<String>,
        daemon_ws_url: impl Into<String>,
        buffer_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            machine_id: machine_id.into(),
            daemon_ws_url: daemon_ws_url.into(),
            buffer: EventBuffer::new(buffer_path),
            is_connected: Arc::new(Mutex::new(false)),
        }
    }

    /// Send event. If disconnected, buffers to local disk.
    pub async fn emit_event(&self, event: AgentEvent) -> Result<bool, AgentError> {
        let connected = *self.is_connected.lock().await;
        if !connected {
            self.buffer.push(&event)?;
            return Ok(false);
        }
        Ok(true)
    }

    /// Flush any stored buffer events over the open WebSocket.
    pub async fn flush_buffer<S>(&self, sink: &mut S) -> Result<usize, AgentError>
    where
        S: SinkExt<Message> + Unpin,
        S::Error: std::fmt::Display,
    {
        let events = self.buffer.drain()?;
        let count = events.len();
        for event in events {
            let json = serde_json::to_string(&event)?;
            sink.send(Message::Text(json))
                .await
                .map_err(|e| AgentError::WsError(tokio_tungstenite::tungstenite::Error::Io(
                    std::io::Error::new(std::io::ErrorKind::ConnectionAborted, e.to_string()),
                )))?;
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_buffer_offline_and_replay() {
        let temp_dir = std::env::temp_dir().join(format!("agent_buf_{}", uuid::Uuid::new_v4()));
        let buf_path = temp_dir.join("buffer.jsonl");
        let buffer = EventBuffer::new(&buf_path);

        assert_eq!(buffer.count(), 0);

        // Push 3 events during simulated network drop
        for i in 1..=3 {
            let event = AgentEvent::telemetry("gpu-box-1", serde_json::json!({ "step": i, "gpu_temp": 40 + i }));
            buffer.push(&event).expect("push event");
        }

        assert_eq!(buffer.count(), 3);

        // Drain & replay
        let drained = buffer.drain().expect("drain buffer");
        assert_eq!(drained.len(), 3);
        assert_eq!(drained[0].payload["step"], 1);
        assert_eq!(drained[2].payload["step"], 3);

        // Buffer should be clean now
        assert_eq!(buffer.count(), 0);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn test_agent_emits_to_buffer_when_offline() {
        let temp_dir = std::env::temp_dir().join(format!("agent_offline_{}", uuid::Uuid::new_v4()));
        let client = AgentClient::new("node-1", "ws://localhost:9999", temp_dir.join("buf.jsonl"));

        let event = AgentEvent::telemetry("node-1", serde_json::json!({"vram": 8000}));
        let sent = client.emit_event(event).await.expect("emit offline event");

        assert!(!sent); // false means saved to disk buffer
        assert_eq!(client.buffer.count(), 1);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
