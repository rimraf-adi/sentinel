//! Scheduler, Hardware Monitor, and Dynamic GPU Worker Pool.

use sentinel_core::models::HardwareStats;
use sentinel_core::vitals::{Vitals, VitalsMonitor, Watch};
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Hardware monitor: real host, GPU and process telemetry (see `sentinel_core::vitals`).
/// GPUs are only reported when they exist; nothing is simulated.
pub struct HardwareMonitor {
    vitals: VitalsMonitor,
}

impl Default for HardwareMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl HardwareMonitor {
    pub fn new() -> Self {
        Self { vitals: VitalsMonitor::new() }
    }

    /// Full snapshot including per-core CPU, swap, load, daemon footprint and per-job process trees.
    pub fn sample(&mut self, machine_id: &str, watch: &[Watch]) -> Vitals {
        self.vitals.sample(machine_id, watch)
    }

    /// Number of NVIDIA GPUs that can actually be scheduled (CUDA_VISIBLE_DEVICES applies to these only).
    pub fn schedulable_gpus(&mut self) -> u32 {
        if self.vitals.has_nvidia() {
            self.vitals.sample("local", &[]).gpus.iter().filter(|g| g.vendor == "nvidia").count() as u32
        } else {
            0
        }
    }

    pub fn system_info(&mut self) -> sentinel_core::vitals::SystemInfo {
        self.vitals.system_info()
    }

    pub fn gpu_error(&self) -> Option<&str> {
        self.vitals.gpu_error()
    }

    /// Compact snapshot kept for the original `/api/hardware` shape.
    pub fn collect_stats(&mut self, machine_id: &str) -> HardwareStats {
        let v = self.vitals.sample(machine_id, &[]);
        HardwareStats {
            timestamp: v.timestamp,
            machine_id: v.machine_id,
            cpu_util_percent: v.cpu_util_percent,
            ram_used_bytes: v.ram_used_bytes,
            ram_total_bytes: v.ram_total_bytes,
            gpus: v.gpus,
        }
    }
}

/// Dynamic GPU worker pool allocating physical devices to queued trials.
#[derive(Clone)]
pub struct DynamicGpuPool {
    available_gpus: Arc<Mutex<HashSet<u32>>>,
    total_gpus: u32,
}

impl DynamicGpuPool {
    pub fn new(total_gpus: u32) -> Self {
        let mut gpus = HashSet::new();
        for i in 0..total_gpus {
            gpus.insert(i);
        }
        Self {
            available_gpus: Arc::new(Mutex::new(gpus)),
            total_gpus,
        }
    }

    /// Allocate requested number of GPU IDs. Returns None if not enough GPUs are free.
    pub async fn allocate(&self, count: u32) -> Option<Vec<u32>> {
        let mut lock = self.available_gpus.lock().await;
        if count == 0 {
            // CPU-only job
            return Some(Vec::new());
        }

        if (lock.len() as u32) < count {
            return None;
        }

        let mut allocated = Vec::new();
        let to_take: Vec<u32> = lock.iter().take(count as usize).copied().collect();
        for id in to_take {
            lock.remove(&id);
            allocated.push(id);
        }

        Some(allocated)
    }

    /// Release GPU IDs back to the pool.
    pub async fn release(&self, gpus: &[u32]) {
        let mut lock = self.available_gpus.lock().await;
        for &id in gpus {
            if id < self.total_gpus {
                lock.insert(id);
            }
        }
    }

    /// Number of free GPUs remaining.
    pub async fn free_count(&self) -> usize {
        self.available_gpus.lock().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_monitor_telemetry() {
        let mut monitor = HardwareMonitor::new();
        let stats = monitor.collect_stats("local-node");

        assert_eq!(stats.machine_id, "local-node");
        assert!(stats.ram_total_bytes > 0);
        // GPUs are whatever the machine really has: possibly none, never invented.
        assert!(stats.gpus.iter().all(|g| !g.name.is_empty()));
    }

    #[tokio::test]
    async fn test_dynamic_gpu_pool_allocation_and_release() {
        let pool = DynamicGpuPool::new(4);
        assert_eq!(pool.free_count().await, 4);

        // Allocate 2 GPUs for Run A
        let gpus_a = pool.allocate(2).await.expect("allocate 2 gpus");
        assert_eq!(gpus_a.len(), 2);
        assert_eq!(pool.free_count().await, 2);

        // Allocate 1 GPU for Run B
        let gpus_b = pool.allocate(1).await.expect("allocate 1 gpu");
        assert_eq!(gpus_b.len(), 1);
        assert_eq!(pool.free_count().await, 1);

        // Try allocating 2 GPUs when only 1 is free -> should return None
        assert!(pool.allocate(2).await.is_none());

        // Release Run A GPUs
        pool.release(&gpus_a).await;
        assert_eq!(pool.free_count().await, 3);

        // Now allocation of 2 succeeds
        let gpus_c = pool.allocate(2).await.expect("allocate 2 gpus after release");
        assert_eq!(gpus_c.len(), 2);

        // Clean up
        pool.release(&gpus_b).await;
        pool.release(&gpus_c).await;
        assert_eq!(pool.free_count().await, 4);
    }
}
