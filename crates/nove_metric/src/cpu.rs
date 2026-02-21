use std::sync::Mutex;
use sysinfo::{CpuRefreshKind, RefreshKind, System};

use crate::{Metric, MetricError, MetricValue, ResourceMetric};

/// CPU usage metric.
///
/// # Notes
/// * It depends on the `sysinfo` crate, which provides CPU usage for all cores.
///
/// # Fields
/// * `system` - The system information.
/// * `average` - Whether to return the average CPU usage or the usage of each CPU core in `sample` method.
/// * `value` - The value of the metric.
///
/// # Examples
/// ```
/// use nove::metric::{Metric, MetricValue, ResourceMetric, CpuUsageMetric};
///
/// let metric = CpuUsageMetric::new(true);
/// let temp = metric.sample().unwrap();
/// println!("{}: {:?}", metric.name().unwrap(), temp);
///
/// let metric = CpuUsageMetric::new(false);
/// let temps = metric.sample().unwrap();
/// println!("{}: {:?}", metric.name().unwrap(), temps);
/// ```
pub struct CpuUsageMetric {
    system: Mutex<System>,
    average: bool,
    value: MetricValue,
}

impl CpuUsageMetric {
    /// Creates a new CPU usage metric.
    ///
    /// # Arguments
    /// * `average` - Whether to return the average CPU usage or the usage of each CPU core in `sample` method.
    ///
    /// # Returns
    /// * `CpuUsageMetric` - The new CPU usage metric.
    pub fn new(average: bool) -> Self {
        let mut system =
            System::new_with_specifics(RefreshKind::new().with_cpu(CpuRefreshKind::everything()));
        system.refresh_cpu_all();
        Self {
            system: Mutex::new(system),
            average,
            value: MetricValue::Scalar(0.0),
        }
    }
}

impl Metric for CpuUsageMetric {
    fn name(&self) -> Result<String, MetricError> {
        Ok("CPU Usage".to_string())
    }

    fn value(&self) -> Result<MetricValue, MetricError> {
        Ok(self.value.clone())
    }

    fn update(&mut self, value: MetricValue) -> Result<(), MetricError> {
        self.value = value;
        Ok(())
    }
}

impl ResourceMetric for CpuUsageMetric {
    fn sample(&self) -> Result<MetricValue, MetricError> {
        let mut system = self.system.lock()?;
        system.refresh_cpu_all();

        let cpus = system.cpus();
        let usage_values: Vec<f64> = cpus.iter().map(|cpu| cpu.cpu_usage() as f64).collect();

        match self.average {
            true => {
                let avg = usage_values.iter().sum::<f64>() / usage_values.len() as f64;
                Ok(MetricValue::Scalar(avg))
            }
            false => Ok(MetricValue::Vector(usage_values)),
        }
    }
}

/// CPU frequency metric.
///
/// # Notes
/// * It depends on the `sysinfo` crate, which provides CPU frequency for all cores.
///
/// # Fields
/// * `system` - The system information.
/// * `average` - Whether to return the average CPU frequency or the frequency of each CPU core in `sample` method.
/// * `value` - The value of the metric.
///
/// # Examples
/// ```
/// use nove::metric::{Metric, MetricValue, ResourceMetric, CpuFrequencyMetric};
///
/// let metric = CpuFrequencyMetric::new(true);
/// let freq = metric.sample().unwrap();
/// println!("{}: {:?}", metric.name().unwrap(), freq);
///
/// let metric = CpuFrequencyMetric::new(false);
/// let freqs = metric.sample().unwrap();
/// println!("{}: {:?}", metric.name().unwrap(), freqs);
/// ```
pub struct CpuFrequencyMetric {
    system: Mutex<System>,
    average: bool,
    value: MetricValue,
}

impl CpuFrequencyMetric {
    /// Creates a new CPU frequency metric.
    ///
    /// # Arguments
    /// * `average` - Whether to return the average CPU frequency or the frequency of each CPU core in `sample` method.
    ///
    /// # Returns
    /// * `CpuFrequencyMetric` - The new CPU frequency metric.
    pub fn new(average: bool) -> Self {
        let mut system =
            System::new_with_specifics(RefreshKind::new().with_cpu(CpuRefreshKind::everything()));
        system.refresh_cpu_all();
        Self {
            system: Mutex::new(system),
            average,
            value: MetricValue::Scalar(0.0),
        }
    }
}

impl Metric for CpuFrequencyMetric {
    fn name(&self) -> Result<String, MetricError> {
        Ok("CPU Frequency".to_string())
    }

    fn value(&self) -> Result<MetricValue, MetricError> {
        Ok(self.value.clone())
    }

    fn update(&mut self, value: MetricValue) -> Result<(), MetricError> {
        self.value = value;
        Ok(())
    }
}

impl ResourceMetric for CpuFrequencyMetric {
    fn sample(&self) -> Result<MetricValue, MetricError> {
        let mut system = self.system.lock()?;
        system.refresh_cpu_all();

        let cpus = system.cpus();
        let freq_values: Vec<f64> = cpus.iter().map(|cpu| cpu.frequency() as f64).collect();

        match self.average {
            true => {
                let avg = freq_values.iter().sum::<f64>() / freq_values.len() as f64;
                Ok(MetricValue::Scalar(avg))
            }
            false => Ok(MetricValue::Vector(freq_values)),
        }
    }
}
