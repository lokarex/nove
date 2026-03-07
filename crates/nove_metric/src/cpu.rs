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
/// let mut metric = CpuUsageMetric::new(true);
/// metric.sample().unwrap();
/// let temp = metric.value().unwrap();
/// println!("{}: {:?}", metric.name().unwrap(), temp);
///
/// let mut metric = CpuUsageMetric::new(false);
/// metric.sample().unwrap();
/// let temps = metric.value().unwrap();
/// println!("{}: {:?}", metric.name().unwrap(), temps);
/// ```
pub struct CpuUsageMetric {
    system: Mutex<System>,
    average: bool,
    value: MetricValue,
    sample_count: usize,
    total_usage: f64,
    total_usages: Vec<f64>,
}

impl std::fmt::Debug for CpuUsageMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuUsageMetric")
            .field("average", &self.average)
            .field("value", &self.value)
            .finish()
    }
}

impl Clone for CpuUsageMetric {
    fn clone(&self) -> Self {
        let system =
            System::new_with_specifics(RefreshKind::new().with_cpu(CpuRefreshKind::everything()));
        let cpu_count = system.cpus().len();
        Self {
            system: Mutex::new(system),
            average: self.average,
            value: self.value.clone(),
            sample_count: self.sample_count,
            total_usage: self.total_usage,
            total_usages: vec![0.0; cpu_count],
        }
    }
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
        let cpu_count = system.cpus().len();
        Self {
            system: Mutex::new(system),
            average,
            value: MetricValue::Scalar(0.0),
            sample_count: 0,
            total_usage: 0.0,
            total_usages: vec![0.0; cpu_count],
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

    fn clear(&mut self) -> Result<(), MetricError> {
        self.value = MetricValue::Scalar(0.0);
        self.sample_count = 0;
        self.total_usage = 0.0;
        self.total_usages.clear();
        let system = self.system.lock()?;
        let cpu_count = system.cpus().len();
        self.total_usages.resize(cpu_count, 0.0);
        Ok(())
    }
}

impl ResourceMetric for CpuUsageMetric {
    fn sample(&mut self) -> Result<(), MetricError> {
        let mut system = self.system.lock()?;
        system.refresh_cpu_all();

        let cpus = system.cpus();
        let usage_values: Vec<f64> = cpus.iter().map(|cpu| cpu.cpu_usage() as f64).collect();

        match self.average {
            true => {
                let batch_avg = usage_values.iter().sum::<f64>() / usage_values.len() as f64;
                self.total_usage += batch_avg;
                self.sample_count += 1;
                let avg = if self.sample_count > 0 {
                    self.total_usage / self.sample_count as f64
                } else {
                    0.0
                };
                self.value = MetricValue::Scalar(avg);
            }
            false => {
                if self.total_usages.len() != usage_values.len() {
                    self.total_usages.resize(usage_values.len(), 0.0);
                }

                for (i, &usage) in usage_values.iter().enumerate() {
                    if i < self.total_usages.len() {
                        self.total_usages[i] += usage;
                    }
                }

                self.sample_count += 1;

                let avg_values: Vec<f64> = self
                    .total_usages
                    .iter()
                    .map(|&total| {
                        if self.sample_count > 0 {
                            total / self.sample_count as f64
                        } else {
                            0.0
                        }
                    })
                    .collect();

                self.value = MetricValue::Vector(avg_values);
            }
        }
        Ok(())
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
/// let mut metric = CpuFrequencyMetric::new(true);
/// metric.sample().unwrap();
/// let freq = metric.value().unwrap();
/// println!("{}: {:?}", metric.name().unwrap(), freq);
///
/// let mut metric = CpuFrequencyMetric::new(false);
/// metric.sample().unwrap();
/// let freqs = metric.value().unwrap();
/// println!("{}: {:?}", metric.name().unwrap(), freqs);
/// ```
pub struct CpuFrequencyMetric {
    system: Mutex<System>,
    average: bool,
    value: MetricValue,
    sample_count: usize,
    total_frequency: f64,
    total_frequencies: Vec<f64>,
}

impl std::fmt::Debug for CpuFrequencyMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuFrequencyMetric")
            .field("average", &self.average)
            .field("value", &self.value)
            .finish()
    }
}

impl Clone for CpuFrequencyMetric {
    fn clone(&self) -> Self {
        let system =
            System::new_with_specifics(RefreshKind::new().with_cpu(CpuRefreshKind::everything()));
        let cpu_count = system.cpus().len();
        Self {
            system: Mutex::new(system),
            average: self.average,
            value: self.value.clone(),
            sample_count: self.sample_count,
            total_frequency: self.total_frequency,
            total_frequencies: vec![0.0; cpu_count],
        }
    }
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
        let cpu_count = system.cpus().len();
        Self {
            system: Mutex::new(system),
            average,
            value: MetricValue::Scalar(0.0),
            sample_count: 0,
            total_frequency: 0.0,
            total_frequencies: vec![0.0; cpu_count],
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

    fn clear(&mut self) -> Result<(), MetricError> {
        self.value = MetricValue::Scalar(0.0);
        self.sample_count = 0;
        self.total_frequency = 0.0;
        self.total_frequencies.clear();
        let system = self.system.lock()?;
        let cpu_count = system.cpus().len();
        self.total_frequencies.resize(cpu_count, 0.0);
        Ok(())
    }
}

impl ResourceMetric for CpuFrequencyMetric {
    fn sample(&mut self) -> Result<(), MetricError> {
        let mut system = self.system.lock()?;
        system.refresh_cpu_all();

        let cpus = system.cpus();
        let freq_values: Vec<f64> = cpus.iter().map(|cpu| cpu.frequency() as f64).collect();

        match self.average {
            true => {
                let batch_avg = freq_values.iter().sum::<f64>() / freq_values.len() as f64;
                self.total_frequency += batch_avg;
                self.sample_count += 1;
                let avg = if self.sample_count > 0 {
                    self.total_frequency / self.sample_count as f64
                } else {
                    0.0
                };
                self.value = MetricValue::Scalar(avg);
            }
            false => {
                if self.total_frequencies.len() != freq_values.len() {
                    self.total_frequencies.resize(freq_values.len(), 0.0);
                }

                for (i, &freq) in freq_values.iter().enumerate() {
                    if i < self.total_frequencies.len() {
                        self.total_frequencies[i] += freq;
                    }
                }

                self.sample_count += 1;

                let avg_values: Vec<f64> = self
                    .total_frequencies
                    .iter()
                    .map(|&total| {
                        if self.sample_count > 0 {
                            total / self.sample_count as f64
                        } else {
                            0.0
                        }
                    })
                    .collect();

                self.value = MetricValue::Vector(avg_values);
            }
        }
        Ok(())
    }
}
