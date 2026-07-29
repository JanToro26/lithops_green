import os
import time
import logging
import platform
import threading

logger = logging.getLogger(__name__)


class EnergyMonitor:
    """
    PSUtil-based system resource monitor.
    This monitor does NOT measure energy - it only collects system resource metrics
    using the psutil library for system monitoring and CPU information.
    """
    
    def __init__(self, process_id):
        self.process_id = process_id
        self.start_time = None
        self.end_time = None
        self.function_name = None
        self.initial_metrics = {}
        self.final_metrics = {}

        # --- Periodic process CPU sampling ---
        self._proc = None
        self._proc_samples = []
        self._sampling = False
        self._sampler_thread = None
        self._sample_interval = 0.5
        
        # --- Hardware Discovery ---
        self.arch = platform.machine().lower() # e.g., 'x86_64' or 'aarch64'
        self.cpu_model = self._get_cpu_model()
        self.base_tdp = self._assign_tdp()
        
        logger.info(f"Monitor initialized: Arch={self.arch}, CPU={self.cpu_model}, TDP_Ref={self.base_tdp}W")
        
    def start(self):
        """Start collecting initial system metrics using PSUtil."""
        logger.debug("Starting PSUtil system monitoring")
        
        try:
            import psutil
            
            self.start_time = time.time()
            
            # Initialize CPU percent measurement (non-blocking)
            psutil.cpu_percent(interval=None) 
            self.initial_metrics = self._collect_system_metrics()

            try:
                self._proc = psutil.Process(self.process_id)
                self._proc.cpu_percent(None)
            except Exception as e:
                logger.debug(f"Could not attach to process {self.process_id}: {e}")
                self._proc = None
            self._proc_samples = []
            self._sampling = True
            self._sampler_thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._sampler_thread.start()
            
            logger.info("PSUtil system monitoring started successfully")
            return True
            
        except ImportError:
            logger.warning("PSUtil not available - system monitoring disabled")
            return False
        except Exception as e:
            logger.error(f"Error starting PSUtil system monitoring: {e}")
            return False
            
    def _sample_loop(self):
        """Background loop: sample the process CPU% every _sample_interval seconds."""
        while self._sampling:
            try:
                if self._proc is not None:
                    val = self._proc.cpu_percent(interval=self._sample_interval)
                    self._proc_samples.append(val)
                else:
                    time.sleep(self._sample_interval)
            except Exception:
                time.sleep(self._sample_interval)
    def stop(self):
        """Stop monitoring and collect final system metrics."""
        logger.debug("Stopping PSUtil system monitoring")
        
        if self.start_time is None:
            logger.warning("PSUtil monitoring was not started")
            return

        self._sampling = False
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=self._sample_interval * 2 + 1)
            
        try:
            import psutil
            
            self.end_time = time.time()
            
            # Collect final system metrics
            self.final_metrics = self._collect_system_metrics()
            
            duration = self.end_time - self.start_time
            logger.info(f"PSUtil system monitoring stopped after {duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error stopping PSUtil system monitoring: {e}")
            
    def _collect_system_metrics(self):
        """Collect comprehensive system metrics using PSUtil."""
        metrics = {}
        
        try:
            import psutil
            
            # === CPU INFORMATION ===
            try:
                # Core counts
                physical_cores = psutil.cpu_count(logical=False) or 0
                logical_cores = psutil.cpu_count(logical=True) or 0
                metrics['cpu_cores_physical'] = physical_cores
                metrics['cpu_cores_logical'] = logical_cores
                
                # CPU frequency
                freq_info = psutil.cpu_freq()
                if freq_info:
                    metrics['cpu_freq_current'] = freq_info.current
                    metrics['cpu_freq_max'] = freq_info.max
                    metrics['cpu_freq_min'] = freq_info.min
                else:
                    metrics['cpu_freq_current'] = 0.0
                    metrics['cpu_freq_max'] = 0.0
                    metrics['cpu_freq_min'] = 0.0
                    
            except Exception as e:
                logger.debug(f"Error collecting CPU info: {e}")
                metrics['cpu_cores_physical'] = 0
                metrics['cpu_cores_logical'] = 0
                metrics['cpu_freq_current'] = 0.0
                metrics['cpu_freq_max'] = 0.0
                metrics['cpu_freq_min'] = 0.0
            
            # === SYSTEM-WIDE METRICS ===
            try:
                # CPU usage - use non-blocking call first, then blocking call for accurate measurement
                # First call initializes the measurement, second call gets the actual usage
                psutil.cpu_percent(interval=None)  # Non-blocking call to initialize
                time.sleep(0.5)  # Wait a bit for meaningful measurement
                cpu_percent = psutil.cpu_percent(interval=None)  # Get the actual measurement
                metrics['system_cpu_percent'] = cpu_percent
                
                # Also get per-CPU percentages for more detailed analysis
                per_cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
                metrics['per_cpu_percent'] = per_cpu_percent
                metrics['max_cpu_percent'] = max(per_cpu_percent) if per_cpu_percent else 0.0
                metrics['avg_cpu_percent'] = sum(per_cpu_percent) / len(per_cpu_percent) if per_cpu_percent else 0.0
                
                # Memory usage
                memory = psutil.virtual_memory()
                metrics['system_memory_percent'] = memory.percent
                metrics['system_memory_used_mb'] = memory.used / (1024 * 1024)
                metrics['system_memory_total_mb'] = memory.total / (1024 * 1024)
                
            except Exception as e:
                logger.debug(f"Error collecting system metrics: {e}")
                metrics['system_cpu_percent'] = 0.0
                metrics['per_cpu_percent'] = []
                metrics['max_cpu_percent'] = 0.0
                metrics['avg_cpu_percent'] = 0.0
                metrics['system_memory_percent'] = 0.0
                metrics['system_memory_used_mb'] = 0.0
                metrics['system_memory_total_mb'] = 0.0
            
            # === DISK I/O METRICS ===
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    metrics['disk_read_bytes'] = disk_io.read_bytes
                    metrics['disk_write_bytes'] = disk_io.write_bytes
                    metrics['disk_read_count'] = disk_io.read_count
                    metrics['disk_write_count'] = disk_io.write_count
                else:
                    metrics['disk_read_bytes'] = 0
                    metrics['disk_write_bytes'] = 0
                    metrics['disk_read_count'] = 0
                    metrics['disk_write_count'] = 0
            except Exception as e:
                logger.debug(f"Error collecting disk I/O: {e}")
                metrics['disk_read_bytes'] = 0
                metrics['disk_write_bytes'] = 0
                metrics['disk_read_count'] = 0
                metrics['disk_write_count'] = 0
            
            # === NETWORK I/O METRICS ===
            try:
                net_io = psutil.net_io_counters()
                if net_io:
                    metrics['network_sent_bytes'] = net_io.bytes_sent
                    metrics['network_recv_bytes'] = net_io.bytes_recv
                    metrics['network_sent_packets'] = net_io.packets_sent
                    metrics['network_recv_packets'] = net_io.packets_recv
                else:
                    metrics['network_sent_bytes'] = 0
                    metrics['network_recv_bytes'] = 0
                    metrics['network_sent_packets'] = 0
                    metrics['network_recv_packets'] = 0
            except Exception as e:
                logger.debug(f"Error collecting network I/O: {e}")
                metrics['network_sent_bytes'] = 0
                metrics['network_recv_bytes'] = 0
                metrics['network_sent_packets'] = 0
                metrics['network_recv_packets'] = 0
            
            # === PROCESS-SPECIFIC METRICS ===
            try:
                process = psutil.Process(self.process_id)
                
                # Process CPU and memory - use non-blocking for process too
                process.cpu_percent()  # Initialize measurement
                time.sleep(0.2)  # Short wait for process measurement
                process_cpu = process.cpu_percent()
                metrics['process_cpu_percent'] = process_cpu
                
                process_memory = process.memory_info()
                metrics['process_memory_rss_mb'] = process_memory.rss / (1024 * 1024)
                metrics['process_memory_vms_mb'] = process_memory.vms / (1024 * 1024)
                
                # Process status
                metrics['process_status'] = process.status()
                metrics['process_num_threads'] = process.num_threads()
                
                # Additional process info
                try:
                    metrics['process_cpu_times'] = process.cpu_times()._asdict()
                except:
                    metrics['process_cpu_times'] = {}
                
            except psutil.NoSuchProcess:
                logger.debug(f"Process {self.process_id} no longer exists")
                metrics['process_cpu_percent'] = 0.0
                metrics['process_memory_rss_mb'] = 0.0
                metrics['process_memory_vms_mb'] = 0.0
                metrics['process_status'] = 'not_found'
                metrics['process_num_threads'] = 0
                metrics['process_cpu_times'] = {}
            except Exception as e:
                logger.debug(f"Error collecting process metrics: {e}")
                metrics['process_cpu_percent'] = 0.0
                metrics['process_memory_rss_mb'] = 0.0
                metrics['process_memory_vms_mb'] = 0.0
                metrics['process_status'] = 'error'
                metrics['process_num_threads'] = 0
                metrics['process_cpu_times'] = {}
            
            # === CPU TEMPERATURE (if available) ===
            try:
                temps = psutil.sensors_temperatures()
                cpu_temp = 0.0
                if temps:
                    for name, entries in temps.items():
                        if 'cpu' in name.lower() or 'core' in name.lower():
                            if entries:
                                cpu_temp = entries[0].current
                                break
                metrics['cpu_temp_celsius'] = cpu_temp
            except Exception as e:
                logger.debug(f"Error collecting CPU temperature: {e}")
                metrics['cpu_temp_celsius'] = 0.0
            
            # === DETAILED CPU INFO (if py-cpuinfo available) ===
            try:
                import cpuinfo
                cpu_info = cpuinfo.get_cpu_info()
                metrics['cpu_brand'] = cpu_info.get('brand_raw', 'Unknown')
                metrics['cpu_model'] = cpu_info.get('cpu_info_ver_info', {}).get('model_name', 'Unknown')
                metrics['cpu_arch'] = cpu_info.get('arch', 'Unknown')
                metrics['cpu_vendor'] = cpu_info.get('vendor_id_raw', 'Unknown')
            except ImportError:
                logger.debug("py-cpuinfo not available")
                metrics['cpu_brand'] = 'Unknown'
                metrics['cpu_model'] = 'Unknown'
                metrics['cpu_arch'] = 'Unknown'
                metrics['cpu_vendor'] = 'Unknown'
            except Exception as e:
                logger.debug(f"Error getting detailed CPU info: {e}")
                metrics['cpu_brand'] = 'Unknown'
                metrics['cpu_model'] = 'Unknown'
                metrics['cpu_arch'] = 'Unknown'
                metrics['cpu_vendor'] = 'Unknown'
            
        except ImportError:
            logger.warning("PSUtil not available")
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
        
        return metrics
    
    """def get_energy_data(self):
        
        Get system monitoring data. 
        NOTE: This does NOT provide energy measurements - only system resource metrics.
        
        duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # Create the result dictionary - NO ENERGY DATA, only system monitoring
        result = {
            'duration': duration,
            'source': 'psutil_system_monitoring',
            'energy': {},  # Empty - PSUtil doesn't measure energy
            'system': {},
            'process': {},
            'cpu_info': {}
        }
        
        if not self.initial_metrics or not self.final_metrics:
            logger.warning("PSUtil metrics not available - monitoring may not have been started/stopped properly")
            result['source'] = 'unavailable'
            return result
        
        try:
            # === SYSTEM METRICS (improved CPU calculation) ===
            # Use the better measurement from final metrics, with more precision
            initial_cpu = self.initial_metrics.get('system_cpu_percent', 0.0)
            final_cpu = self.final_metrics.get('system_cpu_percent', 0.0)
            initial_avg_cpu = self.initial_metrics.get('avg_cpu_percent', 0.0)
            final_avg_cpu = self.final_metrics.get('avg_cpu_percent', 0.0)
            initial_max_cpu = self.initial_metrics.get('max_cpu_percent', 0.0)
            final_max_cpu = self.final_metrics.get('max_cpu_percent', 0.0)
            
            # Use the maximum of all measurements to get the best representation
            best_cpu_measurement = max(initial_cpu, final_cpu, initial_avg_cpu, final_avg_cpu, 
                                     initial_max_cpu, final_max_cpu)
            
            # If we still have 0, try to calculate from the difference in measurements
            if best_cpu_measurement == 0.0:
                # Calculate weighted average giving more weight to final measurement
                if final_cpu > 0 or initial_cpu > 0:
                    best_cpu_measurement = (initial_cpu * 0.3 + final_cpu * 0.7)
                elif final_avg_cpu > 0 or initial_avg_cpu > 0:
                    best_cpu_measurement = (initial_avg_cpu * 0.3 + final_avg_cpu * 0.7)
            
            result['system'] = {
                'cpu_percent': round(best_cpu_measurement, 6),  # 6 decimal places for precision
                'cpu_percent_initial': round(initial_cpu, 6),
                'cpu_percent_final': round(final_cpu, 6),
                'cpu_percent_avg_initial': round(initial_avg_cpu, 6),
                'cpu_percent_avg_final': round(final_avg_cpu, 6),
                'cpu_percent_max_initial': round(initial_max_cpu, 6),
                'cpu_percent_max_final': round(final_max_cpu, 6),
                'per_cpu_initial': self.initial_metrics.get('per_cpu_percent', []),
                'per_cpu_final': self.final_metrics.get('per_cpu_percent', []),
                'memory_percent': self.final_metrics.get('system_memory_percent', 0.0),
                'memory_used_mb': self.final_metrics.get('system_memory_used_mb', 0.0),
                'memory_total_mb': self.final_metrics.get('system_memory_total_mb', 0.0),
                'cpu_freq_current': self.final_metrics.get('cpu_freq_current', 0.0),
                'cpu_temp_celsius': self.final_metrics.get('cpu_temp_celsius', 0.0),
            }
            
            # === DISK I/O DELTA (difference between start and end) ===
            disk_read_delta = self.final_metrics.get('disk_read_bytes', 0) - self.initial_metrics.get('disk_read_bytes', 0)
            disk_write_delta = self.final_metrics.get('disk_write_bytes', 0) - self.initial_metrics.get('disk_write_bytes', 0)
            
            result['system']['disk_io_read_mb'] = max(0, disk_read_delta) / (1024 * 1024)
            result['system']['disk_io_write_mb'] = max(0, disk_write_delta) / (1024 * 1024)
            
            # === NETWORK I/O DELTA (difference between start and end) ===
            net_sent_delta = self.final_metrics.get('network_sent_bytes', 0) - self.initial_metrics.get('network_sent_bytes', 0)
            net_recv_delta = self.final_metrics.get('network_recv_bytes', 0) - self.initial_metrics.get('network_recv_bytes', 0)
            
            result['system']['network_sent_mb'] = max(0, net_sent_delta) / (1024 * 1024)
            result['system']['network_recv_mb'] = max(0, net_recv_delta) / (1024 * 1024)
            
            # === PROCESS METRICS ===
            initial_process_cpu = self.initial_metrics.get('process_cpu_percent', 0.0)
            final_process_cpu = self.final_metrics.get('process_cpu_percent', 0.0)
            
            # Use the maximum process CPU measurement
            best_process_cpu = max(initial_process_cpu, final_process_cpu)
            if best_process_cpu == 0.0 and (initial_process_cpu > 0 or final_process_cpu > 0):
                best_process_cpu = (initial_process_cpu * 0.3 + final_process_cpu * 0.7)
            
            result['process'] = {
                'cpu_percent': round(best_process_cpu, 6),  # 6 decimal places for precision
                'cpu_percent_initial': round(initial_process_cpu, 6),
                'cpu_percent_final': round(final_process_cpu, 6),
                'memory_mb': self.final_metrics.get('process_memory_rss_mb', 0.0),
                'memory_vms_mb': self.final_metrics.get('process_memory_vms_mb', 0.0),
                'status': self.final_metrics.get('process_status', 'unknown'),
                'num_threads': self.final_metrics.get('process_num_threads', 0),
                'cpu_times': self.final_metrics.get('process_cpu_times', {}),
            }
            
            # === CPU INFORMATION ===
            result['cpu_info'] = {
                'cores_physical': self.final_metrics.get('cpu_cores_physical', 0),
                'cores_logical': self.final_metrics.get('cpu_cores_logical', 0),
                'frequency_current': self.final_metrics.get('cpu_freq_current', 0.0),
                'frequency_max': self.final_metrics.get('cpu_freq_max', 0.0),
                'frequency_min': self.final_metrics.get('cpu_freq_min', 0.0),
                'brand': self.final_metrics.get('cpu_brand', 'Unknown'),
                'model': self.final_metrics.get('cpu_model', 'Unknown'),
                'arch': self.final_metrics.get('cpu_arch', 'Unknown'),
                'vendor': self.final_metrics.get('cpu_vendor', 'Unknown'),
            }
            
            logger.info(f"PSUtil system monitoring data collected successfully")
            logger.info(f"System CPU: {result['system']['cpu_percent']:.6f}% (initial: {result['system']['cpu_percent_initial']:.6f}%, final: {result['system']['cpu_percent_final']:.6f}%)")
            logger.info(f"Process CPU: {result['process']['cpu_percent']:.6f}% (initial: {result['process']['cpu_percent_initial']:.6f}%, final: {result['process']['cpu_percent_final']:.6f}%)")
            logger.debug(f"Memory: {result['system']['memory_percent']:.1f}%, Process Memory: {result['process']['memory_mb']:.1f} MB")
            
        except Exception as e:
            logger.error(f"Error processing PSUtil system metrics: {e}")
            result['source'] = 'error'
        
        return result"""
    
    
    def get_energy_data(self):
        """
        Calculates estimated energy consumption using an empirical power model.
        Energy (J) = [P_idle + (P_dynamic * %CPU)] * Duration
        """
        import psutil
        duration = self.end_time - self.start_time if self.start_time and self.end_time else 0
        
        if not self.final_metrics:
            return {'energy': {'pkg': 0, 'cores': 0}, 'duration': duration, 'source': 'none'}

        # Average process CPU% over the whole execution (periodic sampling).
        if self._proc_samples:
            raw_proc_cpu = sum(self._proc_samples) / len(self._proc_samples)
        else:
            raw_proc_cpu = (self.initial_metrics.get('system_cpu_percent', 0) +
                            self.final_metrics.get('system_cpu_percent', 0)) / 2.0
        try:
            n_logical = psutil.cpu_count(logical=True) or 1
        except Exception:
            n_logical = 1
        avg_cpu = min(100.0, raw_proc_cpu / n_logical)
        
        # Empirical Power Model
        # We assume 15% of TDP is base (Idle) and 85% is dynamic range
        p_idle = self.base_tdp * 0.15
        p_dynamic = (self.base_tdp * 0.85) * (avg_cpu / 100.0)
        
        total_power_w = p_idle + p_dynamic
        energy_j = total_power_w * duration

        return {
            'duration': round(duration, 4),
            'source': f'psutil_modeled_{self.arch}',
            'energy': {
                'pkg': round(energy_j, 6),
                'cores': round(energy_j * 0.75, 6), # Core estimation (approx 75% of PKG)
                'avg_cpu_percent': round(avg_cpu, 2),
                'proc_cpu_percent': round(raw_proc_cpu, 2),
                'cpu_samples': len(self._proc_samples)
            },
            'cpu_info': {
                'model': self.cpu_model,
                'arch': self.arch,
                'tdp_ref': self.base_tdp
            }
        }
        
    def log_energy_data(self, energy_data, task, cpu_info, function_name=None):
        """
        Log system monitoring data.
        NOTE: This does NOT log energy data since PSUtil doesn't measure energy.
        """
        if function_name:
            self.function_name = function_name
        
        logger.info("=== PSUtil System Monitoring Summary ===")
        
        system_data = energy_data.get('system', {})
        process_data = energy_data.get('process', {})
        cpu_info_data = energy_data.get('cpu_info', {})
        
        # Log system metrics with high precision
        logger.info(f"System CPU Usage: {system_data.get('cpu_percent', 0):.6f}%")
        logger.info(f"  - Initial: {system_data.get('cpu_percent_initial', 0):.6f}%, Final: {system_data.get('cpu_percent_final', 0):.6f}%")
        logger.info(f"  - Per-CPU Initial: {system_data.get('per_cpu_initial', [])}")
        logger.info(f"  - Per-CPU Final: {system_data.get('per_cpu_final', [])}")
        logger.info(f"System Memory Usage: {system_data.get('memory_percent', 0):.1f}% ({system_data.get('memory_used_mb', 0):.1f} MB)")
        logger.info(f"Disk I/O: Read {system_data.get('disk_io_read_mb', 0):.2f} MB, Write {system_data.get('disk_io_write_mb', 0):.2f} MB")
        logger.info(f"Network I/O: Sent {system_data.get('network_sent_mb', 0):.2f} MB, Received {system_data.get('network_recv_mb', 0):.2f} MB")
        
        # Log process metrics with high precision
        logger.info(f"Process CPU Usage: {process_data.get('cpu_percent', 0):.6f}%")
        logger.info(f"  - Initial: {process_data.get('cpu_percent_initial', 0):.6f}%, Final: {process_data.get('cpu_percent_final', 0):.6f}%")
        logger.info(f"Process Memory Usage: {process_data.get('memory_mb', 0):.1f} MB")
        
        # Log CPU info
        logger.info(f"CPU: {cpu_info_data.get('brand', 'Unknown')} ({cpu_info_data.get('cores_physical', 0)} physical, {cpu_info_data.get('cores_logical', 0)} logical cores)")
        
        if system_data.get('cpu_freq_current', 0) > 0:
            logger.info(f"CPU Frequency: {system_data.get('cpu_freq_current', 0):.0f} MHz")
        
        if system_data.get('cpu_temp_celsius', 0) > 0:
            logger.info(f"CPU Temperature: {system_data.get('cpu_temp_celsius', 0):.1f}°C")
        
        logger.info(f"Monitoring Duration: {energy_data.get('duration', 0):.2f} seconds")
        
    def update_function_name(self, task, function_name):
        """Update the function name."""
        self.function_name = function_name
        logger.debug(f"PSUtil monitor function name updated to: {function_name}")
        
    def read_function_name_from_stats(self, stats_file):
        """Read function name from stats file."""
        if not os.path.exists(stats_file):
            logger.warning(f"Stats file not found: {stats_file}")
            return False
        
        try:
            with open(stats_file, 'r') as fid:
                for line in fid.readlines():
                    try:
                        key, value = line.strip().split(" ", 1)
                        if key == 'function_name':
                            self.function_name = value
                            logger.info(f"PSUtil monitor found function name: {self.function_name}")
                            return True
                    except Exception as e:
                        logger.debug(f"Error processing stats file line: {line} - {e}")
        except Exception as e:
            logger.error(f"Error reading stats file: {e}")
            
        return False

    def _get_cpu_model(self):
        """Detects the specific CPU model from the system."""
        try:
            # Check for detailed info using py-cpuinfo if available
            import cpuinfo
            return cpuinfo.get_cpu_info().get('brand_raw', 'Generic CPU')
        except ImportError:
            # Fallback to /proc/cpuinfo
            if os.path.exists('/proc/cpuinfo'):
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if "model name" in line or "Model" in line:
                            return line.split(":")[1].strip()
            return platform.processor() or "Unknown"
        
    def _assign_tdp(self):
        """
        Assigns a Thermal Design Power (TDP) reference based on architecture.
        Essential for the power model in AWS Lambda environments.
        """
        # Case 1: ARM Architecture (e.g., AWS Graviton2)
        if 'arm' in self.arch or 'aarch64' in self.arch:
            return 65.0  # Graviton2 typically has lower TDP per core/node
        
        # Case 2: x86 Architecture (Intel/AMD)
        if "Xeon" in self.cpu_model or "EPYC" in self.cpu_model:
            return 125.0  # Standard high-performance server CPU
        
        return 95.0  # Default for standard instances