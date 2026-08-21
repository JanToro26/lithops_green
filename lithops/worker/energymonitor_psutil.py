import os
import time
import logging
import platform
import threading

from lithops.worker.processor_info import get_processor_info, resolve_tdp

logger = logging.getLogger(__name__)

# Split of package TDP into a static floor and a load-proportional range.
#
# These are PLACEHOLDERS to be replaced by the cluster calibration run; they
# are named here so there is exactly one place to change them afterwards.
# Until then, every absolute joule figure from this monitor inherits them.
#
# How to fit each one, and when it must be revisited:
#
#   IDLE_FRACTION -- directly measurable. Read RAPL package energy over a known
#     window at near-zero utilisation: IDLE_FRACTION = (E/t) / TDP. Do not
#     carry a value across CPU families; idle draw depends on the part and on
#     which C-states the platform actually enters.
#
#   DYNAMIC_FRACTION -- fit as the slope of package power against utilisation.
#     It is NOT constrained to equal 1 - IDLE_FRACTION. That identity assumes
#     P(100%) == TDP exactly, which is false in both directions: packages
#     exceed TDP under short-term turbo/PL2, and rarely reach it on non-AVX
#     workloads. Nothing in this file couples the two, and nothing should.
#
#   CORES_FRACTION_OF_PKG -- prefer measuring over fitting. RAPL exposes the
#     core/pp0 subdomain separately from the package, so on any host with RAPL
#     this ratio is an observation, not a parameter. It is only needed where
#     RAPL is absent (Lambda), and should be carried over from the calibration
#     host of the same microarchitecture rather than assumed.
#
# Revisit them when: (a) the calibration run completes; (b) you move to a
# different CPU family -- Graviton, Xeon and EPYC will not share values; or
# (c) the psutil_vs_rapl_pct column in profiling_avg.csv shows structure. A
# roughly constant offset points at IDLE_FRACTION; an offset that grows with
# utilisation points at DYNAMIC_FRACTION; an offset that tracks neither means
# the model's shape is wrong and no choice of constants will fix it.
#
# LIMITATION: these are module-level globals, so all CPUs share one set. If
# calibration yields materially different values per family, they belong
# alongside the watts in processor_info._TDP_TABLE, not here.
IDLE_FRACTION = 0.15
DYNAMIC_FRACTION = 0.85
CORES_FRACTION_OF_PKG = 0.75


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

        # --- Process-tree CPU accounting ---
        # The monitored process is the worker handler, but on Unix the user's
        # function does NOT run in it: handler.run_task launches the JobRunner
        # in a child `multiprocessing.Process`. psutil.Process.cpu_percent()
        # covers one process only, so measuring the handler alone reported the
        # idle floor and nothing else on every Unix backend (Kubernetes,
        # Lambda). Windows was unaffected only because the JobRunner is a
        # Thread there, inside the very process being measured.
        #
        # Everything below therefore accounts for the whole process TREE rooted
        # at process_id. This keeps the measurement window exactly where it was
        # (handler.py needs no change) and stays correct when several tasks run
        # concurrently, because each queue-consumer process is the root of its
        # own subtree.
        self._proc = None
        self._proc_samples = []
        self._sampling = False
        self._sampler_thread = None
        self._sample_interval = 0.5
        # Cumulative CPU seconds of the tree, sampled at the two window edges.
        self._cpu_seconds_start = None
        self._cpu_seconds_end = None
        
        # --- Hardware Discovery ---
        # Resolved once, from processor_info (cached), instead of re-probing
        # /proc/cpuinfo and py-cpuinfo on every invocation.
        self.processor_info = get_processor_info()
        self.arch = (self.processor_info.get("architecture") or platform.machine()).lower()
        self.cpu_model = self.processor_info.get("processor_name") or "Unknown"

        tdp = resolve_tdp(self.processor_info)
        self.base_tdp = tdp["tdp_w"]
        self.tdp_source = tdp["tdp_source"]
        self.tdp_is_default = tdp["is_default"]

        try:
            import psutil

            self.n_logical = psutil.cpu_count(logical=True) or 1
        except Exception:
            self.n_logical = self.processor_info.get("threads") or 1

        if self.tdp_is_default:
            logger.warning(
                f"TDP for '{self.cpu_model}' (arch={self.arch}) could not be resolved; "
                f"falling back to {self.base_tdp}W. Modelled energy from this worker "
                f"rests on an unverified constant."
            )
        logger.info(
            f"Monitor initialized: Arch={self.arch}, CPU={self.cpu_model}, "
            f"TDP_Ref={self.base_tdp}W ({self.tdp_source}), n_logical={self.n_logical}"
        )
        
    def start(self):
        """Start collecting initial system metrics using PSUtil."""
        logger.debug("Starting PSUtil system monitoring")
        
        try:
            import psutil
            
            # Metrics are collected BEFORE the clock starts. The previous order
            # placed extra seconds of blocking sleep inside the measured window, which
            # was then multiplied by idle power and charged to the function.
            psutil.cpu_percent(interval=None)
            self.initial_metrics = self._collect_system_metrics()
            self.start_time = time.time()

            try:
                self._proc = psutil.Process(self.process_id)
            except Exception as e:
                logger.debug(f"Could not attach to process {self.process_id}: {e}")
                self._proc = None
            self._proc_samples = []
            self._cpu_seconds_start = self._tree_cpu_seconds()
            self._cpu_seconds_end = None
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
            
    @staticmethod
    def _own_and_reaped_cpu(proc):
        """
        CPU seconds charged to `proc`: its own user+system, plus the user+system
        of every child it has already waited for.

        The children_* terms are what make the total safe against a child that
        exits mid-window. While the child lives it is counted through the live
        walk in _tree_cpu_seconds; the moment the parent reaps it, the same
        seconds reappear here. There is no interval in which they are counted
        twice, because children_* only accumulates on wait(), and no interval in
        which they are lost. On Windows these fields are always zero, which is
        correct: the JobRunner is a Thread, so the parent's own times cover it.
        """
        t = proc.cpu_times()
        return (
            float(t.user)
            + float(t.system)
            + float(getattr(t, 'children_user', 0.0) or 0.0)
            + float(getattr(t, 'children_system', 0.0) or 0.0)
        )

    def _tree_cpu_seconds(self):
        """
        Cumulative CPU seconds consumed by the monitored process and all of its
        descendants, or None if the tree cannot be read.

        The value is monotonic, so the difference between two readings is the
        CPU time actually consumed between them -- including by processes that
        started and finished inside the interval.
        """
        if self._proc is None:
            return None
        try:
            total = self._own_and_reaped_cpu(self._proc)
        except Exception as e:
            logger.debug(f"Could not read CPU times of process {self.process_id}: {e}")
            return None
        try:
            descendants = self._proc.children(recursive=True)
        except Exception:
            descendants = []
        for child in descendants:
            try:
                total += self._own_and_reaped_cpu(child)
            except Exception:
                # The child exited between the listing and the read. Its time is
                # not lost: it lands in the parent's children_* on the next call.
                continue
        return total

    def _sample_loop(self):
        """
        Background loop: sample the CPU% of the whole process tree every
        _sample_interval seconds.

        The samples describe the shape of the utilisation over the execution;
        the energy figure itself comes from the exact window difference computed
        in get_energy_data, so a function shorter than one interval still gets a
        real number instead of falling back to a system-wide estimate.
        """
        prev_cpu = self._tree_cpu_seconds()
        prev_ts = time.time()
        while self._sampling:
            time.sleep(self._sample_interval)
            try:
                now_cpu = self._tree_cpu_seconds()
                now_ts = time.time()
                elapsed = now_ts - prev_ts
                if prev_cpu is not None and now_cpu is not None and elapsed > 0:
                    # Percentage of ONE core, the same unit the previous
                    # Process.cpu_percent() call produced, so everything
                    # downstream keeps its meaning.
                    self._proc_samples.append(
                        max(0.0, now_cpu - prev_cpu) / elapsed * 100.0
                    )
                prev_cpu, prev_ts = now_cpu, now_ts
            except Exception:
                continue

    def stop(self):
        """Stop monitoring and collect final system metrics."""
        logger.debug("Stopping PSUtil system monitoring")
        
        if self.start_time is None:
            logger.warning("PSUtil monitoring was not started")
            return

        # Stamp the end of the measured window first. Joining the sampler
        # thread can block for a determinate number of seconds; counting
        # that as function time inflated duration, and therefore energy.
        self.end_time = time.time()
        # Read the tree counter at the same instant as the clock. handler.py
        # calls stop() straight after jrp.join(), so the JobRunner has just been
        # reaped and its full CPU time is already in the handler's children_*.
        # Reading here rather than relying on the last periodic sample is what
        # keeps the final fraction of a second of work from being dropped.
        self._cpu_seconds_end = self._tree_cpu_seconds()

        self._sampling = False
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=self._sample_interval * 2 + 1)

        try:
            import psutil

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
                # Non-blocking. psutil.cpu_percent(interval=None) reports usage
                # since the previous call, so calling it at start and at stop
                # yields the average over the function window at zero cost.
                # The former sleep(0.5) here ran twice per invocation and, on
                # Lambda, would be billed and charged to the energy measurement.
                cpu_percent = psutil.cpu_percent(interval=None)
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
                
                # Non-blocking, same reasoning as system_cpu_percent above.
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
        Estimated energy from an empirical package power model.

            P_pkg(U) = TDP * (IDLE_FRACTION + DYNAMIC_FRACTION * U)

        where U is *package* utilisation in [0, 1].

        Attribution across concurrent workers
        """
        duration = self.end_time - self.start_time if self.start_time and self.end_time else 0

        if not self.final_metrics:
            return {'energy': {'pkg': 0, 'cores': 0}, 'duration': duration, 'source': 'none'}

        n_logical = self.n_logical or 1

        # Average CPU% of the monitored process TREE over the whole execution.
        # This is a percentage of ONE core, so 250.0 means 2.5 cores busy.
        #
        # Three sources, in decreasing order of fidelity:
        #   1. the exact CPU seconds consumed between the two window edges,
        #      which needs no sampling at all and is therefore immune both to a
        #      function shorter than one sample interval and to sampler jitter;
        #   2. the mean of the periodic samples, if the edges could not be read;
        #   3. system-wide CPU, which attributes the whole machine to this
        #      worker and is a last resort.
        cpu_source = 'window_delta'
        if (self._cpu_seconds_start is not None
                and self._cpu_seconds_end is not None
                and duration > 0):
            window_cpu_s = max(0.0, self._cpu_seconds_end - self._cpu_seconds_start)
            raw_proc_cpu = window_cpu_s / duration * 100.0
        elif self._proc_samples:
            cpu_source = 'samples'
            raw_proc_cpu = sum(self._proc_samples) / len(self._proc_samples)
        else:
            cpu_source = 'system_wide'
            system_pct = self.final_metrics.get('system_cpu_percent', 0) or 0
            raw_proc_cpu = system_pct * n_logical
            logger.warning(
                "psutil power model fell back to system-wide CPU: the process "
                "tree could not be read. The utilisation of this worker is not "
                "distinguishable from the machine's."
            )

        cores_used = raw_proc_cpu / 100.0
        util_share = min(1.0, cores_used / n_logical)  # this worker's share of the package

        p_idle_machine = self.base_tdp * IDLE_FRACTION          # machine-level, count once
        p_dynamic = self.base_tdp * DYNAMIC_FRACTION * util_share  # attributable, summable

        energy_dynamic_j = p_dynamic * duration
        energy_idle_machine_j = p_idle_machine * duration

        return {
            'duration': round(duration, 4),
            'source': f'psutil_modeled_{self.arch}',
            'energy': {
                # Summable across workers without double-counting idle.
                'pkg': round(energy_dynamic_j, 6),
                'cores': round(energy_dynamic_j * CORES_FRACTION_OF_PKG, 6),
                'pkg_dynamic': round(energy_dynamic_j, 6),
                # Machine floor for this window. Add ONCE per execution, not per worker.
                'pkg_idle_machine': round(energy_idle_machine_j, 6),
                # The pre-fix per-worker value is deliberately NOT emitted. It
                # equals pkg_dynamic + p_idle_machine_w * duration, so it stays
                # reconstructible without carrying a known-wrong column.
                'p_idle_machine_w': round(p_idle_machine, 4),
                'avg_cpu_percent': round(util_share * 100.0, 2),
                'proc_cpu_percent': round(raw_proc_cpu, 2),
                'cores_used': round(cores_used, 4),
                'util_share': round(util_share, 6),
                'cpu_samples': len(self._proc_samples),
                # Which of the three paths above produced raw_proc_cpu.
                # 'system_wide' means this worker's utilisation could not be
                # isolated and the figure should not be summed across workers.
                'cpu_source': cpu_source,
            },
            'cpu_info': {
                'model': self.cpu_model,
                'arch': self.arch,
                'tdp_ref': self.base_tdp,
                'tdp_source': self.tdp_source,
                'tdp_is_default': self.tdp_is_default,
                'n_logical': n_logical,
                'idle_fraction': IDLE_FRACTION,
                'dynamic_fraction': DYNAMIC_FRACTION,
            }
        }
        
    def log_energy_data(self, energy_data, task, cpu_info, function_name=None):
        """
        Log the modelled power/energy summary for this invocation.

        Reads the dict `get_energy_data` actually returns, i.e. its 'energy' and
        'cpu_info' sub-dicts. The previous version read 'system' and 'process'
        keys, which belonged to the older resource-metrics-only implementation
        left commented out above; against the current dict every `.get` missed
        its default, so the whole summary logged zeros no matter what had been
        measured.
        """
        if function_name:
            self.function_name = function_name
        
        logger.info("=== PSUtil Modelled Energy Summary ===")
        
        energy = energy_data.get('energy', {})
        info = energy_data.get('cpu_info', {})
        duration = energy_data.get('duration', 0.0)
        
        # The hardware the power model was applied to. The TDP is the model's
        # most influential input, so it is logged with its resolution status.
        logger.info(
            f"CPU: {info.get('model', 'Unknown')} ({info.get('arch', 'Unknown')}), "
            f"{info.get('n_logical', 0)} logical cores, "
            f"TDP_ref={info.get('tdp_ref', 0.0)}W"
            + (" [UNRESOLVED DEFAULT]" if info.get('tdp_is_default', True) else "")
        )
        
        # Utilisation actually observed for the monitored process. proc_cpu is a
        # percentage of ONE core (250% == 2.5 cores busy); util_share is this
        # worker's share of the whole package, which is what drives the model.
        logger.info(
            f"Utilisation: {energy.get('proc_cpu_percent', 0.0):.2f}% of one core "
            f"({energy.get('cores_used', 0.0):.3f} cores), package share "
            f"{energy.get('util_share', 0.0) * 100:.2f}%, "
            f"{energy.get('cpu_samples', 0)} samples, "
            f"source={energy.get('cpu_source', 'unknown')}"
        )
        
        # Modelled energy. The idle floor is a property of the host, not of this
        # worker: count it ONCE per host per window, never once per co-located
        # worker, or local energy looks superlinear in the worker count.
        logger.info(
            f"Energy (modelled): dynamic {energy.get('pkg_dynamic', 0.0):.6f} J, "
            f"machine idle floor {energy.get('pkg_idle_machine', 0.0):.6f} J "
            f"(P_idle={energy.get('p_idle_machine_w', 0.0):.2f} W, count once per host)"
        )
        
        if info.get('tdp_is_default', True):
            logger.warning(
                f"TDP for '{info.get('model', 'Unknown')}' was not resolved; the "
                f"{energy.get('pkg_dynamic', 0.0):.3f} J above rest on a fallback constant."
            )
        
        logger.info(f"Source: {energy_data.get('source', 'unknown')}")
        
        logger.info(f"Monitoring duration: {duration:.2f} seconds")
        
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
        """
        Deprecated. Kept only so external callers do not break.

        Processor identity now comes from ``processor_info.get_processor_info``,
        which is cached, handles ARM/Graviton parts that expose no "model name"
        line, and does not import py-cpuinfo on every invocation.
        """
        return get_processor_info().get("processor_name") or platform.processor() or "Unknown"

    def _assign_tdp(self):
        """
        Deprecated. Superseded by ``processor_info.resolve_tdp``.

        The old logic returned 95.0 W unless the model string literally
        contained "Xeon" or "EPYC". On Lambda the string is frequently neither,
        so the most influential input of the power model defaulted silently and
        the run gave no indication that it had. ``resolve_tdp`` matches against
        a sourced table and reports ``is_default`` when it cannot resolve.
        """
        return resolve_tdp(self.processor_info)["tdp_w"]