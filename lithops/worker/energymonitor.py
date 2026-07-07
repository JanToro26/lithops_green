import os
import time
import logging

logger = logging.getLogger(__name__)

class EnergyManager:
    """
    Unified energy manager that runs all available energy monitoring methods simultaneously.
    Collects data from all methods and stores them as separate fields in the result.
    """

    def __init__(self, process_id=None):
        """
        Initialize the energy manager with all available monitoring methods.

        Args:
            process_id: The process ID to monitor (defaults to current PID)
        """
        self.process_id = process_id if process_id is not None else os.getpid()
        self.function_name = None
        self.start_time = None
        self.end_time = None

        # Initialize all energy monitors
        self.monitors = {}
        self.monitor_status = {}

        # Initialize each monitoring method
        self._initialize_monitors()

    def _initialize_monitors(self):
        """Initialize all available energy monitoring methods."""
        monitor_configs = {
            'rapl': {
                'class': 'EnergyMonitor',
                'module': 'lithops.worker.energymonitor_rapl'
            },
            'psutil': {
                'class': 'EnergyMonitor',
                'module': 'lithops.worker.energymonitor_psutil'
            }
        }

        for method_name, config in monitor_configs.items():
            try:
                module = __import__(config['module'], fromlist=[config['class']])
                monitor_class = getattr(module, config['class'])
                monitor = monitor_class(self.process_id)
                self.monitors[method_name] = monitor
                self.monitor_status[method_name] = False
                logger.debug(f"Initialized {method_name} energy monitor")
            except Exception as e:
                logger.warning(f"Failed to initialize {method_name} energy monitor: {e}")
                self.monitors[method_name] = None
                self.monitor_status[method_name] = False

    def start(self):
        """Start all available energy monitoring methods."""
        self.start_time = time.time()
        self.end_time = None
        any_started = False

        for method_name, monitor in self.monitors.items():
            if monitor is not None:
                try:
                    started = monitor.start()
                    self.monitor_status[method_name] = started
                    if started:
                        logger.info(f"Started {method_name} energy monitor")
                        any_started = True
                    else:
                        logger.warning(f"Failed to start {method_name} energy monitor")
                except Exception as e:
                    logger.error(f"Error starting {method_name} energy monitor: {e}")
                    self.monitor_status[method_name] = False
            else:
                self.monitor_status[method_name] = False

        logger.info(f"Energy monitoring started. Active monitors: {[m for m, s in self.monitor_status.items() if s]}")
        return any_started

    def stop(self):
        """Stop all active energy monitoring methods (idempotent)."""
        if self.end_time is not None:
            return
        for method_name, monitor in self.monitors.items():
            if monitor is not None and self.monitor_status[method_name]:
                try:
                    monitor.stop()
                    logger.debug(f"Stopped {method_name} energy monitor")
                except Exception as e:
                    logger.error(f"Error stopping {method_name} energy monitor: {e}")
        self.end_time = time.time()

    def _get_aws_processor_info(self):
        """Best-effort AWS processor info; never raises."""
        info = {'processor': 'unknown', 'arch': 'unknown', 'is_lambda': False}
        try:
            import platform
            info['arch'] = platform.machine()
            info['is_lambda'] = bool(os.environ.get('AWS_LAMBDA_RUNTIME_API'))
            arch = info['arch'].lower()
            if 'aarch64' in arch or 'arm' in arch:
                info['processor'] = 'aws_graviton'
            elif info['is_lambda']:
                info['processor'] = 'x86_64'
        except Exception as e:
            logger.debug(f"AWS processor info unavailable: {e}")
        return info

    def _get_env_info(self):
        import platform
        """Detect whether we are running in AWS Lambda and get resource information."""
        info = {
            'is_lambda': bool(os.environ.get('AWS_LAMBDA_RUNTIME_API')),
            'memory_limit': int(os.environ.get('AWS_LAMBDA_FUNCTION_MEMORY_SIZE', 0)),
            'arch': platform.machine()
        }
        if info['is_lambda'] and info['memory_limit'] > 0:
            self.vcpu_share = info['memory_limit'] / 1769.0
        return info

    def read_function_name_from_stats(self, stats_file):
        """Read function name from stats file."""
        if not os.path.exists(stats_file):
            return None
        logger.info(f"Reading stats file for function name: {stats_file}")
        try:
            with open(stats_file, 'r') as fid:
                for line in fid.readlines():
                    try:
                        key, value = line.strip().split(" ", 1)
                        if key == 'function_name':
                            self.function_name = value
                            logger.info(f"Found function name in stats file: {self.function_name}")
                            return self.function_name
                    except Exception as e:
                        logger.error(f"Error processing stats file line: {line} - {e}")
        except Exception as e:
            logger.error(f"Error reading stats file: {e}")
        return None

    def process_energy_data(self, task, call_status, cpu_info):
        """Process energy data from all monitors and add to call status."""
        avg_cpu_usage = sum(cpu_info['usage']) / len(cpu_info['usage']) if cpu_info['usage'] else 0
        energy_consumption = avg_cpu_usage * round(cpu_info['user'], 8)

        energy_fields = {
            'worker_func_energy_duration': 0.0,
            'worker_func_rapl_energy_pkg': 0.0,
            'worker_func_rapl_energy_cores': 0.0,
            'worker_func_rapl_energy_total': 0.0,
            'worker_func_rapl_source': 'unavailable',
            'worker_func_rapl_available': False,
            'worker_func_psutil_energy_pkg': 0.0,
            'worker_func_psutil_energy_cores': 0.0,
            'worker_func_psutil_avg_cpu_percent': 0.0,
            'worker_func_psutil_source': 'unavailable',
            'worker_func_psutil_available': False,
            'worker_func_psutil_cpu_model': 'Unknown',
            'worker_func_psutil_cpu_architecture': 'Unknown',
            'worker_func_psutil_cpu_tdp_ref': 0.0,
            'worker_func_avg_cpu_usage': avg_cpu_usage,
            'worker_func_energy_consumption': energy_consumption,
        }

        import platform
        energy_fields['worker_func_psutil_cpu_architecture'] = platform.machine()

        max_duration = 0.0
        for method_name, monitor in self.monitors.items():
            if monitor is not None and self.monitor_status.get(method_name, False):
                try:
                    energy_data = monitor.get_energy_data()
                    duration = energy_data.get('duration', 0.0)
                    max_duration = max(max_duration, duration)

                    energy = energy_data.get('energy', {})
                    pkg_energy = energy.get('pkg', 0.0)
                    cores_energy = energy.get('cores', 0.0)
                    total_energy = pkg_energy + cores_energy if (pkg_energy > 0 or cores_energy > 0) else 0.0
                    source = energy_data.get('source', 'unknown')

                    if method_name == 'rapl':
                        energy_fields['worker_func_rapl_energy_pkg'] = pkg_energy
                        energy_fields['worker_func_rapl_energy_cores'] = cores_energy
                        energy_fields['worker_func_rapl_energy_total'] = total_energy
                        energy_fields['worker_func_rapl_source'] = source
                        energy_fields['worker_func_rapl_available'] = True

                    elif method_name == 'psutil':
                        cpu_info_data = energy_data.get('cpu_info', {})
                        energy_fields['worker_func_psutil_energy_pkg'] = pkg_energy
                        energy_fields['worker_func_psutil_energy_cores'] = cores_energy
                        energy_fields['worker_func_psutil_avg_cpu_percent'] = energy.get('avg_cpu_percent', 0.0)
                        energy_fields['worker_func_psutil_source'] = source
                        energy_fields['worker_func_psutil_available'] = True
                        energy_fields['worker_func_psutil_cpu_model'] = cpu_info_data.get('model', 'Unknown')
                        energy_fields['worker_func_psutil_cpu_architecture'] = cpu_info_data.get('arch', energy_fields['worker_func_psutil_cpu_architecture'])
                        energy_fields['worker_func_psutil_cpu_tdp_ref'] = cpu_info_data.get('tdp_ref', 0.0)
                        logger.info(f"Collected CPU info from PSUtil: {energy_fields['worker_func_psutil_cpu_model']}")

                    try:
                        monitor.log_energy_data(energy_data, task, cpu_info, self.function_name)
                    except Exception as log_e:
                        logger.warning(f"Failed to log energy data for {method_name}: {log_e}")

                except Exception as e:
                    logger.error(f"Error processing energy data from {method_name}: {e}")

        energy_fields['worker_func_energy_duration'] = max_duration

        for field_name, field_value in energy_fields.items():
            call_status.add(field_name, field_value)

        aws_processor_info = self._get_aws_processor_info()
        for key, value in aws_processor_info.items():
            call_status.add(f'worker_func_aws_{key}', value)

        method_order = ['rapl', 'psutil']
        available_methods = []
        for method in method_order:
            if self.monitor_status.get(method, False) and self.monitors.get(method) is not None:
                available_methods.append(method)
            else:
                available_methods.append('null')
        energy_method_used = ', '.join(available_methods) if available_methods else 'n/a'
        call_status.add('worker_func_energy_method_used', energy_method_used)

        active_methods = [m for m, s in self.monitor_status.items() if s and self.monitors[m] is not None]
        logger.info(f"Energy data collected from {len(active_methods)} methods: {active_methods}")
        logger.info(f"Energy method used: {energy_method_used}")

        non_zero_fields = {k: v for k, v in energy_fields.items() if isinstance(v, (int, float)) and v > 0}
        if non_zero_fields:
            logger.debug(f"Non-zero energy values: {non_zero_fields}")

    def update_function_name(self, task, cpu_info, stats_file):
        """Update function name in energy data for all monitors if available."""
        if not any(self.monitor_status.values()):
            return
        if not os.path.exists(stats_file):
            logger.warning("Stats file not found for updating function name")
            return
        function_name = self.read_function_name_from_stats(stats_file)
        if not function_name:
            logger.warning("Function name not found in stats file for energy monitoring")
            return
        logger.info(f"Updating function name in energy data: {function_name}")
        for method_name, monitor in self.monitors.items():
            if monitor is not None and self.monitor_status.get(method_name, False):
                try:
                    if hasattr(monitor, 'update_function_name'):
                        monitor.update_function_name(task, function_name)
                    elif hasattr(monitor, '_store_energy_data_json'):
                        monitor._store_energy_data_json(
                            monitor.get_energy_data(),
                            task,
                            cpu_info,
                            function_name
                        )
                except Exception as e:
                    logger.warning(f"Failed to update function name for {method_name}: {e}")