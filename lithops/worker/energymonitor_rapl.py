import os
import time
import glob
import logging
from .energymonitor_json_utils import store_energy_data_json, update_function_name

logger = logging.getLogger(__name__)


class EnergyMonitor:
    """
    Energy monitor that uses direct RAPL access via /sys/class/powercap/
    This bypasses the need for perf and perf_event_paranoid restrictions.
    """

    # Domains that must never be added to any total. `psys` is the
    # platform-level counter and a SUPERSET of the package domains, so counting
    # it alongside package-N would report the same energy twice.
    _EXCLUDED_DOMAINS = ('psys',)

    def __init__(self, process_id):
        self.process_id = process_id
        self.start_time = None
        self.end_time = None
        #self.start_energy_pkg = None
        #self.start_energy_cores = None
        #self.end_energy_pkg = None
        #self.end_energy_cores = None
        self.function_name = None
        self.rapl_pkg_files = []
        self.rapl_cores_files = []
        # Domains that are neither package nor core (dram, uncore, ...). Kept
        # for diagnostics only: dram is a sibling of the package, not part of
        # it, and adding it to a "package energy" figure would change what the
        # number means.
        self.rapl_other_files = {}
        self.rapl_max_range = {}   # {ruta_energy_uj: max_energy_range_uj}
        self.start_energy_pkg = {} # dictionaries to hold start and end energy readings for each file
        self.start_energy_cores = {} # this way we can keep track of the real value even after a wrap-around
        self.end_energy_pkg = {}
        self.end_energy_cores = {}
        
        # Print directly to terminal for debugging
        print(f"\n==== RAPL ENERGY MONITOR INITIALIZED FOR PROCESS {process_id} ====")
        
        # Find available RAPL files
        self._find_rapl_files()
        
    @staticmethod
    def _read_sysfs(path):
        """Read a one-line sysfs attribute, stripped. Raises on failure."""
        with open(path, 'r') as f:
            return f.read().strip()

    def _find_rapl_files(self):
        """
        Discover the readable RAPL domains and classify them by what they
        actually measure.

        Classification reads each domain's `name` attribute ('package-0',
        'core', 'dram', 'uncore', 'psys'). The previous version inferred the
        kind from the directory suffix instead, with

            if ':0:' in file or file.endswith(':0/energy_uj'):  -> cores

        which matches `intel-rapl:0`, the PACKAGE of socket 0. On a
        single-socket host -- every machine in the target cluster -- that left
        rapl_pkg_files empty, so start() returned False and RAPL never ran at
        all. The failure was silent: the manager logs a warning and falls
        through to the modelled estimate, so the run still produces numbers.

        One glob, not two. /sys/class/powercap exposes every domain as a
        flat entry, so `intel-rapl:*` already covers `intel-rapl:0:1`; the old
        second pattern re-matched the subdomains, and each one was appended
        twice. Readings are keyed by realpath so any remaining alias collapses
        to a single entry.
        """
        logger.debug("Discovering RAPL domains under /sys/class/powercap")

        seen = set()
        for domain_dir in sorted(glob.glob('/sys/class/powercap/intel-rapl:*')):
            energy_file = os.path.join(domain_dir, 'energy_uj')

            try:
                canonical = os.path.realpath(energy_file)
            except OSError:
                canonical = energy_file
            if canonical in seen:
                continue

            try:
                int(self._read_sysfs(energy_file))
            except Exception as e:
                # Unreadable since Linux 5.10, which restricted energy_uj to
                # root (CVE-2020-8694). Not an error worth raising: the manager
                # degrades to the next mechanism.
                logger.debug(f"RAPL counter not readable: {energy_file} ({e})")
                continue
            seen.add(canonical)

            try:
                domain = self._read_sysfs(os.path.join(domain_dir, 'name')).lower()
            except Exception as e:
                logger.warning(
                    f"RAPL domain {domain_dir} exposes no readable 'name'; "
                    f"skipped rather than guessed ({e})"
                )
                continue

            if domain in self._EXCLUDED_DOMAINS:
                logger.debug(f"Skipping RAPL domain '{domain}' ({domain_dir})")
                continue

            try:
                self.rapl_max_range[energy_file] = int(
                    self._read_sysfs(os.path.join(domain_dir, 'max_energy_range_uj'))
                )
            except Exception:
                # Left as None on purpose. _compute_diff refuses to invent a
                # wrap range: a fabricated constant would turn a counter wrap
                # into a plausible-looking but wrong energy figure.
                self.rapl_max_range[energy_file] = None
                logger.warning(
                    f"RAPL domain '{domain}' exposes no max_energy_range_uj; a "
                    f"counter wrap in this domain cannot be corrected."
                )

            if domain.startswith('package'):
                self.rapl_pkg_files.append(energy_file)
            elif domain == 'core':
                self.rapl_cores_files.append(energy_file)
            else:
                self.rapl_other_files.setdefault(domain, []).append(energy_file)
            logger.debug(f"RAPL domain '{domain}' -> {energy_file}")

        logger.info(
            f"RAPL domains found: {len(self.rapl_pkg_files)} package, "
            f"{len(self.rapl_cores_files)} core, "
            f"{sum(len(v) for v in self.rapl_other_files.values())} other "
            f"({', '.join(sorted(self.rapl_other_files)) or 'none'})"
        )
        if not self.rapl_pkg_files and self.rapl_cores_files:
            logger.warning(
                "No RAPL package domain is readable; core energy will be used "
                "as the total. It excludes uncore and is therefore a lower bound."
            )


    def _read_rapl_energy(self, files):
        """Read energy from RAPL files and return a dict {file: microjoules}."""
        readings = {}
        for file in files:
            try:
                with open(file, 'r') as f:
                    readings[file] = int(f.read().strip())
            except Exception as e:
                print(f"❌ Error reading {file}: {e}")
        return readings
        
    def start(self):
        """Start monitoring energy consumption using RAPL."""
        print("\n==== STARTING RAPL ENERGY MONITORING ====")
        
        # Start on ANY readable domain, not on a package domain only. A host
        # that exposes core but no package is unusual but not useless, and the
        # manager already resolves the total as `pkg if pkg > 0 else cores`.
        if not self.rapl_pkg_files and not self.rapl_cores_files:
            logger.info("No readable RAPL domain; RAPL monitoring disabled")
            return False

        try:
            self.start_time = time.time()
            self.start_energy_pkg = self._read_rapl_energy(self.rapl_pkg_files)
            self.start_energy_cores = self._read_rapl_energy(self.rapl_cores_files)
            
            print(f"✅ RAPL monitoring started at: {self.start_time}")
            print(f"Initial package energy: {self.start_energy_pkg} microjoules")
            print(f"Initial cores energy: {self.start_energy_cores} microjoules")
            return True
            
        except Exception as e:
            print(f"❌ Error starting RAPL monitoring: {e}")
            return False
            
    def stop(self):
        """Stop monitoring energy consumption and collect results."""
        print("\n==== STOPPING RAPL ENERGY MONITORING ====")
        
        if self.start_time is None:
            print("❌ RAPL monitoring was not started")
            return
            
        try:
            self.end_time = time.time()
            self.end_energy_pkg = self._read_rapl_energy(self.rapl_pkg_files)
            self.end_energy_cores = self._read_rapl_energy(self.rapl_cores_files)
            
            pkg_diff   = self._compute_diff(self.start_energy_pkg,   self.end_energy_pkg)
            cores_diff = self._compute_diff(self.start_energy_cores, self.end_energy_cores)
            
            duration = self.end_time - self.start_time
            print(f"RAPL monitoring stopped at: {self.end_time}")
            print(f"Monitoring duration: {duration:.2f} seconds")
            print(f"Final package energy: {self.end_energy_pkg} microjoules")
            print(f"Final cores energy: {self.end_energy_cores} microjoules")
            
            # Calculate energy differences NOT NECESSARY ANYMORE since we compute the diff in _compute_diff for each file
            # pkg_diff = self.end_energy_pkg - self.start_energy_pkg  
            # if pkg_diff < 0:                                # Fixed to prevent negative values from overflow
            #    pkg_diff += self.max_energy_range_pkg
            #cores_diff = self.end_energy_cores - self.start_energy_cores
            
            print(f"Package energy consumed: {pkg_diff} microjoules ({pkg_diff / 1000000:.6f} Joules)")
            print(f"Cores energy consumed: {cores_diff} microjoules ({cores_diff / 1000000:.6f} Joules)")
            
        except Exception as e:
            print(f"❌ Error stopping RAPL monitoring: {e}")
            
    def _compute_diff(self, start_readings, end_readings):
        """
        Sum the per-domain energy consumed between the two readings, in
        microjoules, correcting each domain's own counter wrap.

        Three failure modes that previously raised inside get_energy_data --
        where the exception was caught and turned into a zero -- are handled
        explicitly here:

          * a domain present at stop but not at start (KeyError). It is skipped:
            a partial interval cannot be attributed to this window.
          * a wrap in a domain whose max_energy_range_uj was unreadable
            (TypeError on None). It is skipped and warned about, rather than
            corrected with an invented range.
          * a wrap that even the domain's own range cannot explain, which means
            more than one wrap occurred and the true value is unrecoverable.
        """
        total = 0
        for file, end_value in end_readings.items():
            start_value = start_readings.get(file)
            if start_value is None:
                logger.warning(
                    f"RAPL domain {file} has no start reading; excluded from this window"
                )
                continue

            diff = end_value - start_value
            if diff < 0:
                max_range = self.rapl_max_range.get(file)
                if max_range is None:
                    logger.warning(
                        f"RAPL counter {file} wrapped and its max_energy_range_uj "
                        f"is unknown; excluded from this window"
                    )
                    continue
                diff += max_range
                if diff < 0:
                    logger.warning(
                        f"RAPL counter {file} wrapped more than once; excluded "
                        f"from this window"
                    )
                    continue
            total += diff
        return total

    def get_energy_data(self):
        """Get the collected energy data from RAPL."""
        print("\n==== GETTING RAPL ENERGY DATA ====")
        
        if self.start_time is None or self.end_time is None:
            print("❌ RAPL monitoring was not completed")
            return {
                'energy': {'pkg': 0, 'cores': 0, 'core_percentage': 0},
                'duration': 0,
                'source': 'none'
            }
        
        duration = self.end_time - self.start_time
        
        # Calculate energy differences in Joules
        #pkg_energy_uj = self.end_energy_pkg - self.start_energy_pkg
        #cores_energy_uj = self.end_energy_cores - self.start_energy_cores
        pkg_energy_uj   = self._compute_diff(self.start_energy_pkg,   self.end_energy_pkg)
        cores_energy_uj = self._compute_diff(self.start_energy_cores, self.end_energy_cores)
        
        pkg_energy_j = pkg_energy_uj / 1000000.0  # Convert microjoules to Joules
        cores_energy_j = cores_energy_uj / 1000000.0
        
        
        # Calculate core percentage
        core_percentage = cores_energy_j / max(pkg_energy_j, 0.000001)
        
        result = {
            'energy': {
                'pkg': pkg_energy_j,
                'cores': cores_energy_j,
                'core_percentage': core_percentage
            },
            'duration': duration,
            'source': 'rapl_direct'
        }
        
        print(f"✅ RAPL energy data collected:")
        print(f"  Package: {pkg_energy_j:.6f} Joules")
        print(f"  Cores: {cores_energy_j:.6f} Joules")
        print(f"  Core percentage: {core_percentage:.4f} ({core_percentage * 100:.2f}%)")
        print(f"  Duration: {duration:.2f} seconds")
        
        return result
        
    def log_energy_data(self, energy_data, task, cpu_info, function_name=None):
        """Log energy data and store it in JSON format."""
        import json
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Store function name if provided
        if function_name:
            self.function_name = function_name
        
        # Log energy consumption
        logger.info(f"RAPL Energy consumption: {energy_data['energy'].get('pkg', 'N/A')} Joules (pkg), {energy_data['energy'].get('cores', 'N/A')} Joules (cores)")
        logger.info(f"Core percentage: {energy_data['energy'].get('core_percentage', 0) * 100:.2f}%")
        logger.info(f"Energy efficiency: {energy_data['energy'].get('pkg', 0) / max(energy_data['duration'], 0.001):.2f} Watts")
        
        # Print energy data in the format requested by the user
        print("\nPerformance counter stats for 'system wide' (RAPL):")
        print()
        
        # Get the actual measured values from RAPL
        pkg_energy = energy_data['energy'].get('pkg', 0)
        cores_energy = energy_data['energy'].get('cores', 0)
        
        # Format and print the values
        pkg_energy_str = f"{pkg_energy:,.6f}".replace(",", "X").replace(".", ",").replace("X", ".")
        print(f"          {pkg_energy_str} Joules power/energy-pkg/ (RAPL)")
        
        cores_energy_str = f"{cores_energy:,.6f}".replace(",", "X").replace(".", ",").replace("X", ".")
        print(f"          {cores_energy_str} Joules power/energy-cores/ (RAPL)")
        
        # Print core percentage
        if pkg_energy > 0:
            core_percentage = cores_energy / pkg_energy
        else:
            core_percentage = 0
        print(f"          {core_percentage * 100:.2f}% core percentage (cores/pkg)")
        print()
        
        # Store energy consumption data in JSON format using shared utilities
        monitor_specific_data = {
            'rapl_pkg_files': self.rapl_pkg_files,  # Track which files were used
            'rapl_cores_files': self.rapl_cores_files,
        }
        store_energy_data_json(energy_data, task, cpu_info, pkg_energy, cores_energy, 
                              core_percentage, function_name, monitor_specific_data)
        
    '''
    def update_function_name(self, task, function_name):
        """Update the function name in the JSON files."""
        # Store function name
        self.function_name = function_name
        
        # Use shared utility function
        update_function_name(task, function_name)
    
    '''