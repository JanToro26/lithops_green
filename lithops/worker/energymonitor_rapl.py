import os
import time
import glob
import logging
#from .energymonitor_json_utils import store_energy_data_json, update_function_name

logger = logging.getLogger(__name__)


class EnergyMonitor:
    """
    Energy monitor that uses direct RAPL access via /sys/class/powercap/
    This bypasses the need for perf and perf_event_paranoid restrictions.
    """
    def __init__(self, process_id):
        self.process_id = process_id
        self.start_time = None
        self.end_time = None
        self.start_energy_pkg = None
        self.start_energy_cores = None
        self.end_energy_pkg = None
        self.end_energy_cores = None
        self.function_name = None
        self.rapl_pkg_files = []
        self.rapl_cores_files = []
        
        # Print directly to terminal for debugging
        print(f"\n==== RAPL ENERGY MONITOR INITIALIZED FOR PROCESS {process_id} ====")
        
        # Find available RAPL files
        self._find_rapl_files()
        
    def _find_rapl_files(self):
        """Find available RAPL energy files."""
        print("\n==== FINDING RAPL ENERGY FILES ====")
        
        # Look for package energy files (main CPU energy)
        pkg_patterns = [
            '/sys/class/powercap/intel-rapl:*/energy_uj',
            '/sys/class/powercap/intel-rapl:*:*/energy_uj'
        ]
        
        for pattern in pkg_patterns:
            files = glob.glob(pattern)
            for file in files:
                try:
                    # Test if we can read the file
                    with open(file, 'r') as f:
                        value = f.read().strip()
                        int(value)  # Verify it's a valid number
                    
                    # Determine if it's a package or core file
                    if ':0:' in file or file.endswith(':0/energy_uj'):
                        # This is likely a core/uncore file
                        self.rapl_cores_files.append(file)
                        print(f"✅ Found RAPL cores file: {file}")
                    else:
                        # This is likely a package file
                        self.rapl_pkg_files.append(file)
                        print(f"✅ Found RAPL package file: {file}")
                        
                except Exception as e:
                    print(f"❌ Cannot read RAPL file {file}: {e}")
        
        print(f"Total RAPL package files: {len(self.rapl_pkg_files)}")
        print(f"Total RAPL cores files: {len(self.rapl_cores_files)}")
        
    def _read_rapl_energy(self, files):
        """Read energy from RAPL files and return total in microjoules."""
        total_energy = 0
        for file in files:
            try:
                with open(file, 'r') as f:
                    value = int(f.read().strip())
                    total_energy += value
            except Exception as e:
                print(f"❌ Error reading {file}: {e}")
        return total_energy
        
    def start(self):
        """Start monitoring energy consumption using RAPL."""
        print("\n==== STARTING RAPL ENERGY MONITORING ====")
        
        if not self.rapl_pkg_files:
            print("❌ No RAPL package files found, cannot start monitoring")
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
            
            duration = self.end_time - self.start_time
            print(f"RAPL monitoring stopped at: {self.end_time}")
            print(f"Monitoring duration: {duration:.2f} seconds")
            print(f"Final package energy: {self.end_energy_pkg} microjoules")
            print(f"Final cores energy: {self.end_energy_cores} microjoules")
            
            # Calculate energy differences
            pkg_diff = self.end_energy_pkg - self.start_energy_pkg
            cores_diff = self.end_energy_cores - self.start_energy_cores
            
            print(f"Package energy consumed: {pkg_diff} microjoules ({pkg_diff / 1000000:.6f} Joules)")
            print(f"Cores energy consumed: {cores_diff} microjoules ({cores_diff / 1000000:.6f} Joules)")
            
        except Exception as e:
            print(f"❌ Error stopping RAPL monitoring: {e}")
            
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
        pkg_energy_uj = self.end_energy_pkg - self.start_energy_pkg
        cores_energy_uj = self.end_energy_cores - self.start_energy_cores
        
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
    
    