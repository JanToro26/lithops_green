#
# (C) Copyright IBM Corp. 2020
# (C) Copyright Cloudlab URV 2020
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import time
import re
import subprocess
import logging
from .energymonitor_json_utils import store_energy_data_json, update_function_name

logger = logging.getLogger(__name__)


class EnergyMonitor:
    """
    Energy monitor that gets REAL perf energy values by trying different event combinations.
    Prioritizes getting actual hardware measurements over estimates.
    """
    def __init__(self, process_id):
        self.process_id = process_id
        self.perf_process = None
        self.start_time = None
        self.end_time = None
        self.energy_value = None
        self.cpu_percent = None
        self.perf_output_file = f"/tmp/perf_energy_{process_id}.txt"
        self.function_name = None
        self.energy_events_used = None
        
        # Print directly to terminal for debugging
        print(f"\n==== ENERGY MONITOR INITIALIZED FOR PROCESS {process_id} ====")
        
    def _find_working_perf_binary(self):
        """Find a working perf binary for the current system."""
        import os
        import subprocess
        
        # List of possible perf binary locations
        perf_paths = [
            '/usr/bin/perf',
            '/usr/lib/linux-tools/6.8.0-79-generic/perf',
            '/usr/lib/linux-tools/6.8.0-78-generic/perf',
            '/usr/lib/linux-tools/6.8.0-52-generic/perf',
        ]
        
        # Also try to find perf tools for current kernel
        kernel_version = os.uname().release
        kernel_perf_path = f'/usr/lib/linux-tools/{kernel_version}/perf'
        if kernel_perf_path not in perf_paths:
            perf_paths.insert(1, kernel_perf_path)
        
        for perf_path in perf_paths:
            if os.path.exists(perf_path):
                try:
                    # Test if this perf binary works
                    result = subprocess.run([perf_path, '--version'], 
                                          capture_output=True, timeout=5)
                    if result.returncode == 0:
                        print(f"✅ Found working perf binary: {perf_path}")
                        return perf_path
                except Exception as e:
                    print(f"❌ Error testing perf binary {perf_path}: {e}")
                    continue
                    
        print("❌ No working perf binary found")
        return None
        
    def _test_energy_event(self, event_string):
        """Test if a specific energy event string works with perf."""
        print(f"\n==== TESTING ENERGY EVENT: {event_string} ====")
        try:
            # Find the correct perf binary
            perf_binary = self._find_working_perf_binary()
            if not perf_binary:
                print("❌ No working perf binary found")
                return False
                
            # Try without sudo first (since perf_event_paranoid = -1)
            cmd = f"{perf_binary} stat -e {event_string} -a sleep 0.1 2>&1"
            print(f"Testing command (without sudo): {cmd}")
            
            result = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=5
            )
            
            output = result.stdout
            print(f"Test result (return code {result.returncode}): {output[:200]}...")
            
            # Check if the command succeeded and contains energy data
            if result.returncode == 0 and "Joules" in output and "event syntax error" not in output:
                print(f"✅ SUCCESS: Event {event_string} works without sudo!")
                return True
            
            # If without sudo failed, try with sudo as fallback
            print("❌ Failed without sudo, trying with sudo...")
            cmd_sudo = f"sudo {perf_binary} stat -e {event_string} -a sleep 0.1 2>&1"
            print(f"Testing command (with sudo): {cmd_sudo}")
            
            result_sudo = subprocess.run(
                cmd_sudo,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=5
            )
            
            output_sudo = result_sudo.stdout
            print(f"Sudo test result (return code {result_sudo.returncode}): {output_sudo[:200]}...")
            
            if result_sudo.returncode == 0 and "Joules" in output_sudo and "event syntax error" not in output_sudo:
                print(f"✅ SUCCESS: Event {event_string} works with sudo!")
                return True
            else:
                print(f"❌ FAILED: Event {event_string} doesn't work even with sudo")
                return False
                
        except Exception as e:
            print(f"❌ ERROR testing event {event_string}: {e}")
            return False
    
    def _get_working_energy_events(self):
        """Get energy events that actually work on this system."""
        print("\n==== FINDING WORKING ENERGY EVENTS ====")
        
        # List of event combinations to try, in order of preference
        event_combinations = [
            "power/energy-pkg/,power/energy-cores/",  # Try both first (this is what we want)
            "power/energy-pkg/",  # Try pkg only as fallback
            "power/energy-cores/",  # Try cores only
            "energy-pkg",  # Alternative format
            "energy-cores",  # Alternative format
        ]
        
        for events in event_combinations:
            print(f"\nTrying event combination: {events}")
            if self._test_energy_event(events):
                print(f"✅ Found working energy events: {events}")
                return events
        
        print("❌ No working energy events found")
        return None
        
    def start(self):
        """Start monitoring energy consumption using perf."""
        print("\n==== STARTING ENERGY MONITORING ====")
        try:
            # Find working energy events
            self.energy_events_used = self._get_working_energy_events()
            
            if not self.energy_events_used:
                print("❌ No working energy events found, cannot start monitoring")
                return False
            
            print(f"Using energy events: {self.energy_events_used}")
            
            # Create a unique output file for this run
            self.perf_output_file = f"/tmp/perf_energy_{self.process_id}_{int(time.time())}.txt"
            
            # Find the correct perf binary
            perf_binary = self._find_working_perf_binary()
            if not perf_binary:
                print("❌ No working perf binary found")
                return False
            
            # Start perf in the background to monitor the entire function execution
            print("Starting perf stat to monitor energy consumption...")
            
            # Try without sudo first (since perf_event_paranoid = -1)
            cmd = [
                perf_binary, "stat",
                "-e", self.energy_events_used,
                "-a",  # Monitor all CPUs
                "-o", self.perf_output_file  # Output to a file
            ]
            
            print(f"Running command (without sudo): {' '.join(cmd)}")
            
            # Start perf in the background
            try:
                self.perf_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Give it a moment to start and check if it's working
                time.sleep(0.1)
                if self.perf_process.poll() is not None:
                    # Process already exited, try with sudo
                    print("❌ Perf without sudo failed, trying with sudo...")
                    cmd_sudo = [
                        "sudo", perf_binary, "stat",
                        "-e", self.energy_events_used,
                        "-a",  # Monitor all CPUs
                        "-o", self.perf_output_file  # Output to a file
                    ]
                    
                    print(f"Running command (with sudo): {' '.join(cmd_sudo)}")
                    self.perf_process = subprocess.Popen(
                        cmd_sudo,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                else:
                    print("✅ Perf started successfully without sudo!")
                    
            except Exception as e:
                print(f"❌ Error starting perf without sudo: {e}")
                print("Trying with sudo as fallback...")
                cmd_sudo = [
                    "sudo", perf_binary, "stat",
                    "-e", self.energy_events_used,
                    "-a",  # Monitor all CPUs
                    "-o", self.perf_output_file  # Output to a file
                ]
                
                print(f"Running command (with sudo): {' '.join(cmd_sudo)}")
                self.perf_process = subprocess.Popen(
                    cmd_sudo,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            
            self.start_time = time.time()
            print(f"✅ Energy monitoring started at: {self.start_time}")
            print(f"Perf process PID: {self.perf_process.pid}")
            return True
        except Exception as e:
            print(f"❌ Error starting energy monitoring: {e}")
            return False
            
    def stop(self):
        """Stop monitoring energy consumption and collect results."""
        print("\n==== STOPPING ENERGY MONITORING ====")
        
        if self.perf_process is None:
            print("No perf process to stop")
            return
            
        try:
            # Record the end time
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            print(f"Energy monitoring stopped at: {self.end_time}")
            print(f"Monitoring duration: {duration:.2f} seconds")
            
            # Stop the perf process
            print(f"Stopping perf process (PID: {self.perf_process.pid})...")
            
            # Send SIGINT to perf to make it output the results
            import signal
            os.kill(self.perf_process.pid, signal.SIGINT)
            
            # Wait for the process to exit
            try:
                stdout, stderr = self.perf_process.communicate(timeout=5)
                print("Perf process exited")
                print(f"Perf stdout: {stdout}")
                print(f"Perf stderr: {stderr}")
            except subprocess.TimeoutExpired:
                print("Perf process did not exit, killing it")
                self.perf_process.kill()
                stdout, stderr = self.perf_process.communicate()
            
            # Initialize energy values
            self.energy_pkg = None
            self.energy_cores = None
            
            # Read the output file
            print(f"Reading perf output file: {self.perf_output_file}")
            try:
                if os.path.exists(self.perf_output_file):
                    with open(self.perf_output_file, 'r') as f:
                        perf_output = f.read()
                        print(f"Perf output file content: {perf_output}")
                        
                        # Process the output to extract energy values
                        for line in perf_output.splitlines():
                            print(f"Processing line: {line}")
                            if "Joules" in line:
                                # Use a more precise regex to extract the energy value
                                match = re.search(r'\s*([\d,.]+)\s+Joules\s+(\S+)', line)
                                if match:
                                    value_str = match.group(1).replace(',', '.')
                                    event_name = match.group(2)
                                    try:
                                        # Handle numbers with multiple dots (e.g. 1.043.75 -> 1043.75)
                                        if value_str.count('.') > 1:
                                            parts = value_str.split('.')
                                            value_str = ''.join(parts[:-1]) + '.' + parts[-1]
                                            print(f"Converted malformed value with multiple dots to {value_str}")
                                        
                                        energy_value = float(value_str)
                                        print(f"✅ Found energy value: {energy_value} Joules for {event_name}")
                                        
                                        # Store the value based on the event type
                                        if "energy-pkg" in event_name:
                                            self.energy_pkg = energy_value
                                            print(f"✅ Stored energy-pkg value: {self.energy_pkg} Joules")
                                        elif "energy-cores" in event_name:
                                            self.energy_cores = energy_value
                                            print(f"✅ Stored energy-cores value: {self.energy_cores} Joules")
                                    except ValueError as e:
                                        print(f"Could not convert '{value_str}' to float: {e}")
                else:
                    print(f"❌ Perf output file not found: {self.perf_output_file}")
            except Exception as e:
                print(f"❌ Error reading perf output file: {e}")
            
            # If we couldn't get energy values from the file, try to get them from stderr
            if (self.energy_pkg is None and self.energy_cores is None) and stderr:
                print("Trying to extract energy values from stderr...")
                for line in stderr.splitlines():
                    print(f"Processing stderr line: {line}")
                    if "Joules" in line:
                        match = re.search(r'\s*([\d,.]+)\s+Joules\s+(\S+)', line)
                        if match:
                            value_str = match.group(1).replace(',', '.')
                            event_name = match.group(2)
                            try:
                                if value_str.count('.') > 1:
                                    parts = value_str.split('.')
                                    value_str = ''.join(parts[:-1]) + '.' + parts[-1]
                                
                                energy_value = float(value_str)
                                print(f"✅ Found energy value in stderr: {energy_value} Joules for {event_name}")
                                
                                if "energy-pkg" in event_name:
                                    self.energy_pkg = energy_value
                                    print(f"✅ Stored energy-pkg value: {self.energy_pkg} Joules")
                                elif "energy-cores" in event_name:
                                    self.energy_cores = energy_value
                                    print(f"✅ Stored energy-cores value: {self.energy_cores} Joules")
                            except ValueError as e:
                                print(f"Could not convert '{value_str}' to float: {e}")
            
            # Clean up the output file
            try:
                if os.path.exists(self.perf_output_file):
                    os.remove(self.perf_output_file)
                    print(f"Removed perf output file: {self.perf_output_file}")
            except Exception as e:
                print(f"Error removing perf output file: {e}")
                
        except Exception as e:
            print(f"❌ Error stopping energy monitoring: {e}")
            
    def get_energy_data(self):
        """Get the collected energy data for energy-pkg and energy-cores."""
        print("\n==== GETTING ENERGY DATA ====")
        duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        print(f"Duration: {duration:.2f} seconds")
        
        # Create the base result dictionary
        result = {
            'energy': {},
            'duration': duration,
            'source': 'perf'
        }
        
        # Add energy values if available from perf
        if self.energy_pkg is not None and self.energy_pkg > 0:
            print(f"✅ Using measured energy-pkg: {self.energy_pkg:.2f} Joules")
            result['energy']['pkg'] = self.energy_pkg
        else:
            print("❌ No energy-pkg data from perf, setting to 0")
            result['energy']['pkg'] = 0
            
        if self.energy_cores is not None and self.energy_cores > 0:
            print(f"✅ Using measured energy-cores: {self.energy_cores:.2f} Joules")
            result['energy']['cores'] = self.energy_cores
        else:
            print("❌ No energy-cores data from perf, setting to 0")
            result['energy']['cores'] = 0
            
        # Calculate core percentage (energy_cores / energy_pkg)
        if result['energy']['pkg'] > 0:
            core_percentage = result['energy']['cores'] / result['energy']['pkg']
            print(f"Core percentage: {core_percentage:.4f} ({core_percentage * 100:.2f}%)")
            result['energy']['core_percentage'] = core_percentage
        else:
            print("Cannot calculate core percentage, energy_pkg is 0")
            result['energy']['core_percentage'] = 0
        
        # If we have no energy data at all, set source to 'none'
        if result['energy']['pkg'] == 0 and result['energy']['cores'] == 0:
            result['source'] = 'none'
            print("❌ No energy data available from perf")
        else:
            print(f"✅ Energy data source: {result['source']}")

        print(f"Final energy data: {result}")
        return result
        
    def log_energy_data(self, energy_data, task, cpu_info, function_name=None):
        """Log energy data and store it in JSON format - NO ESTIMATION, ONLY REAL VALUES."""
        import json
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Store function name if provided
        if function_name:
            self.function_name = function_name
        
        # Log energy consumption
        logger.info(f"Energy consumption: {energy_data['energy'].get('pkg', 'N/A')} Joules (pkg), {energy_data['energy'].get('cores', 'N/A')} Joules (cores)")
        logger.info(f"Core percentage: {energy_data['energy'].get('core_percentage', 0) * 100:.2f}%")
        logger.info(f"Energy efficiency: {energy_data['energy'].get('pkg', 0) / max(energy_data['duration'], 0.001):.2f} Watts")
        
        # Print energy data in the format requested by the user
        print("\nPerformance counter stats for 'system wide':")
        print()
        
        # Get the actual measured values (NO ESTIMATION)
        pkg_energy = energy_data['energy'].get('pkg', 0)
        cores_energy = energy_data['energy'].get('cores', 0)
        
        # Format and print the values
        pkg_energy_str = f"{pkg_energy:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        print(f"          {pkg_energy_str} Joules power/energy-pkg/")
        
        cores_energy_str = f"{cores_energy:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        print(f"          {cores_energy_str} Joules power/energy-cores/")
        
        # Print core percentage
        if pkg_energy > 0:
            core_percentage = cores_energy / pkg_energy
        else:
            core_percentage = 0
        print(f"          {core_percentage * 100:.2f}% core percentage (cores/pkg)")
        print()
        
        # Store energy consumption data in JSON format using shared utilities
        monitor_specific_data = {
            'events_used': self.energy_events_used,  # Track which events were used
        }
        store_energy_data_json(energy_data, task, cpu_info, pkg_energy, cores_energy, 
                              core_percentage, function_name, monitor_specific_data)
        
    def update_function_name(self, task, function_name):
        """Update the function name in the JSON files."""
        # Store function name
        self.function_name = function_name
        
        # Use shared utility function
        update_function_name(task, function_name)
