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
import json
import time
import sys
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def _get_default_energy_data_path():
    """
    Determine the best default path for energy data JSON files.
    This function works both inside Kubernetes pods and on the host system.
    """
    # Target path we want to use
    target_path = "/home/users/iarriazu/lithops_fork/inigo_test/energy_data"
    
    # Check if we can write to the target path directly
    try:
        os.makedirs(target_path, exist_ok=True)
        # Test write access
        test_file = os.path.join(target_path, ".write_test")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print(f"✅ Using target energy data path: {target_path}", file=sys.stderr)
        return target_path
    except Exception as e:
        print(f"❌ Cannot write to target path {target_path}: {e}", file=sys.stderr)
    
    # If we're in a Kubernetes pod, try to create a path that might be accessible
    # Check if we're in a container/pod environment
    if os.path.exists('/proc/1/cgroup'):
        try:
            with open('/proc/1/cgroup', 'r') as f:
                cgroup_content = f.read()
                if 'docker' in cgroup_content or 'kubepods' in cgroup_content:
                    # We're in a container, try to create a shared path
                    shared_paths = [
                        "/tmp/shared_energy_data",
                        "/var/tmp/energy_data", 
                        "/tmp/lithops_energy_data"
                    ]
                    
                    for shared_path in shared_paths:
                        try:
                            os.makedirs(shared_path, exist_ok=True)
                            os.chmod(shared_path, 0o777)  # Make it accessible
                            
                            # Create a symlink or copy mechanism to the target
                            # Also create the JSON files in a way that can be copied out
                            print(f"✅ Using container energy data path: {shared_path}", file=sys.stderr)
                            print(f"📋 Note: Files will be created in container at {shared_path}", file=sys.stderr)
                            print(f"📋 Target path for host access: {target_path}", file=sys.stderr)
                            return shared_path
                        except Exception as e:
                            continue
        except Exception:
            pass
    
    # Fallback to a writable directory
    fallback_path = "/tmp/lithops_energy_data"
    try:
        os.makedirs(fallback_path, exist_ok=True)
        print(f"✅ Using fallback energy data path: {fallback_path}", file=sys.stderr)
        return fallback_path
    except Exception as e:
        print(f"❌ Cannot create fallback path: {e}", file=sys.stderr)
        return "/tmp"


class EnergyDataJSONLogger:
    """
    Centralized JSON logging utility for energy monitoring data.
    Provides consistent data storage and retrieval across all energy monitors.
    """
    
    def __init__(self, output_dir: str = None, fallback_dir: str = "/tmp/lithops_energy_data"):
        """
        Initialize the JSON logger.
        
        Args:
            output_dir: Primary output directory for JSON files (auto-detected if None)
            fallback_dir: Fallback directory if primary fails
        """
        if output_dir is None:
            output_dir = _get_default_energy_data_path()
        
        self.output_dir = output_dir
        self.fallback_dir = fallback_dir
        self.json_dir = None
        self._ensure_output_directory()
    
    def _ensure_output_directory(self):
        """Ensure output directory exists with proper permissions."""
        try:
            # Always use the specified output directory
            self.json_dir = self.output_dir
            
            os.makedirs(self.json_dir, exist_ok=True)
            os.chmod(self.json_dir, 0o777)  # rwx for all users
            logger.info(f"Created energy data directory: {self.json_dir}")
            print(f"✅ Energy JSON files will be stored in: {self.json_dir}", file=sys.stderr)
        except Exception as e:
            logger.error(f"Error creating energy data directory: {e}")
            print(f"❌ Error creating {self.json_dir}, using fallback directory")
            # Fallback to /tmp directory which should be writable
            self.json_dir = self.fallback_dir
            os.makedirs(self.json_dir, exist_ok=True)
            logger.info(f"Using fallback energy data directory: {self.json_dir}")
            print(f"✅ Using fallback energy data directory: {self.json_dir}")
    
    def store_energy_data_json(self, energy_data: Dict[str, Any], task: Any, cpu_info: Dict[str, Any], 
                              pkg_energy: float, cores_energy: float, core_percentage: float,
                              function_name: Optional[str] = None, monitor_specific_data: Optional[Dict[str, Any]] = None):
        """
        Store energy data in JSON format with comprehensive metadata.
        
        Args:
            energy_data: Energy measurement data from monitor
            task: Task object containing job_key and call_id
            cpu_info: CPU usage information
            pkg_energy: Package energy in Joules
            cores_energy: Core energy in Joules
            core_percentage: Core energy percentage (cores/pkg)
            function_name: Optional function name
            monitor_specific_data: Optional monitor-specific additional data
        """
        timestamp = time.time()
        
        try:
            # Create a unique ID for this execution
            execution_id = f"{task.job_key}_{task.call_id}"
            
            # Calculate additional metrics
            duration = energy_data['duration']
            energy_efficiency = pkg_energy / max(duration, 0.001)  # Watts
            
            # Calculate average CPU usage
            avg_cpu_usage = sum(cpu_info['usage']) / len(cpu_info['usage']) if cpu_info['usage'] else 0
            
            # Calculate energy per CPU usage
            energy_per_cpu = pkg_energy / max(avg_cpu_usage, 0.01)  # Joules per % CPU
            
            # Main energy consumption data
            energy_consumption = {
                'job_key': task.job_key,
                'call_id': task.call_id,
                'timestamp': timestamp,
                'energy_pkg': pkg_energy,
                'energy_cores': cores_energy,
                'core_percentage': core_percentage,
                'duration': duration,
                'source': energy_data.get('source', 'unknown'),
                'function_name': function_name,
                # Additional metrics
                'energy_efficiency': energy_efficiency,  # Watts
                'avg_cpu_usage': avg_cpu_usage,  # %
                'energy_per_cpu': energy_per_cpu,  # Joules per % CPU
                'cpu_count': len(cpu_info['usage']),  # Number of CPU cores
                'active_cpus': sum(1 for cpu in cpu_info['usage'] if cpu > 5),  # Number of active CPU cores (>5%)
                'max_cpu_usage': max(cpu_info['usage']) if cpu_info['usage'] else 0,  # Maximum CPU usage
                'system_time': cpu_info.get('system', 0),  # System CPU time
                'user_time': cpu_info.get('user', 0)  # User CPU time
            }
            
            # Add monitor-specific data if provided
            if monitor_specific_data:
                energy_consumption.update(monitor_specific_data)
            
            # CPU usage data
            cpu_usage = self._create_cpu_usage_data(cpu_info, timestamp)
            
            # Combine all data into one object
            all_data = {
                'energy_consumption': energy_consumption,
                'cpu_usage': cpu_usage
            }
            
            # Write to a single JSON file
            json_file = os.path.join(self.json_dir, f"{execution_id}.json")
            with open(json_file, 'w') as f:
                json.dump(all_data, f, indent=2)
            
            logger.info(f"Energy data stored in JSON file: {json_file}")
            print(f"📄 JSON file created: {json_file}")
            
            # Also try to create a copy in the target directory if we're in a container
            target_path = "/home/users/iarriazu/lithops_fork/inigo_test/energy_data"
            if self.json_dir != target_path:
                try:
                    os.makedirs(target_path, exist_ok=True)
                    target_file = os.path.join(target_path, f"{execution_id}.json")
                    with open(target_file, 'w') as f:
                        json.dump(all_data, f, indent=2)
                    print(f"📄 JSON file also created in target: {target_file}")
                except Exception as e:
                    print(f"❌ Could not create copy in target directory: {e}")
            
            # Update summary file
            self._update_summary_file(execution_id, function_name, timestamp, pkg_energy, 
                                    cores_energy, core_percentage, energy_efficiency, 
                                    avg_cpu_usage, energy_per_cpu, energy_data.get('source', 'unknown'))
            
        except Exception as e:
            logger.error(f"Error writing energy data to JSON file: {e}")
            print(f"❌ Error writing JSON file: {e}")
            # Fallback to simple text file
            self._write_fallback_file(task, pkg_energy, cores_energy, core_percentage, energy_data)
    
    def _create_cpu_usage_data(self, cpu_info: Dict[str, Any], timestamp: float) -> List[Dict[str, Any]]:
        """Create CPU usage data structure."""
        cpu_usage = []
        
        # Get start timestamp and end timestamps from cpu_info if available
        start_timestamp = cpu_info.get('start_timestamp', timestamp)
        end_timestamps = cpu_info.get('end_timestamps', [])
        
        # If end_timestamps is not available or empty, use the current timestamp for all cores
        if not end_timestamps:
            end_timestamps = [timestamp] * len(cpu_info['usage'])
        
        # Create CPU usage entries with both start and end timestamps
        for cpu_id, cpu_percent in enumerate(cpu_info['usage']):
            # Get the end timestamp for this CPU core
            end_timestamp = end_timestamps[cpu_id] if cpu_id < len(end_timestamps) else timestamp
            
            cpu_usage.append({
                'cpu_id': cpu_id,
                'cpu_percent': cpu_percent,
                'start_timestamp': start_timestamp,
                'end_timestamp': end_timestamp
            })
        
        return cpu_usage
    
    def _update_summary_file(self, execution_id: str, function_name: Optional[str], timestamp: float,
                           pkg_energy: float, cores_energy: float, core_percentage: float,
                           energy_efficiency: float, avg_cpu_usage: float, energy_per_cpu: float,
                           source: str):
        """Update the summary file with execution data."""
        summary_file = os.path.join(self.json_dir, 'summary.json')
        summary = []
        
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
            except Exception as e:
                logger.error(f"Error reading summary file: {e}")
                summary = []
        
        summary.append({
            'execution_id': execution_id,
            'function_name': function_name,
            'timestamp': timestamp,
            'energy_pkg': pkg_energy,
            'energy_cores': cores_energy,
            'core_percentage': core_percentage,
            'energy_efficiency': energy_efficiency,
            'avg_cpu_usage': avg_cpu_usage,
            'energy_per_cpu': energy_per_cpu,
            'source': source
        })
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Summary file updated: {summary_file}")
        
        # Also try to create summary in target directory
        target_path = "/home/users/iarriazu/lithops_fork/inigo_test/energy_data"
        if self.json_dir != target_path:
            try:
                os.makedirs(target_path, exist_ok=True)
                target_summary = os.path.join(target_path, 'summary.json')
                with open(target_summary, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"📄 Summary file also updated in target: {target_summary}")
            except Exception as e:
                print(f"❌ Could not update summary in target directory: {e}")
    
    def _write_fallback_file(self, task: Any, pkg_energy: float, cores_energy: float, 
                           core_percentage: float, energy_data: Dict[str, Any]):
        """Write fallback text file if JSON fails."""
        energy_file = os.path.join(self.fallback_dir, f'lithops_energy_consumption_{task.job_key}_{task.call_id}.txt')
        os.makedirs(os.path.dirname(energy_file), exist_ok=True)
        
        with open(energy_file, 'w') as f:
            f.write("Performance counter stats for 'system wide':\n\n")
            f.write(f"          {pkg_energy:.2f} Joules power/energy-pkg/ [{energy_data.get('source', 'unknown')}]\n")
            f.write(f"          {cores_energy:.2f} Joules power/energy-cores/ [{energy_data.get('source', 'unknown')}]\n")
            f.write(f"          {core_percentage * 100:.2f}% core percentage (cores/pkg)\n")
        
        logger.info(f"Energy data stored in fallback file: {energy_file}")
        print(f"📄 Fallback file created: {energy_file}")
    
    def update_function_name(self, task: Any, function_name: str) -> bool:
        """
        Update the function name in existing JSON files.
        
        Args:
            task: Task object containing job_key and call_id
            function_name: Function name to update
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            json_file = os.path.join(self.json_dir, f"{task.job_key}_{task.call_id}.json")
            
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Update function name
                if 'energy_consumption' in data:
                    data['energy_consumption']['function_name'] = function_name
                
                # Write updated data back to file
                with open(json_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                logger.info(f"Updated function name in JSON file: {function_name}")
                
                # Also update in target directory if different
                target_path = "/home/users/iarriazu/lithops_fork/inigo_test/energy_data"
                if self.json_dir != target_path:
                    try:
                        target_file = os.path.join(target_path, f"{task.job_key}_{task.call_id}.json")
                        if os.path.exists(target_file):
                            with open(target_file, 'w') as f:
                                json.dump(data, f, indent=2)
                            print(f"📄 Function name also updated in target: {target_file}")
                    except Exception as e:
                        print(f"❌ Could not update function name in target: {e}")
                
                # Also update the summary file
                self._update_function_name_in_summary(task, function_name)
                return True
            else:
                logger.warning(f"JSON file not found for updating function name: {json_file}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating function name in JSON file: {e}")
            return False
    
    def _update_function_name_in_summary(self, task: Any, function_name: str):
        """Update function name in summary file."""
        summary_file = os.path.join(self.json_dir, 'summary.json')
        
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                
                for entry in summary:
                    if entry.get('execution_id') == f"{task.job_key}_{task.call_id}":
                        entry['function_name'] = function_name
                
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Error updating function name in summary file: {e}")


# Global instance for shared use across all energy monitors
# Auto-detects the best path for energy data
energy_json_logger = EnergyDataJSONLogger()


def store_energy_data_json(energy_data: Dict[str, Any], task: Any, cpu_info: Dict[str, Any], 
                          pkg_energy: float, cores_energy: float, core_percentage: float,
                          function_name: Optional[str] = None, monitor_specific_data: Optional[Dict[str, Any]] = None):
    """
    Convenience function for storing energy data in JSON format.
    
    This function provides the same interface as the original _store_energy_data_json
    methods in the individual monitors, making refactoring easier.
    """
    energy_json_logger.store_energy_data_json(
        energy_data, task, cpu_info, pkg_energy, cores_energy, core_percentage,
        function_name, monitor_specific_data
    )

def update_function_name(task: Any, function_name: str) -> bool:
    """
    Convenience function for updating function name in JSON files.
    
    This function provides the same interface as the original update_function_name
    methods in the individual monitors.
    """
    return energy_json_logger.update_function_name(task, function_name)


def get_json_logger() -> EnergyDataJSONLogger:
    """Get the global JSON logger instance."""
    return energy_json_logger


def create_custom_json_logger(output_dir: str = "energy_data", 
                             fallback_dir: str = "/tmp/lithops_energy_data") -> EnergyDataJSONLogger:
    """Create a custom JSON logger with specific directories."""
    return EnergyDataJSONLogger(output_dir, fallback_dir)


def configure_energy_json_output(output_dir: str) -> None:
    """
    Configure the global energy JSON logger to use a specific output directory.
    
    Args:
        output_dir: Directory path where JSON files should be stored.
                   Can be relative (to current working directory) or absolute.
    
    Example:
        # To store in inigo_test/energy_data when running from lithops_fork:
        configure_energy_json_output("/home/users/iarriazu/lithops_fork/inigo_test/energy_data")
        
        # Or relative to current directory:
        configure_energy_json_output("inigo_test/energy_data")
    """
    global energy_json_logger
    energy_json_logger = EnergyDataJSONLogger(output_dir)
    print(f"🔧 Configured energy JSON output to: {energy_json_logger.json_dir}")


def get_current_json_output_dir() -> str:
    """Get the current JSON output directory."""
    return energy_json_logger.json_dir