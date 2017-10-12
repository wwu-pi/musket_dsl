package de.wwu.musket.generator.cpu

import org.eclipse.emf.ecore.resource.Resource

import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*

class Config {
	// project paths
	public static final String base_path = "CPU/"
	public static final String include_path = "include/"
	public static final String source_path = "src/" // assumption that depth is only one folder
	
	// file extensions
	public static final String header_extension = ".hpp"
	public static final String source_extension = ".cpp"
	
	// variable names
	public static final String var_np = "number_of_processes"
	public static final String var_pid = "process_id"
	public static final String var_mpi_procs = "mpi_world_size"
	
	// project config
	public static int processes;

	def static init(Resource resource) {
		processes = resource.Processes
	}
}
