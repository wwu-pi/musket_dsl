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
	
	public static final String tmp_size_t = "tmp_size_t"
	
	
	public static final String var_loop_counter = "counter"
	public static final String var_loop_counter_rows = "counter_rows"
	public static final String var_loop_counter_cols = "counter_cols"
	public static final String var_fold_result = "fold_result"
	
	public static final String var_row_offset = "row_offset"
	public static final String var_col_offset = "col_offset"
	public static final String var_elem_offset = "elem_offset"
	
	public static final String mpi_op_suffix = "_mpi_op"
	
	//rotatePartitons
	public static final String var_rotate_source = "rotate_source"
	public static final String var_rotate_target = "rotate_target"
	public static final String var_rotate_steps = "rotate_steps"
	public static final String tmp_rotate_buffer = "tmp_rotate_buffer"
	
	public static final String var_rng_array = "random_engines"
	
	// project config
	public static int processes;

	def static init(Resource resource) {
		processes = resource.Processes
	}
}
