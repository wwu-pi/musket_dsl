package de.wwu.musket.generator.cpu

import org.eclipse.emf.ecore.resource.Resource

import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*
import de.wwu.musket.musket.Mode

/**
 * Global config class for the generator.
 * <p>
 * The config contains certain values such as variable names or paths.
 * Some values can also be read from the model, but it might be nice to set them once and access them via the config class.
 * The init method is used for that purpose.
 * 
 * TODO: might be good to also include types for the variables.
 */
class Config {
	// project paths
	public static String base_path = ""
	public static final String include_path = "include/"
	public static final String source_path = "src/" // assumption that depth is only one folder
	// file extensions
	public static final String header_extension = ".hpp"
	public static final String source_extension = ".cpp"

	// variable names
	public static final String var_np = "number_of_processes"
	public static final String var_pid = "process_id"
	public static final String var_mpi_procs = "mpi_world_size"
	public static final String var_omp_tid = "omp_tid"

	public static final String tmp_size_t = "tmp_size_t"

	public static final String var_loop_counter = "counter"
	public static final String var_loop_counter_rows = "counter_rows"
	public static final String var_loop_counter_cols = "counter_cols"
	public static final String var_fold_result = "fold_result"

	public static final String var_row_offset = "row_offset"
	public static final String var_col_offset = "col_offset"
	public static final String var_elem_offset = "elem_offset"

	public static final String mpi_op_suffix = "_mpi_op"

	// shiftPartitons
	public static final String var_shift_source = "shift_source"
	public static final String var_shift_target = "shift_target"
	public static final String var_shift_steps = "shift_steps"
	public static final String tmp_shift_buffer = "tmp_shift_buffer"

	public static final String var_rng_array = "random_engines"

	// project config
	public static int processes;
	public static int cores;
	public static Mode mode;

	/**
	 * This method is called once in the beginning of the CPU generator and initializes some values.
	 * 
	 * @param resource the resource object
	 */
	def static init(Resource resource) {
		processes = resource.Processes
		cores = resource.ConfigBlock.cores
		base_path = resource.ProjectName + "/CPU/"
		mode = resource.ConfigBlock.mode
	}
}
