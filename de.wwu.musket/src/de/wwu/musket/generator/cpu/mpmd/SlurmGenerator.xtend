package de.wwu.musket.generator.cpu.mpmd

import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.eclipse.emf.ecore.resource.Resource
import org.eclipse.xtext.generator.IFileSystemAccess2
import org.eclipse.xtext.generator.IGeneratorContext

import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*

/**
 * Generates slurm job files, which are needed for clusters using Slurm, such as Palma and Taurus.
 */
class SlurmGenerator {
	private static final Logger logger = LogManager.getLogger(RunScriptGenerator)

	/**
	 * Generates the job file with the name job.sh.
	 */
	def static void generateSlurmJob(Resource resource, IFileSystemAccess2 fsa, IGeneratorContext context) {
		logger.info("Generate job.sh.")
		fsa.generateFile(Config.base_path + "job.sh", JobScriptContent(resource))
		logger.info("Generation of job.sh done.")
		
		logger.info("Generate job.conf.")
		fsa.generateFile(Config.base_path + "job.conf", JobConfContent(resource))
		logger.info("Generation of job.conf done.")
		
		logger.info("Generate callgrind job files.")
		fsa.generateFile(Config.base_path + "job-callgrind.sh", JobCallgrindScriptContent(resource))
		fsa.generateFile(Config.base_path + "job-callgrind.conf", JobCallgrindConfContent(resource))
		logger.info("Generation callgrind job files done.")
	}

	/**
	 * Generates the content of the job file.
	 * 
	 * @param resource the resource object
	 * @return the content of the job file
	 */
	def static JobScriptContent(Resource resource) '''
		#!/bin/bash
		#SBATCH --job-name «resource.ProjectName»-CPU-MPMD-nodes-«resource.ConfigBlock.processes»-cpu-«resource.ConfigBlock.cores»
		#SBATCH --ntasks «resource.ConfigBlock.processes»
		#SBATCH --nodes «resource.ConfigBlock.processes»
		#SBATCH --ntasks-per-node 1
		#SBATCH --partition haswell
		#SBATCH --exclude taurusi[1001-1270],taurusi[3001-3180],taurusi[2001-2108],taurussmp[1-7],taurusknl[1-32]
		#SBATCH --output «Config.out_path»«resource.ProjectName»-nodes-«resource.ConfigBlock.processes»-cpu-«resource.ConfigBlock.cores».out
		#SBATCH --cpus-per-task 24
		#SBATCH --mail-type ALL
		#SBATCH --mail-user fabian.wrede@mailbox.tu-dresden.de
		#SBATCH --time 05:00:00
		#SBATCH -A p_algcpugpu
		
		export OMP_NUM_THREADS=«resource.ConfigBlock.cores»
		
		RUNS=10
		for ((i=1;i<=RUNS;i++)); do
		    srun --multi-prog «Config.home_path_source»«Config.base_path»job.conf
		done	
	'''
	
	/**
	 * Generates the content of the job config file.
	 * 
	 * @param resource the resource object
	 * @return the content of the job file
	 */
	def static JobConfContent(Resource resource) '''
		«FOR p : 0 ..< Config.processes»
			«p» «Config.build_path»benchmark/bin/«resource.ProjectName»_«p»
		«ENDFOR»
	'''

	/**
	 * Generates the content of the callgrind job file.
	 * 
	 * @param resource the resource object
	 * @return the content of the job file
	 */
	def static JobCallgrindScriptContent(Resource resource) '''
		#!/bin/bash
		#SBATCH --job-name «resource.ProjectName»-CPU-MPMD-callgrind-nodes-«resource.ConfigBlock.processes»-cpu-«resource.ConfigBlock.cores»
		#SBATCH --ntasks «resource.ConfigBlock.processes»
		#SBATCH --nodes «resource.ConfigBlock.processes»
		#SBATCH --ntasks-per-node 1
		#SBATCH --partition haswell
		#SBATCH --exclude taurusi[1001-1270],taurusi[3001-3180],taurusi[2001-2108],taurussmp[1-7],taurusknl[1-32]
		#SBATCH --output «Config.out_path»«resource.ProjectName»-nodes-«resource.ConfigBlock.processes»-cpu-«resource.ConfigBlock.cores».out
		#SBATCH --cpus-per-task 24
		#SBATCH --mail-type ALL
		#SBATCH --mail-user fabian.wrede@mailbox.tu-dresden.de
		#SBATCH --time 05:00:00
		#SBATCH -A p_algcpugpu
		
		export OMP_NUM_THREADS=«resource.ConfigBlock.cores»

		srun --multi-prog «Config.home_path_source»«Config.base_path»job-callgrind.conf
	'''
	
	/**
	 * Generates the content of the callgrind job config file.
	 * 
	 * @param resource the resource object
	 * @return the content of the job file
	 */
	def static JobCallgrindConfContent(Resource resource) '''
		«FOR p : 0 ..< Config.processes»
			«p» valgrind --tool=callgrind --cache-sim=yes --cacheuse=yes --callgrind-out-file=«Config.out_path»«resource.ProjectName»-nodes-«resource.ConfigBlock.processes»-cpu-«resource.ConfigBlock.cores».out.%p «Config.build_path»cg/bin/«resource.ProjectName»_«p»
		«ENDFOR»
	'''

}
