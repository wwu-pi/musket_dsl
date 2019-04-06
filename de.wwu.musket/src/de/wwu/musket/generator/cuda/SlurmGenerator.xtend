package de.wwu.musket.generator.cuda

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
		
		logger.info("Generate nvprof-job.sh.")
		fsa.generateFile(Config.base_path + "nvprof-job.sh", NvprofJobScriptContent(resource))
		logger.info("Generation of nvprof-job.sh done.")
		
		logger.info("Generate nvprof-job.conf.")
		fsa.generateFile(Config.base_path + "nvprof-job.conf", NvprofJobConfContent(resource))
		logger.info("Generation of nvprof-job.conf done.")
	}

	def static JobScriptContent(Resource resource) '''
		#!/bin/bash
		#SBATCH --job-name «resource.ProjectName»-GPU-nodes-«resource.ConfigBlock.processes»-gpu-«resource.ConfigBlock.gpus»
		#SBATCH --ntasks «resource.ConfigBlock.processes»
		#SBATCH --nodes «resource.ConfigBlock.processes»
		#SBATCH --ntasks-per-node 1
		#SBATCH --partition gpu2
		#SBATCH --exclusive
		#SBATCH --output «Config.out_path»«resource.ProjectName»-nodes-«resource.ConfigBlock.processes»-gpu-«resource.ConfigBlock.gpus».out
		#SBATCH --cpus-per-task 24
		#SBATCH --mail-type ALL
		#SBATCH --mail-user fabian.wrede@mailbox.tu-dresden.de
		#SBATCH --time 00:05:00
		#SBATCH -A p_algcpugpu
		#SBATCH --gres gpu:4
		
		export OMP_NUM_THREADS=24
		
		RUNS=1
		for ((i=1;i<=RUNS;i++)); do
		    srun --multi-prog «Config.home_path_source»«Config.base_path»job.conf
		done
	'''
	
	def static JobConfContent(Resource resource) '''
		«FOR p : 0 ..< Config.processes»
			«p» «Config.build_path»benchmark/bin/«resource.ProjectName»_«p»
		«ENDFOR»
	'''
	
	def static NvprofJobScriptContent(Resource resource) '''
		#!/bin/bash
		#SBATCH --job-name «resource.ProjectName»-GPU-nvprof-nodes-«resource.ConfigBlock.processes»-gpu-«resource.ConfigBlock.gpus»
		#SBATCH --ntasks «resource.ConfigBlock.processes»
		#SBATCH --nodes «resource.ConfigBlock.processes»
		#SBATCH --ntasks-per-node 1
		#SBATCH --partition gpu2
		#SBATCH --exclusive
		#SBATCH --output «Config.out_path»«resource.ProjectName»-nvprof-nodes-«resource.ConfigBlock.processes»-gpu-«resource.ConfigBlock.gpus».out
		#SBATCH --cpus-per-task 24
		#SBATCH --mail-type ALL
		#SBATCH --mail-user fabian.wrede@mailbox.tu-dresden.de
		#SBATCH --time 01:00:00
		#SBATCH -A p_algcpugpu
		#SBATCH --gres gpu:4
		
		export OMP_NUM_THREADS=24
		
		RUNS=1
		for ((i=1;i<=RUNS;i++)); do
		    srun --multi-prog «Config.home_path_source»«Config.base_path»nvprof-job.conf
		done
	'''
	
	def static NvprofJobConfContent(Resource resource) '''
		«FOR p : 0 ..< Config.processes»
			«p» nvprof -f -o «Config.out_path»«resource.ProjectName»-nodes-«resource.ConfigBlock.processes»-gpu-«resource.ConfigBlock.gpus»-nvprof-%p.out --annotate-mpi openmpi --cpu-thread-tracing on --devices all «Config.build_path»nvprof/bin/«resource.ProjectName»_«p»
		«ENDFOR»
	'''
}
