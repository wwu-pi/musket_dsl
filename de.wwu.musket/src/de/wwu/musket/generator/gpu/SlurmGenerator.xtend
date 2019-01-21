package de.wwu.musket.generator.gpu

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
	}

	/**
	 * Generates the content of the job file.
	 * 
	 * @param resource the resource object
	 * @return the content of the job file
	 */
	def static JobScriptContent(Resource resource) '''
		#!/bin/bash
		#SBATCH --job-name «resource.ProjectName»-nodes-«resource.ConfigBlock.processes»-cpu-«resource.ConfigBlock.cores»
		#SBATCH --ntasks «resource.ConfigBlock.processes»
		#SBATCH --nodes «resource.ConfigBlock.processes»
		#SBATCH --ntasks-per-node 1
		#SBATCH --partition normal
		#SBATCH --output «Config.out_path»«resource.ProjectName»-nodes-«resource.ConfigBlock.processes»-cpu-«resource.ConfigBlock.cores».out
		#SBATCH --cpus-per-task 64
		#SBATCH --mail-type ALL
		#SBATCH --mail-user my@e-mail.de
		#SBATCH --time 01:00:00
		
		export OMP_NUM_THREADS=«resource.ConfigBlock.cores»
		
		«IF Config.processes > 1»
			mpirun «Config.build_path»bin/«resource.ProjectName»
		«ELSE»
			«Config.build_path»bin/«resource.ProjectName»
		«ENDIF»		
	'''
}
