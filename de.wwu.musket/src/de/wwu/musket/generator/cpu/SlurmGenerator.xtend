package de.wwu.musket.generator.cpu

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
		    srun «Config.build_path»benchmark/bin/«resource.ProjectName»
		done		
	'''
}
