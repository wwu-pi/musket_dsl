package de.wwu.musket.generator.cpu

/**
 * Generates MPI routines.
 * For all communication the type MPI_BYTE is used. Therefore, the size is calculated before the MPI call.
 * The variable counter is used to name temporary variables.
 * 
 */
class MPIRoutines {
	var static counter = 0

	def static generateMPIGather(String send_buffer, int count, String type, String recv_buffer) '''
		«Config.tmp_size_t» = «count» * sizeof(«type»);
		MPI_Gather(«send_buffer», «Config.tmp_size_t», MPI_BYTE, «recv_buffer», «Config.tmp_size_t», MPI_BYTE, 0, MPI_COMM_WORLD);
	'''

	def static generateMPIAllgather(String send_buffer, int count, String type, String recv_buffer) '''
		«Config.tmp_size_t» = «count» * sizeof(«type»);
		MPI_Allgather(«send_buffer», «Config.tmp_size_t», MPI_BYTE, «recv_buffer», «Config.tmp_size_t», MPI_BYTE, MPI_COMM_WORLD);
	'''

	def static generateMPIIsend(int source, String send_buffer, int count, String type, int target, String request) '''
		«Config.tmp_size_t» = «count» * sizeof(«type»);
		«val tag = ((source + target) * (source + target + 1)) / 2 + target»
		MPI_Isend(«send_buffer», «Config.tmp_size_t», MPI_BYTE, «target», «tag», MPI_COMM_WORLD, «request»);
	'''

	def static generateMPIIrecv(int target, String recv_buffer, int count, String type, int source, String request) '''
		«Config.tmp_size_t» = «count» * sizeof(«type»);
		«val tag = ((source + target) * (source + target + 1)) / 2 + target»
		MPI_Irecv(«recv_buffer», «Config.tmp_size_t», MPI_BYTE, «source», «tag», MPI_COMM_WORLD, «request»);
	'''

	def static generateMPIIsend(int source, String send_buffer, int count, String type, String target,
		String request) '''
		«Config.tmp_size_t» = «count» * sizeof(«type»);
		int tag_«counter» = ((«source» + «target») * («source» + «target» + 1)) / 2 + «target»;
		MPI_Isend(«send_buffer», «Config.tmp_size_t», MPI_BYTE, «target», tag_«counter++», MPI_COMM_WORLD, «request»);
	'''

	def static generateMPIIrecv(int target, String recv_buffer, int count, String type, String source,
		String request) '''
		«Config.tmp_size_t» = «count» * sizeof(«type»);
		int tag_«counter» = ((«source» + «target») * («source» + «target» + 1)) / 2 + «target»;
		MPI_Irecv(«recv_buffer», «Config.tmp_size_t», MPI_BYTE, «source», tag_«counter++», MPI_COMM_WORLD, «request»);
	'''

	def static generateMPIWaitall(int count, String requests, String statuses) '''
		MPI_Waitall(«count», «requests», «statuses»);
	'''
}
