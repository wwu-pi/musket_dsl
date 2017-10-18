package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.Type

class MPIRoutines {
	def static generateMPIGather(String send_buffer, int count, String type, String recv_buffer) '''
		«Config.tmp_size_t» = «count» * sizeof(«type»);
		MPI_Gather(«send_buffer», «Config.tmp_size_t», MPI_BYTE, «recv_buffer», «Config.tmp_size_t», MPI_BYTE, 0, MPI_COMM_WORLD);
	'''
}
