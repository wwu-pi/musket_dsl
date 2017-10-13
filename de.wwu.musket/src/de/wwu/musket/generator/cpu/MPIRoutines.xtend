package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.Type

class MPIRoutines {
	def static generateMPIGather(String send_buffer, int count, String type, String recv_buffer) '''
		size_t bytes«Status.temp_count» = «count» * sizeof(«type»);
		MPI_Gather(«send_buffer», bytes«Status.temp_count», MPI_BYTE, «recv_buffer», bytes«Status.temp_count++», MPI_BYTE, 0, MPI_COMM_WORLD);
	'''
}
