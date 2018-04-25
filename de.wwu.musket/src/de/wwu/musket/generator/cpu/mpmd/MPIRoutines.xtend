package de.wwu.musket.generator.cpu.mpmd

import de.wwu.musket.util.MusketType
import static extension de.wwu.musket.util.TypeHelper.*
import de.wwu.musket.musket.Struct

/**
 * Generates MPI routines.
 * For all communication the type MPI_BYTE is used. Therefore, the size is calculated before the MPI call.
 * The variable counter is used to name temporary variables.
 * 
 */
class MPIRoutines {
	var static counter = 0

	def static generateMPIGather(String send_buffer, long count, MusketType type, String recv_buffer) '''
		MPI_Gather(«send_buffer», «count», «type.MPIType», «recv_buffer», «count», «type.MPIType», 0, MPI_COMM_WORLD);
	'''

	def static generateMPIAllgather(String send_buffer, long count, MusketType type, String recv_buffer) '''
		MPI_Allgather(«send_buffer», «count», «type.MPIType», «recv_buffer», «count», «type.MPIType», MPI_COMM_WORLD);
	'''

	def static generateMPIIsend(int source, String send_buffer, long count, MusketType type, int target,
		String request) '''
		«val tag = ((source + target) * (source + target + 1)) / 2 + target»
		MPI_Isend(«send_buffer», «count», «type.MPIType», «target», «tag», MPI_COMM_WORLD, «request»);
	'''

	def static generateMPIIrecv(int target, String recv_buffer, long count, MusketType type, int source,
		String request) '''
		«val tag = ((source + target) * (source + target + 1)) / 2 + target»
		MPI_Irecv(«recv_buffer», «count», «type.MPIType», «source», «tag», MPI_COMM_WORLD, «request»);
	'''

	def static generateMPIIsend(int source, String send_buffer, long count, MusketType type, String target,
		String request) '''
		int tag_«counter» = ((«source» + «target») * («source» + «target» + 1)) / 2 + «target»;
		MPI_Isend(«send_buffer», «count», «type.MPIType», «target», tag_«counter++», MPI_COMM_WORLD, «request»);
	'''

	def static generateMPIIrecv(int target, String recv_buffer, long count, MusketType type, String source,
		String request) '''
		«Config.tmp_size_t» = «count» * sizeof(«type»);
		int tag_«counter» = ((«source» + «target») * («source» + «target» + 1)) / 2 + «target»;
		MPI_Irecv(«recv_buffer», «Config.tmp_size_t», «type.MPIType», «source», tag_«counter++», MPI_COMM_WORLD, «request»);
	'''

	def static generateMPIWaitall(int count, String requests, String statuses) '''
		MPI_Waitall(«count», «requests», «statuses»);
	'''

	def static generateCreateDatatypeStruct(Struct s) '''
		MPI_Datatype «s.name»_mpi_type_temp, «s.name»_mpi_type;
		MPI_Type_create_struct(«s.attributes.size», (std::array<int,«s.attributes.size»>{«FOR i : 0 ..< s.attributes.size SEPARATOR ", "»1«ENDFOR»}).data(), (std::array<MPI_Aint,«s.attributes.size»>{«FOR i : 0 ..< s.attributes.size SEPARATOR ", "»static_cast<MPI_Aint>(offsetof(struct «s.name», «s.attributes.get(i).name»))«ENDFOR»}).data(), (std::array<MPI_Datatype,«s.attributes.size»>{«FOR i : 0 ..< s.attributes.size SEPARATOR ", "»«s.attributes.get(i).calculateType.MPIType»«ENDFOR»}).data(), &«s.name»_mpi_type_temp);
		MPI_Type_create_resized(«s.name»_mpi_type_temp, 0, sizeof(«s.name»), &«s.name»_mpi_type);
		MPI_Type_free(&«s.name»_mpi_type_temp);
	'''
}
