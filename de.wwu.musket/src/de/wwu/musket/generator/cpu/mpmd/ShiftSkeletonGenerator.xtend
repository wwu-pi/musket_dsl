package de.wwu.musket.generator.cpu.mpmd

import de.wwu.musket.musket.MatrixType
import de.wwu.musket.musket.CollectionObject
import static extension de.wwu.musket.generator.cpu.mpmd.util.DataHelper.*
import static extension de.wwu.musket.util.TypeHelper.*
import static extension de.wwu.musket.util.MusketHelper.*

class ShiftSkeletonGenerator {
	def static generateShiftSkeletonVariables(int processId) '''
		int «Config.var_shift_source» = «processId»;
		int «Config.var_shift_target» = «processId»;
		int «Config.var_shift_steps» = 0;
	'''
			
	def static generateMPIVectorType(MatrixType m, int processId) '''
		«val type_name = (m.eContainer as CollectionObject).name + "_partition_type"»
		MPI_Datatype «type_name»;
		MPI_Type_vector(«m.rowsLocal», «m.colsLocal», «m.cols.concreteValue», «m.calculateCollectionType.MPIType», &«type_name»);
		MPI_Type_create_resized(«type_name», 0, sizeof(«m.calculateCollectionType.cppType») * «m.colsLocal», &«type_name»_resized);
		MPI_Type_free(&«type_name»);
		MPI_Type_commit(&«type_name»_resized);
	'''

	def static generateMPIVectorTypeVariable(MatrixType m) '''
		«val type_name = (m.eContainer as CollectionObject).name + "_partition_type"»
		MPI_Datatype «type_name»_resized;
	'''

}