package de.wwu.musket.generator.cpu.mpmd

import de.wwu.musket.musket.MatrixType
import de.wwu.musket.musket.CollectionObject
import static extension de.wwu.musket.generator.cpu.mpmd.util.DataHelper.*
import static extension de.wwu.musket.util.TypeHelper.*
import static extension de.wwu.musket.util.MusketHelper.*
import static extension de.wwu.musket.generator.extensions.ModelElementAccess.*
import static de.wwu.musket.generator.cpu.mpmd.MPIRoutines.*
import de.wwu.musket.musket.SkeletonExpression
import org.eclipse.emf.ecore.resource.Resource
import de.wwu.musket.musket.ShiftPartitionsHorizontallySkeleton
import de.wwu.musket.musket.ShiftPartitionsVerticallySkeleton

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

	def static generateShiftHorizontallySkeletonsFunctionDeclarations() '''
		template<typename T, typename Functor>
		void shift_partitions_horizontally(mkt::DMatrix<T>& m, const Functor& f);
	'''

	def static generateShiftVerticallySkeletonsFunctionDeclarations() '''		
		template<typename T, typename Functor>
		void shift_partitions_vertically(mkt::DMatrix<T>& m, const Functor& f);
	'''
	
	def static generateShiftHorizontallyFunctionDefinitions(Resource resource) {
		var result = ''''''
		var typeFunctorPairs = newArrayList
		for(se : resource.SkeletonExpressions.filter[it.skeleton instanceof ShiftPartitionsHorizontallySkeleton]){
			val type = se.obj.calculateCollectionType.cppType
			val functor = se.getFunctorName(se.skeleton.param)
			val pair = type -> functor
			if(!typeFunctorPairs.contains(pair)){
				result += generateShiftHorizontallyFunctionDefinition(se.obj, se)
				typeFunctorPairs.add(pair)
			}			
		}
		return result
	}
	
	def static generateShiftVerticallyFunctionDefinitions(Resource resource) {
		var result = ''''''
		var typeFunctorPairs = newArrayList
		for(se : resource.SkeletonExpressions.filter[it.skeleton instanceof ShiftPartitionsVerticallySkeleton]){
			val type = se.obj.calculateCollectionType.cppType
			val functor = se.getFunctorName(se.skeleton.param)
			val pair = type -> functor
			if(!typeFunctorPairs.contains(pair)){
				result += generateShiftVerticallyFunctionDefinition(se.obj, se)
				typeFunctorPairs.add(pair)
			}			
		}
		return result
	}
		
	def static generateShiftHorizontallyFunctionDefinition(CollectionObject co, SkeletonExpression se) '''
		«val type = co.calculateCollectionType.cppType»
		«val mpitype = co.calculateCollectionType.MPIType»
		«val functorName = getFunctorName(se, se.skeleton.param)»
		template<>
		void mkt::shift_partitions_horizontally<«type», «functorName»>(mkt::DMatrix<«type»>& m, const «functorName»& f){
			int steps = f(m.get_partition_x_pos());
			
			int partitions_in_row = m.get_partitions_in_row();
			
			int target = ((((m.get_partition_y_pos() + steps) % partitions_in_row) + partitions_in_row) % partitions_in_row) + (m.get_partition_x_pos() * partitions_in_row);
			int source = ((((m.get_partition_y_pos() - steps) % partitions_in_row) + partitions_in_row) % partitions_in_row) + (m.get_partition_x_pos() * partitions_in_row);
				
			if(target != «Config.var_mpi_rank»){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto buffer = std::make_unique<std::vector<«type»>>(m.get_size_local());

				int tag_rec = ((source + «Config.var_mpi_rank») * (source + «Config.var_mpi_rank» + 1)) / 2 + «Config.var_mpi_rank»;
				int tag_send = ((«Config.var_mpi_rank» + target) * («Config.var_mpi_rank» + target + 1)) / 2 + target;
				
				MPI_Irecv(buffer->data(), m.get_size_local(), «mpitype», source, tag_rec, MPI_COMM_WORLD, &requests[1]);
				MPI_Isend(m.get_data(), m.get_size_local(), «mpitype», target, tag_send, MPI_COMM_WORLD, &requests[0]);
				«generateMPIWaitall(2, "requests", "statuses")»
				
				std::move(buffer->begin(), buffer->end(), m.begin());
			}
		}
	'''
	
	def static generateShiftVerticallyFunctionDefinition(CollectionObject co, SkeletonExpression se) '''
		«val type = co.calculateCollectionType.cppType»
		«val mpitype = co.calculateCollectionType.MPIType»
		«val functorName = getFunctorName(se, se.skeleton.param)»
		template<>
		void mkt::shift_partitions_vertically<«type», «functorName»>(mkt::DMatrix<«type»>& m, const «functorName»& f){
			int steps = f(m.get_partition_y_pos());
			
			int partitions_in_row = m.get_partitions_in_row();
			int partitions_in_column = m.get_partitions_in_column();
			
			int target = ((((m.get_partition_x_pos() + steps) % partitions_in_column) + partitions_in_column) % partitions_in_column) * partitions_in_row + m.get_partition_y_pos();
			int source = ((((m.get_partition_x_pos() - steps) % partitions_in_column) + partitions_in_column) % partitions_in_column) * partitions_in_row + m.get_partition_y_pos();
			
			
			if(target != «Config.var_mpi_rank»){
				MPI_Request requests[2];
				MPI_Status statuses[2];
				auto buffer = std::make_unique<std::vector<«type»>>(m.get_size_local());

				int tag_rec = ((source + «Config.var_mpi_rank») * (source + «Config.var_mpi_rank» + 1)) / 2 + «Config.var_mpi_rank»;
				int tag_send = ((«Config.var_mpi_rank» + target) * («Config.var_mpi_rank» + target + 1)) / 2 + target;
				
				MPI_Irecv(buffer->data(), m.get_size_local(), «mpitype», source, tag_rec, MPI_COMM_WORLD, &requests[1]);
				MPI_Isend(m.get_data(), m.get_size_local(), «mpitype», target, tag_send, MPI_COMM_WORLD, &requests[0]);
				«generateMPIWaitall(2, "requests", "statuses")»
				
				std::move(buffer->begin(), buffer->end(), m.get_data());
			}
		}
	'''
}