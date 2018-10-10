package de.wwu.musket.generator.cpu.mpmd

import de.wwu.musket.musket.ArrayType
import de.wwu.musket.musket.CollectionFunctionCall
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.DistributionMode
import de.wwu.musket.musket.MatrixType
import de.wwu.musket.musket.Struct
import de.wwu.musket.musket.StructArrayType
import de.wwu.musket.musket.StructMatrixType

import static de.wwu.musket.generator.cpu.mpmd.MPIRoutines.*
import static extension de.wwu.musket.util.MusketHelper.*
import static extension de.wwu.musket.util.TypeHelper.*

import static extension de.wwu.musket.generator.cpu.mpmd.util.DataHelper.*

/**
 * Generator for collection functions.
 * <p>
 * Entry point is the method generateCollectionFunctionCall(CollectionFunctionCall)
 * Collection functions are functions provided by the distributed array and matrix, such as show() or size().
 * For some functions the result can already be calculated by the generator (e.g. size()), while other functions
 * have to do more work, such as show(), which for example needs to generate a gather statement for distributed types.
 */
class CollectionFunctionsGenerator {

	/**
	 * Entry point for the Collection Function call generator.
	 * 
	 * @param cfc the collection function call
	 * @return generated code
	 */
	def static generateCollectionFunctionCall(CollectionFunctionCall cfc, int processId) {
		switch cfc.function {
			case SIZE:
				generateSize(cfc, processId)
			case SIZE_LOCAL:
				generateSizeLocal(cfc, processId)
			case SHOW: {
				switch (cfc.^var.type) {
					ArrayType: generateShowArray(cfc.^var, processId)
					MatrixType: generateShowMatrix(cfc.^var, processId)
				}
			}
			
			case SHOW_LOCAL: {
				switch (cfc.^var.type) {
					ArrayType: generateShowLocalArray(cfc.^var, processId)
					MatrixType: generateShowLocalMatrix(cfc.^var, processId)
				}
			}
			
			case COLUMNS:
				(cfc.^var.type as MatrixType).cols
			case COLUMNS_LOCAL:
				(cfc.^var.type as MatrixType).colsLocal
			case ROWS:
				(cfc.^var.type as MatrixType).rows
			case ROWS_LOCAL:
				(cfc.^var.type as MatrixType).rowsLocal
			case BLOCKS_IN_ROW:
				(cfc.^var.type as MatrixType).blocksInRow
			case BLOCKS_IN_COLUMN:
				(cfc.^var.type as MatrixType).blocksInColumn
		}
	}

	/**
	 * Generates the size() method.
	 * 
	 * @param cfc the collection function call
	 * @return generated code
	 */
	def static generateSize(CollectionFunctionCall cfc, int processId) '''«cfc.^var.type.size»'''

	/**
	 * Generates the localSize() method.
	 * 
	 * @param cfc the collection function call
	 * @return generated code
	 */
	def static generateSizeLocal(CollectionFunctionCall cfc, int processId) '''«cfc.^var.type.sizeLocal(processId)»'''

	/**
	 * Generates the show() method for arrays.
	 * 
	 * @param a the array
	 * @return generated code
	 */
	def static generateShowArray(CollectionObject a, int processId) '''
		«IF Config.processes == 1»
			mkt::print("«a.name»", «a.name»);
		«ELSEIF a.type.distributionMode == DistributionMode.COPY»
			«IF processId == 0»
				mkt::print("«a.name»", «a.name»);
			«ELSE»
				// show array (copy) --> only in p0
		  	«ENDIF»
		«ELSEIF a.type.distributionMode == DistributionMode.DIST»
			«IF processId == 0»
				mkt::print_dist("«a.name»", «a.name»);
		  	«ELSE»
				«generateMPIGather(a.name + ".get_data()", (a.type as ArrayType).sizeLocal(processId), a.calculateType, "nullptr")»
		  	«ENDIF»
		«ELSE»
		  // Collection Functions: generateShowArray default case --> something went wrong
		«ENDIF»
	'''

	/**
	 * Generates the show() method for matrices.
	 *  
	 * @param m the matrix
	 * @return generated code
	 */
	def static generateShowMatrix(CollectionObject m, int processId) '''
		«IF Config.processes == 1»
			mkt::print("«m.name»", «m.name»);
		«ELSEIF m.type.distributionMode == DistributionMode.COPY»
			«IF processId == 0»
				mkt::print("«m.name»", «m.name»);
			«ELSE»
				// show matrix (copy) --> only in p0
		  	«ENDIF»
		«ELSEIF m.type.distributionMode == DistributionMode.DIST»
			«IF processId == 0»
				mkt::print_dist_«m.name»(«m.name», «m.name»_partition_type_resized);
		  	«ELSE»
				// show matrix dist
				«generateMPIGathervNonRoot(m.name + ".get_data()", (m.type as MatrixType).sizeLocal(processId), m.calculateCollectionType)»
		  	«ENDIF»
		«ELSE»
		  // Collection Functions: generateShowArray default case --> something went wrong
		«ENDIF»
	'''

	/**
	 * Generates the showLocal() method for arrays.
	 * 
	 * @param a the array
	 * @return generated code
	 */
	def static generateShowLocalArray(CollectionObject a, int processId) '''
		«val streamName = 's' + State.counter»
		«State.incCounter»
		std::ostringstream «streamName»;
		«streamName» << "«a.name» on «processId»: " << std::endl << "[";
		for (int i = 0; i < «(a.type as ArrayType).sizeLocal(processId) - 1»; i++) {
		«streamName» << «generateShowElements(a, a.name, "i")»;
		«streamName» << "; ";
		}
		«streamName» << «generateShowElements(a, a.name, ((a.type as ArrayType).sizeLocal(processId) - 1).toString)» << "]" << std::endl;
		«streamName» << std::endl;
		printf("%s", «streamName».str().c_str());
	'''

	/**
	 * Generates the showLocal() method for matrices.
	 * 
	 * @param a the array
	 * @return generated code
	 */
	def static generateShowLocalMatrix(CollectionObject m, int processId) '''
		«val type = m.type as MatrixType»
		«val streamName = 's' + State.counter»
		«State.incCounter»
		std::ostringstream «streamName»;
		«streamName» << "«m.name» on «processId»: " << std::endl << "[";
		
		for(int «Config.var_loop_counter_rows» = 0; «Config.var_loop_counter_rows» < «type.rowsLocal»; ++«Config.var_loop_counter_rows»){
			«streamName» << "[";
			for(int «Config.var_loop_counter_cols» = 0; «Config.var_loop_counter_cols» < «type.colsLocal»; ++«Config.var_loop_counter_cols»){
				«streamName» << «generateShowElements(m, m.name, Config.var_loop_counter_rows + " * " + type.rowsLocal + " + " + Config.var_loop_counter_cols)»;
				if(«Config.var_loop_counter_cols» < «type.colsLocal - 1»){
					«streamName» << "; ";
				}else{
					«streamName» << "]" << std::endl;
				}
			}
		}
		printf("%s", «streamName».str().c_str());
	'''

	// helper
	/**
	 * Helper used by the methods generateShowArray(CollectionObject) and generateShowMatrix(CollectionObject).
	 * If the collection contains primitive values, the value can just be accessed.
	 * If the collection contains structs, all the elements within the struct have to be shown. 
	 * 
	 * @param co the collection object
	 * @param arrayName the name of the array in the generated cpp code (might be different than name of co if the values are gathered before).
	 * @param index the index in the cpp code
	 * @return generated code
	 */
	def static String generateShowElements(CollectionObject co, String arrayName, String index) {
		switch co.type {
			StructArrayType: (co.type as StructArrayType).type.generateShowStruct(arrayName, index)
			StructMatrixType: (co.type as StructMatrixType).type.generateShowStruct(arrayName, index)
			default: '''«arrayName»[«index»]'''
		}
	}

	/**
	 * Helper used by the method generateShowElements(CollectionObject, String, String).
	 * It generates the output for a struct, i.e. primitive values or if there is another nested struct,
	 * method calls itself again.
	 * TODO: this should not work if the struct contains another array or matrix
	 * 
	 * @param s the struct
	 * @param arrayName the name of the array in the generated cpp code (might be different than name of co if the values are gathered before).
	 * @param index the index in the cpp code
	 * @return generated code
	 */
	def static String generateShowStruct(Struct s, String arrayName, String index) {
		var result = '"["'

		for (i : 0 ..< s.attributes.size) {
			val ca = s.attributes.get(i)
			var separator = ' << ", "'

			if (i == s.attributes.size - 1) {
				separator = ''
			}

			switch ca {
				Struct:
					result += ca.generateShowStruct(arrayName, index) + separator
				default:
					result +=
						' << "' + s.attributes.get(i).name + ' = " << ' + arrayName + '[' + index + '].' +
							s.attributes.get(i).name + separator
			}
		}

		return result + ' << "]"'
	}

	/**
	 * Internal state to keep track of temp counter and current array name. 
	 * This can be nice to avoid setting variables in templates, since that produces an output.
	 * The setter here return void so there is none.
	 */
	static class State {
		static String arrayName;
		static int counter = 0;

		def static void setArrayName(String arrayName) {
			State.arrayName = arrayName;
		}

		def static void incCounter() {
			counter = counter + 1;
		}
	}
}
