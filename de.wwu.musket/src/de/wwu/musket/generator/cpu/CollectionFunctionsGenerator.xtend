package de.wwu.musket.generator.cpu

import de.wwu.musket.musket.ArrayType
import de.wwu.musket.musket.CollectionFunctionCall
import de.wwu.musket.musket.CollectionObject
import de.wwu.musket.musket.DistributionMode
import de.wwu.musket.musket.MatrixType
import de.wwu.musket.musket.Struct
import de.wwu.musket.musket.StructArrayType
import de.wwu.musket.musket.StructMatrixType

import static de.wwu.musket.generator.cpu.MPIRoutines.*
import static extension de.wwu.musket.util.MusketHelper.*
import static extension de.wwu.musket.generator.extensions.ObjectExtension.*
import static extension de.wwu.musket.util.TypeHelper.*

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
	def static generateCollectionFunctionCall(CollectionFunctionCall cfc) {
		switch cfc.function {
			case SIZE:
				generateSize(cfc)
			case SIZE_LOCAL:
				generateSizeLocal(cfc)
			case SHOW: {
				switch (cfc.^var.type) {
					ArrayType: generateShowArray(cfc.^var)
					MatrixType: generateShowMatrix(cfc.^var)
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
	def static generateSize(CollectionFunctionCall cfc) '''«cfc.^var.type.size»'''

	/**
	 * Generates the localSize() method.
	 * 
	 * @param cfc the collection function call
	 * @return generated code
	 */
	def static generateSizeLocal(CollectionFunctionCall cfc) '''«cfc.^var.type.sizeLocal»'''

	/**
	 * Generates the show() method for arrays.
	 * 
	 * @param a the array
	 * @return generated code
	 */
	def static generateShowArray(CollectionObject a) '''
		«IF a.type.distributionMode == DistributionMode.COPY || Config.processes == 1»
			«State.setArrayName(a.name)»
		«ELSE»
			«State.setArrayName("temp" + State.counter)»			
			std::array<«a.calculateCollectionType.cppType», «a.type.size»> «State.arrayName»{};
			
			«generateMPIGather(a.name + '.data()', a.type.sizeLocal, a.calculateCollectionType.cppType, State.arrayName + '.data()')»
		«ENDIF»
		
		«IF Config.processes > 1»
			if («Config.var_pid» == 0) {
		«ENDIF»
			«val streamName = 's' + State.counter»
			«State.incCounter»
			std::ostringstream «streamName»;
			«streamName» << "«a.name»: " << std::endl << "[";
			for (int i = 0; i < «(a.type as ArrayType).size.concreteValue - 1»; i++) {
			«streamName» << «generateShowElements(a, State.arrayName, "i")»;				
			«streamName» << "; ";
			}
			«streamName» << «generateShowElements(a, State.arrayName, ((a.type as ArrayType).size.concreteValue - 1).toString)» << "]" << std::endl;
			«streamName» << std::endl;
			printf("%s", «streamName».str().c_str());
		«IF Config.processes > 1»
			}
		«ENDIF»
	'''

	/**
	 * Generates the show() method for matrices.
	 *  
	 * @param m the matrix
	 * @return generated code
	 */
	def static generateShowMatrix(CollectionObject m) '''
		«val type = m.type as MatrixType»
		«IF m.type.distributionMode == DistributionMode.COPY || Config.processes == 1»
			«State.setArrayName(m.name)»
		«ELSE»
			«State.setArrayName("temp" + State.counter)»
			std::array<«m.calculateCollectionType.cppType», «m.type.size»> «State.arrayName»{};
			
			«generateMPIGather(m.name + '.data()', m.type.sizeLocal, m.calculateCollectionType.cppType, State.arrayName + '.data()')»
		«ENDIF»
		
		«val streamName = 's' + State.counter»
		«State.incCounter»
		«IF Config.processes > 1»
			if («Config.var_pid» == 0) {
		«ENDIF»
			std::ostringstream «streamName»;
			«streamName» << "«m.name»: " << std::endl << "[" << std::endl;
			
			for(int «Config.var_loop_counter_rows» = 0; «Config.var_loop_counter_rows» < «type.rows.concreteValue»; ++«Config.var_loop_counter_rows»){
				«streamName» << "[";
				for(int «Config.var_loop_counter_cols» = 0; «Config.var_loop_counter_cols» < «type.cols.concreteValue»; ++«Config.var_loop_counter_cols»){
					«IF type.blocksInRow == 1»
						«streamName» << «generateShowElements(m, State.arrayName, Config.var_loop_counter_rows + " * " + type.rows.concreteValue + " + " + Config.var_loop_counter_cols)»;
					«ELSE»
						«val index = "(" + Config.var_loop_counter_rows + "%" + type.rowsLocal + ") * " + type.colsLocal + " + (" + Config.var_loop_counter_rows + " / " + type.rowsLocal + ") * " + type.sizeLocal + " * " + type.blocksInColumn + " + (" + Config.var_loop_counter_cols + "/ " + type.colsLocal + ") * " + type.sizeLocal + " + " + Config.var_loop_counter_cols + "%" + type.colsLocal»
						«streamName» << «generateShowElements(m, State.arrayName, index)»;
					«ENDIF»
					if(«Config.var_loop_counter_cols» < «type.cols.concreteValue - 1»){
						«streamName» << "; ";
					}else{
						«streamName» << "]" << std::endl;
					}
				}
			}
			
			«streamName» << "]" << std::endl;
			printf("%s", «streamName».str().c_str());
		«IF Config.processes > 1»
			}
		«ENDIF»
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
