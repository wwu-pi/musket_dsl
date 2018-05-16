package de.wwu.musket.generator.preprocessor

import de.wwu.musket.generator.preprocessor.transformations.MapVariantTransformation
import de.wwu.musket.generator.preprocessor.transformations.PreprocessorTransformation
import de.wwu.musket.generator.preprocessor.util.MusketComplexElementFactory
import org.eclipse.emf.ecore.resource.Resource

import static de.wwu.musket.generator.preprocessor.util.PreprocessorUtil.*
import java.util.Collection
import de.wwu.musket.generator.preprocessor.transformations.MapFusionTransformation
import de.wwu.musket.generator.preprocessor.transformations.DummyTransformation

class MusketPreprocessor {
	
	/**
	 * Singleton instance of this preprocessor.
	 */
	protected static MusketPreprocessor instance
	
	/**
	 * The element factory to be used.
	 */
	protected static MusketComplexElementFactory factory
	
	/**
	 * A reference to the previous (unpreprocessed) resource set is stored to compare it with
	 * the current one. If they equal, the preprocessing can be skipped and the stored preprocessedModel
	 * is returned.
	 */
	private static Resource unprocessedModel

	/**
	 * Cloned working model, only generated once and stored in this class attribute.
	 */
	private static Resource preprocessedModel
	
	/**
	 * Constructor ensures to initialize the preprocessor with the factory to be used to 
	 * generate new model elements.
	 */
	new() {
		factory = new MusketComplexElementFactory;
	}
		
	/**
	 * For each different model resource input, generate the preprocessed model.
	 * The preprocessed Resource is stored in a class attribute, so that for each unique
	 * input model the preprocessor is only run once. Normally, for all generators the input
	 * model is the same so that this factory should be considerably faster.
	 * 
	 * @param input - Original (unprocessed) model.
	 * @return Resource with the preprocessed models.
	 */
	def static getPreprocessedModel(Resource input) {
		if (!input.equals(unprocessedModel)) {
			if (instance === null) {
				instance = new MusketPreprocessor()
			}
			unprocessedModel = copyModel(input)
			preprocessedModel = instance.preprocessModel
		}
		return preprocessedModel
	}
	
	/**
	 * Actual preprocessing
	 */
	private def Resource preprocessModel() {
		
		val workingModel = copyModel(unprocessedModel)
		
		val Collection<PreprocessorTransformation> transformations = #[
			new MapVariantTransformation(factory),
			new MapFusionTransformation(factory),
			new DummyTransformation(factory)
		]
				
		transformations.forEach[it.run(workingModel)]
		
		return workingModel
	}
}