package de.wwu.musket.generator

import de.wwu.musket.MusketStandaloneSetup
import org.apache.log4j.LogManager
import org.apache.log4j.Logger
import org.eclipse.emf.common.util.URI
import org.eclipse.xtext.generator.GeneratorDelegate
import org.eclipse.xtext.generator.JavaIoFileSystemAccess
import org.eclipse.xtext.resource.XtextResource
import org.eclipse.xtext.resource.XtextResourceSet
import org.eclipse.xtext.util.CancelIndicator
import org.eclipse.xtext.validation.CheckMode
import java.io.FileNotFoundException

/**
 * Starting point for the MusketStandaloneGenerator.
 * <p>
 * This can be used to generate models without opening an additional eclipse instance.
 * Just choose run as > java application.
 * If models should be added, include them in the list "val models = #[....]"
 * Remember to use relative paths.
 */
class MusketStandaloneGenerator {
	private static final Logger logger = LogManager.getLogger(MusketStandaloneGenerator)

	def static void main(String[] args) {

		logger.info("Start standalone Musket generator.")
		val injector = new MusketStandaloneSetup().createInjectorAndDoEMFRegistration()

		// obtain a resourceset from the injector
		val resourceSet = injector.getInstance(XtextResourceSet)

		val models = #[
//			'../de.wwu.musket.models/src/double.musket'
//			'../de.wwu.musket.models/src/fold.musket',
//			'../de.wwu.musket.models/src/frobenius.musket',
//			'../de.wwu.musket.models/src/frobenius_float.musket',
//			'../de.wwu.musket.models/src/fss.musket',
//			'../de.wwu.musket.models/src/fss-test.musket'
//			'../de.wwu.musket.models/src/matmult.musket',
//			'../de.wwu.musket.models/src/matmult_float.musket',
//			'../de.wwu.musket.models/src/matmult_float_transposed.musket',
			//'../de.wwu.musket.models/src/nbody.musket',
			//			'../de.wwu.musket.models/src/hlpp19/nbody-n-1-g-1.musket'
//			'../de.wwu.musket.models/src/nbody_float.musket',
//			'../de.wwu.musket.models/src/plus-row.musket',
//			'../de.wwu.musket.models/src/de/wwu/musket/models/test/map.musket',
//			'../de.wwu.musket.models/src/de/wwu/musket/models/test/map-function-array-matrix.musket',
//			'../de.wwu.musket.models/src/de/wwu/musket/models/test/scatter.musket',
//			'../de.wwu.musket.models/src/de/wwu/musket/models/test/gather.musket',
//			'../de.wwu.musket.models/src/de/wwu/musket/models/test/shift-partitions.musket',
//			'../de.wwu.musket.models/src/de/wwu/musket/models/test/zip.musket',
//			'../de.wwu.musket.models/src/de/wwu/musket/models/test/array.musket',
//			'../de.wwu.musket.models/src/de/wwu/musket/models/test/matrix.musket'
//			'../de.wwu.musket.models/src/de/wwu/musket/models/test/lambda.musket'
//			'../de.wwu.musket.models/src/de/wwu/musket/models/test/struct_test.musket',
//			'../de.wwu.musket.models/src/de/wwu/musket/models/test/acc_array_test.musket'
//			'../de.wwu.musket.models/src/mss.musket'
//			'../de.wwu.musket.models/src/pfb.musket',
//			'../de.wwu.musket.models/src/aco.musket',
			'../de.wwu.musket.models/src/aco_iroulette.musket',
			'../de.wwu.musket.models/src/aco_min.musket'

//			'../de.wwu.musket.models/src/pfb.musket'

		]

		val benchmark_names = #['frobenius', 'matmult', 'nbody', 'fss']
		val nodes = #[1, 4, 16]
		val gpus = #[1, 2, 4]

		var benchmark_models = newArrayList

		/*for (name : benchmark_names) {
			for (n : nodes) {
				for (g : gpus) {
					benchmark_models.add('../de.wwu.musket.models/src/hlpp19/' + name + '-n-' + n + '-g-' + g +
						'.musket')
				}
			}
		}*/

		benchmark_models.addAll(models)
 		
 
		for (String s : benchmark_models) {
			logger.info("Generate: " + s + '.')
			// load a resource by URI, in this case from the file system
			val resource = try{
				resourceSet.getResource(URI.createFileURI(s), true)
			}catch(RuntimeException e){
				logger.warn("File: " + s + " not found.")
				null
			}
			
			if(resource !== null){
			// Validation
			val validator = (resource as XtextResource).getResourceServiceProvider().getResourceValidator()
			val issues = validator.validate(resource, CheckMode.ALL, CancelIndicator.NullImpl)
			for (issue : issues) {
				logger.error(issue.getMessage())
			}

			// Code Generator
			val generator = injector.getInstance(GeneratorDelegate)
			val fsa = injector.getInstance(JavaIoFileSystemAccess)
			fsa.setOutputPath("../src-gen/")
			generator.doGenerate(resource, fsa)
			logger.info("Generate: " + s + '. Done.')
			}
		}
		logger.info("Musket standalone generator done.")
	}
}
