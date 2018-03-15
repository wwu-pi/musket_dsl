package de.wwu.musket.generator.extensions

/**
 * General helper for String manipulation.
 */
class StringExtension {
	
	/**
	 * This methods removes a line break at the end of a string.
	 * If there is none, nothing happens.
	 * 
	 * @param s String to manipulate
	 * @return String without line break in the end
	 */
	def static removeLineBreak(String s) {
		if(s.endsWith('\n')){
			s.substring(0, s.length-1)
		}
	}
	
	/**
	 * This methods removes all line breaks in a string.
	 * If there are none, nothing happens.
	 * 
	 * @param s String to manipulate
	 * @return String without line breaks 
	 */
	def static removeLineBreaks(String s) {
		s.replaceAll("\n", "")
	}
}
