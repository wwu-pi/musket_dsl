package de.wwu.musket.generator.cpu.mpmd

class ShiftSkeletonGenerator {
	def static generateShiftSkeletonVariables(int processId) '''
		int «Config.var_shift_source» = «processId»;
		int «Config.var_shift_target» = «processId»;
		int «Config.var_shift_steps» = 0;
	'''

}