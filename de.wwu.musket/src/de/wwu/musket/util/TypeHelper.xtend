package de.wwu.musket.util

import de.wwu.musket.musket.Addition
import de.wwu.musket.musket.And
import de.wwu.musket.musket.BoolArray
import de.wwu.musket.musket.BoolArrayParameter
import de.wwu.musket.musket.BoolConstant
import de.wwu.musket.musket.BoolMatrix
import de.wwu.musket.musket.BoolMatrixParameter
import de.wwu.musket.musket.BoolParameter
import de.wwu.musket.musket.BoolVal
import de.wwu.musket.musket.BoolVariable
import de.wwu.musket.musket.CollectionElementRef
import de.wwu.musket.musket.CompareExpression
import de.wwu.musket.musket.Division
import de.wwu.musket.musket.DoubleArray
import de.wwu.musket.musket.DoubleArrayParameter
import de.wwu.musket.musket.DoubleConstant
import de.wwu.musket.musket.DoubleMatrix
import de.wwu.musket.musket.DoubleMatrixParameter
import de.wwu.musket.musket.DoubleParameter
import de.wwu.musket.musket.DoubleVal
import de.wwu.musket.musket.DoubleVariable
import de.wwu.musket.musket.IntArray
import de.wwu.musket.musket.IntArrayParameter
import de.wwu.musket.musket.IntConstant
import de.wwu.musket.musket.IntMatrix
import de.wwu.musket.musket.IntMatrixParameter
import de.wwu.musket.musket.IntParameter
import de.wwu.musket.musket.IntVal
import de.wwu.musket.musket.IntVariable
import de.wwu.musket.musket.InternalFunctionCall
import de.wwu.musket.musket.Multiplication
import de.wwu.musket.musket.MusketBoolVariable
import de.wwu.musket.musket.MusketDoubleVariable
import de.wwu.musket.musket.MusketFunctionCall
import de.wwu.musket.musket.MusketIntVariable
import de.wwu.musket.musket.Not
import de.wwu.musket.musket.ObjectRef
import de.wwu.musket.musket.Or
import de.wwu.musket.musket.ParameterInput
import de.wwu.musket.musket.PostDecrement
import de.wwu.musket.musket.PostIncrement
import de.wwu.musket.musket.PreDecrement
import de.wwu.musket.musket.PreIncrement
import de.wwu.musket.musket.SignedArithmetic
import de.wwu.musket.musket.StandardFunctionCall
import de.wwu.musket.musket.StringVal
import de.wwu.musket.musket.Subtraction
import de.wwu.musket.musket.Type
import org.eclipse.emf.ecore.EObject
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.StructArray
import de.wwu.musket.musket.StructMatrixParameter
import de.wwu.musket.musket.StructArrayParameter
import de.wwu.musket.musket.StructMatrix
import de.wwu.musket.musket.StructParameter

class TypeHelper {
	
	static def isNumeric(Type type){
		return type === Type.INT || type === Type.DOUBLE
	}
	
	static def isCollection(Type type){
		return type === Type.INT_ARRAY || type === Type.DOUBLE_ARRAY  || type === Type.BOOL_ARRAY  || type === Type.STRING_ARRAY
		   || type === Type.INT_MATRIX || type === Type.DOUBLE_MATRIX || type === Type.BOOL_MATRIX || type === Type.STRING_MATRIX
		   || type === Type.STRUCT_ARRAY || type === Type.STRUCT_MATRIX
		//return input instanceof ObjectRef && (input as ObjectRef).value instanceof CollectionObject
	}
	
	// Helper to check the expression type of a collection
	static dispatch def Type calculateCollectionType(IntArray obj){
		return Type.INT
	}
	
	static dispatch def Type calculateCollectionType(DoubleArray obj){
		return Type.DOUBLE
	}
	
	static dispatch def Type calculateCollectionType(BoolArray obj){
		return Type.BOOL
	}
	
	static dispatch def Type calculateCollectionType(StructArray obj){
		return Type.STRUCT
	}
	
	static dispatch def Type calculateCollectionType(IntMatrix obj){
		return Type.INT
	}
	
	static dispatch def Type calculateCollectionType(DoubleMatrix obj){
		return Type.DOUBLE
	}
	
	static dispatch def Type calculateCollectionType(BoolMatrix obj){
		return Type.BOOL
	}
	
	static dispatch def Type calculateCollectionType(StructMatrix obj){
		return Type.STRUCT
	}
	
	static dispatch def Type calculateCollectionType(IntArrayParameter obj){
		return Type.INT
	}
	
	static dispatch def Type calculateCollectionType(DoubleArrayParameter obj){
		return Type.DOUBLE
	}
	
	static dispatch def Type calculateCollectionType(BoolArrayParameter obj){
		return Type.BOOL
	}
	
	static dispatch def Type calculateCollectionType(StructArrayParameter obj){
		return Type.STRUCT
	}
	
	static dispatch def Type calculateCollectionType(IntMatrixParameter obj){
		return Type.INT
	}
	
	static dispatch def Type calculateCollectionType(DoubleMatrixParameter obj){
		return Type.DOUBLE
	}
	
	static dispatch def Type calculateCollectionType(BoolMatrixParameter obj){
		return Type.BOOL
	}
	
	static dispatch def Type calculateCollectionType(StructMatrixParameter obj){
		return Type.STRUCT
	}
		
	static dispatch def Type calculateCollectionType(CollectionElementRef obj){
		return null // a collection _element_ is no collection itself
	}
	
	static dispatch def Type calculateCollectionType(ObjectRef obj){
		return obj.value.calculateCollectionType
	}
	
	static dispatch def Type calculateCollectionType(ParameterInput obj){
		println("try to calculate collection type for " + obj)
		return null;
	}
	
	// Helper to check the expression type (hard to see within type hierarchy)
	// TODO handle collections??
	static dispatch def Type calculateType(IntVal exp){
		return Type.INT
	}
	
	static dispatch def Type calculateType(DoubleVal exp){
		return Type.DOUBLE
	}
	
	static dispatch def Type calculateType(BoolVal exp){
		return Type.BOOL
	}
	
	static dispatch def Type calculateType(StringVal exp){
		return Type.STRING
	}
	
	static dispatch def Type calculateType(InternalFunctionCall exp){
		return if (exp.value?.returnType !== null) Type.STRUCT else exp.value?.returnTypePrimitive
	}
	
	static dispatch def Type calculateType(StandardFunctionCall exp){
		if (exp.value === null) return null;
		
		switch exp.value {
			case ATOF: return Type.DOUBLE
			case ATOI: return Type.INT
			case ATOL: return Type.INT
			case ATOLL: return Type.INT
			case STRTOD: return Type.DOUBLE
			case STRTOF: return Type.DOUBLE
			case STRTOL: return Type.INT
			case STRTOLD: return Type.DOUBLE
			case STRTOLL: return Type.INT
			case STRTOUL: return Type.INT
			case STRTOULL: return Type.INT
			case RAND: return Type.INT
			case SRAND: return null
			case CALLOC: return null
			case FREE: return null
			case MALLOW: return null
			case REALLOC: return null
			case ABORT: return null
			case ATEXIT: return null
			case AT_QUICK_EXIT: return null
			case EXIT: return null
			case GETENV: return Type.STRING
			case QUICK_EXIT: return null
			case SYSTEM: return null
			case BSEARCH: return Type.INT
			case QSORT: return null
			case ABS: return Type.INT
			case DIV: return Type.INT
			case LABS: return Type.INT
			case LDIV: return Type.INT
			case LLABS: return Type.INT
			case LLDIV: return Type.INT
			case MBLEN: return Type.INT
			case MBTOWC: return Type.INT
			case WCTOMB: return Type.INT
			case MBSTOWCS: return Type.INT
			case WCSTOMBS: return Type.INT
			case REMOVE: return Type.INT
			case RENAME: return Type.INT
			case TMPFILE: return null
			case TMPNAM: return Type.STRING
			case FCLOSE: return Type.INT
			case FFLUSH: return Type.INT
			case FOPEN: return null
			case FREOPEN: return null
			case SETBUF: return null
			case SETVBUF: return Type.INT
			case FPRINTF: return Type.INT
			case FSCANF: return Type.INT
			case PRINTF: return Type.INT
			case SCANF: return Type.INT
			case SNPRINTF: return Type.INT
			case SPRINTF: return Type.INT
			case SSCANF: return Type.INT
			case VFPRINTF: return Type.INT
			case VFSCANF: return Type.INT
			case VPRINTF: return Type.INT
			case VSCANF: return Type.INT
			case VSNPRINTF: return Type.INT
			case VSPRINTF: return Type.INT
			case VSSCANF: return Type.INT
			case FGETC: return Type.INT
			case FGETS: return Type.STRING
			case FPUTC: return Type.INT
			case FPUTS: return Type.INT
			case GETC: return Type.INT
			case GETCHAR: return Type.INT
			case GETS: return Type.STRING
			case PUTC: return Type.INT
			case PUTCHAR: return Type.INT
			case PUTS: return Type.INT
			case UNGETC: return Type.INT
			case FREAD: return Type.INT
			case FWRITE: return Type.INT
			case FGETPOS: return Type.INT
			case FSEEK: return Type.INT
			case FSETPOS: return Type.INT
			case FTELL: return Type.INT
			case REWIND: return null
			case CLEARERR: return null
			case FEOF: return Type.INT
			case FERROR: return Type.INT
			case PERROR: return null
			case COS: return Type.DOUBLE
			case SIN: return Type.DOUBLE
			case TAN: return Type.DOUBLE
			case ACOS: return Type.DOUBLE
			case ASIN: return Type.DOUBLE
			case ATAN: return Type.DOUBLE
			case ATAN2: return Type.DOUBLE
			case COSH: return Type.DOUBLE
			case SINH: return Type.DOUBLE
			case TANH: return Type.DOUBLE
			case ACOSH: return Type.DOUBLE
			case ASINH: return Type.DOUBLE
			case ATANH: return Type.DOUBLE
			case EXP: return Type.DOUBLE
			case FREXP: return Type.DOUBLE
			case LDEXP: return Type.DOUBLE
			case LOG: return Type.DOUBLE
			case LOG10: return Type.DOUBLE
			case MODF: return Type.DOUBLE
			case EXP2: return Type.DOUBLE
			case EXPM1: return Type.DOUBLE
			case ILOGB: return Type.INT
			case LOG1P: return Type.DOUBLE
			case LOG2: return Type.DOUBLE
			case LOGB: return Type.DOUBLE
			case SCALBN: return Type.DOUBLE
			case SCALBLN: return Type.DOUBLE
			case POW: return Type.DOUBLE
			case SQRT: return Type.DOUBLE
			case CBRT: return Type.DOUBLE
			case HYPOT: return Type.DOUBLE
			case ERF: return Type.DOUBLE
			case ERFC: return Type.DOUBLE
			case TGAMMA: return Type.DOUBLE
			case LGAMMA: return Type.DOUBLE
			case CEIL: return Type.DOUBLE
			case FLOOR: return Type.DOUBLE
			case FMOD: return Type.DOUBLE
			case TRUNC: return Type.DOUBLE
			case ROUND: return Type.DOUBLE
			case LROUND: return Type.INT
			case LLROUND: return Type.INT
			case RINT: return Type.DOUBLE
			case LRINT: return Type.INT
			case LLRINT: return Type.INT
			case NEARBYINT: return Type.DOUBLE
			case REMAINDER: return Type.DOUBLE
			case REMQUO: return Type.DOUBLE
			case COPYSIGN: return Type.DOUBLE
			case NAN: return Type.DOUBLE
			case NEXTAFTER: return Type.DOUBLE
			case NEXTTOWARD: return Type.DOUBLE
			case FDIM: return Type.DOUBLE
			case FMAX: return Type.DOUBLE
			case FMIN: return Type.DOUBLE
			case FABS: return Type.DOUBLE
			case FMA: return Type.DOUBLE
			case FPCLASSIFY: return Type.INT
			case ISFINITE: return Type.BOOL
			case ISINF: return Type.BOOL
			case ISNAN: return Type.BOOL
			case ISNORMAL: return Type.BOOL
			case SIGNBIT: return Type.BOOL
			case ISGREATER: return Type.BOOL
			case ISGREATEREQUAL: return Type.BOOL
			case ISLESS: return Type.BOOL
			case ISLESSEQUAL: return Type.BOOL
			case ISLESSGREATER: return Type.BOOL
			case ISUNORDERED: return Type.BOOL
			case MEMCPY: return null
			case MEMMOVE: return null
			case STRCPY: return Type.STRING
			case STRNCPY: return Type.STRING
			case STRCAT: return Type.STRING
			case STRNCAT: return Type.STRING
			case MEMCMP: return Type.INT
			case STRCMP: return Type.INT
			case STRCOLL: return Type.INT
			case STRNCMP: return Type.INT
			case STRXFRM: return Type.INT
			case MEMCHR: return null
			case STRCHR: return null
			case STRCSPN: return Type.INT
			case STRPBRK: return null
			case STRRCHR: return null
			case STRSPN: return Type.INT
			case STRSTR: return null
			case STRTOK: return null
			case MEMSET: return null
			case STRERROR: return Type.STRING
			case STRLEN: return Type.INT
		}
	}
	
	static dispatch def Type calculateType(MusketFunctionCall exp){
		if (exp.value === null) return null;
		
		switch exp.value {
			case PRINT: return Type.STRING
			case RAND: return Type.DOUBLE
		}
	}
	
	static dispatch def Type calculateType(IntConstant exp){
		return Type.INT
	}
	
	static dispatch def Type calculateType(DoubleConstant exp){
		return Type.DOUBLE
	}
	
	static dispatch def Type calculateType(BoolConstant exp){
		return Type.BOOL
	}
	
	static dispatch def Type calculateType(MusketIntVariable exp){
		return Type.INT
	}
	
	static dispatch def Type calculateType(MusketDoubleVariable exp){
		return Type.DOUBLE
	}
	
	static dispatch def Type calculateType(MusketBoolVariable exp){
		return Type.BOOL
	}
	
	static dispatch def Type calculateType(IntVariable exp){
		return Type.INT
	}
	
	static dispatch def Type calculateType(DoubleVariable exp){
		return Type.DOUBLE
	}
	
	static dispatch def Type calculateType(BoolVariable exp){
		return Type.BOOL
	}
	
	static dispatch def Type calculateType(IntParameter exp){
		return Type.INT
	}
	
	static dispatch def Type calculateType(DoubleParameter exp){
		return Type.DOUBLE
	}
	
	static dispatch def Type calculateType(BoolParameter exp){
		return Type.BOOL
	}
	
	static dispatch def Type calculateType(StructParameter exp){
		return Type.STRUCT
	}
	
	static dispatch def Type calculateType(IntArrayParameter exp){
		return Type.INT_ARRAY
	}
	
	static dispatch def Type calculateType(DoubleArrayParameter exp){
		return Type.DOUBLE_ARRAY
	}
	
	static dispatch def Type calculateType(BoolArrayParameter exp){
		return Type.BOOL_ARRAY
	}
	
	static dispatch def Type calculateType(StructArrayParameter exp){
		return Type.STRUCT_ARRAY
	}
	
	static dispatch def Type calculateType(IntMatrixParameter exp){
		return Type.INT_MATRIX
	}
	
	static dispatch def Type calculateType(DoubleMatrixParameter exp){
		return Type.DOUBLE_MATRIX
	}
	
	static dispatch def Type calculateType(BoolMatrixParameter exp){
		return Type.BOOL_MATRIX
	}
	
	static dispatch def Type calculateType(StructMatrixParameter exp){
		return Type.STRUCT_MATRIX
	}
	
	static dispatch def Type calculateType(IntArray exp){
		return Type.INT_ARRAY
	}
	
	static dispatch def Type calculateType(DoubleArray exp){
		return Type.DOUBLE_ARRAY
	}
	
	static dispatch def Type calculateType(BoolArray exp){
		return Type.BOOL_ARRAY
	}
	
	static dispatch def Type calculateType(StructArray exp){
		return Type.STRUCT_ARRAY
	}
	
	static dispatch def Type calculateType(IntMatrix exp){
		return Type.INT_MATRIX
	}
	
	static dispatch def Type calculateType(DoubleMatrix exp){
		return Type.DOUBLE_MATRIX
	}
	
	static dispatch def Type calculateType(BoolMatrix exp){
		return Type.BOOL_MATRIX
	}
	
	static dispatch def Type calculateType(StructMatrix exp){
		return Type.STRUCT_MATRIX
	}
	
	static dispatch def Type calculateType(ObjectRef exp){
		if(exp instanceof CollectionElementRef) return exp.calculateCollectionType
		return exp.value.calculateType
	}
	
	static dispatch def Type calculateType(SignedArithmetic exp){
		return exp.expression.calculateType
	}
	
	static dispatch def Type calculateType(PostIncrement exp){
		return exp.value.calculateType
	}
	
	static dispatch def Type calculateType(PostDecrement exp){
		return exp.value.calculateType
	}
	
	static dispatch def Type calculateType(PreIncrement exp){
		return exp.value.calculateType
	}
	
	static dispatch def Type calculateType(PreDecrement exp){
		return exp.value.calculateType
	}
	
	static dispatch def Type calculateType(Addition exp){
		if(exp.right !== null) {
			if(!exp.left.calculateType?.isNumeric || !exp.right.calculateType?.isNumeric) return null // Calculation error
			if(exp.right.calculateType === Type.DOUBLE) return Type.DOUBLE // anything plus double results in double
		}
		return exp.left.calculateType 
	}
	
	static dispatch def Type calculateType(Subtraction exp){
		if(exp.right !== null) {
			if(!exp.left.calculateType?.isNumeric || !exp.right.calculateType?.isNumeric) return null // Calculation error
			if(exp.right.calculateType === Type.DOUBLE) return Type.DOUBLE // anything minus double results in double
		}
		return exp.left.calculateType 
	}
	
	static dispatch def Type calculateType(Multiplication exp){
		if(exp.right !== null) {
			if(!exp.left.calculateType?.isNumeric || !exp.right.calculateType?.isNumeric) return null // Calculation error
			if(exp.right.calculateType === Type.DOUBLE) return Type.DOUBLE // anything times double results in double
		}
		return exp.left.calculateType 
	}
	
	static dispatch def Type calculateType(Division exp){
		if(exp.right !== null) {
			if(!exp.left.calculateType?.isNumeric || !exp.right.calculateType?.isNumeric) return null // Calculation error
			return exp.right.calculateType // division result depends on right side 
		}
		return exp.left.calculateType 
	}
	
	static dispatch def Type calculateType(Not exp){
		return Type.BOOL
	}
	
	static dispatch def Type calculateType(And exp){
		if(exp.rightExpression !== null) return Type.BOOL
		return exp.leftExpression.calculateType 
	}
	
	static dispatch def Type calculateType(Or exp){
		if(exp.rightExpression !== null) return Type.BOOL
		return exp.leftExpression.calculateType
	}
	
	static dispatch def Type calculateType(CompareExpression exp){
		if(exp.eqRight !== null) return Type.BOOL
		return exp.eqLeft?.calculateType
	}
	
	static dispatch def Type calculateType(Function exp){
		if(exp.returnType !== null) return Type.STRUCT
		return exp.returnTypePrimitive
	}
	
	static dispatch def Type calculateType(EObject exp){ // Else case
		println("try to calculate type for " + exp)
		return null
	}
}