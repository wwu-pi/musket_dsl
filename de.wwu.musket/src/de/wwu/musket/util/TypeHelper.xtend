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
import de.wwu.musket.musket.CollectionObject
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
import de.wwu.musket.musket.Function
import de.wwu.musket.musket.IntArray
import de.wwu.musket.musket.IntArrayParameter
import de.wwu.musket.musket.IntConstant
import de.wwu.musket.musket.IntMatrix
import de.wwu.musket.musket.IntMatrixParameter
import de.wwu.musket.musket.IntParameter
import de.wwu.musket.musket.IntVal
import de.wwu.musket.musket.IntVariable
import de.wwu.musket.musket.InternalFunctionCall
import de.wwu.musket.musket.Modulo
import de.wwu.musket.musket.Multiplication
import de.wwu.musket.musket.MusketBoolVariable
import de.wwu.musket.musket.MusketDoubleVariable
import de.wwu.musket.musket.MusketFunctionCall
import de.wwu.musket.musket.MusketIntVariable
import de.wwu.musket.musket.MusketStructVariable
import de.wwu.musket.musket.Not
import de.wwu.musket.musket.ObjectRef
import de.wwu.musket.musket.Or
import de.wwu.musket.musket.ParameterInput
import de.wwu.musket.musket.PostDecrement
import de.wwu.musket.musket.PostIncrement
import de.wwu.musket.musket.PreDecrement
import de.wwu.musket.musket.PreIncrement
import de.wwu.musket.musket.Ref
import de.wwu.musket.musket.ReturnStatement
import de.wwu.musket.musket.SignedArithmetic
import de.wwu.musket.musket.StandardFunctionCall
import de.wwu.musket.musket.StringVal
import de.wwu.musket.musket.Struct
import de.wwu.musket.musket.StructArray
import de.wwu.musket.musket.StructArrayParameter
import de.wwu.musket.musket.StructMatrix
import de.wwu.musket.musket.StructMatrixParameter
import de.wwu.musket.musket.StructParameter
import de.wwu.musket.musket.StructVariable
import de.wwu.musket.musket.Subtraction
import de.wwu.musket.musket.TypeCast
import org.eclipse.emf.ecore.EObject

import static extension de.wwu.musket.util.CollectionHelper.*

class TypeHelper {
	static dispatch def MusketType calculateCollectionType(IntArray obj){
		return MusketType.INT
	}
	
	static dispatch def MusketType calculateCollectionType(DoubleArray obj){
		return MusketType.DOUBLE
	}
	
	static dispatch def MusketType calculateCollectionType(BoolArray obj){
		return MusketType.BOOL
	}
	
	static dispatch def MusketType calculateCollectionType(StructArray obj){
		return new MusketType(obj.type)
	}
	
	static dispatch def MusketType calculateCollectionType(IntMatrix obj){
		return MusketType.INT
	}
	
	static dispatch def MusketType calculateCollectionType(DoubleMatrix obj){
		return MusketType.DOUBLE
	}
	
	static dispatch def MusketType calculateCollectionType(BoolMatrix obj){
		return MusketType.BOOL
	}
	
	static dispatch def MusketType calculateCollectionType(StructMatrix obj){
		return new MusketType(obj.type).toMatrix
	}
	
	static dispatch def MusketType calculateCollectionType(IntArrayParameter obj){
		return MusketType.INT
	}
	
	static dispatch def MusketType calculateCollectionType(DoubleArrayParameter obj){
		return MusketType.DOUBLE
	}
	
	static dispatch def MusketType calculateCollectionType(BoolArrayParameter obj){
		return MusketType.BOOL
	}
	
	static dispatch def MusketType calculateCollectionType(StructArrayParameter obj){
		return new MusketType(obj.type).toArray
	}
	
	static dispatch def MusketType calculateCollectionType(IntMatrixParameter obj){
		return MusketType.INT
	}
	
	static dispatch def MusketType calculateCollectionType(DoubleMatrixParameter obj){
		return MusketType.DOUBLE
	}
	
	static dispatch def MusketType calculateCollectionType(BoolMatrixParameter obj){
		return MusketType.BOOL
	}
	
	static dispatch def MusketType calculateCollectionType(StructMatrixParameter obj){
		return new MusketType(obj.type).toMatrix
	}
		
	static dispatch def MusketType calculateCollectionType(ObjectRef obj){
		if((obj.globalCollectionIndex !== null && obj.globalCollectionIndex.size > 0) || (obj.localCollectionIndex !== null && obj.globalCollectionIndex.size > 0)) {
			// A collection _element_ has no collection type
			return null
		}
		
		return obj.value.calculateCollectionType
	}
	
	static dispatch def MusketType calculateCollectionType(MusketType t){
		return t.toSingleValued;
	}
	
	static dispatch def MusketType calculateCollectionType(CollectionObject obj){
		println("try to calculate collection type for " + obj)
		return null;
	}
	
	static dispatch def MusketType calculateCollectionType(ParameterInput obj){
		println("try to calculate collection type for " + obj)
		return null;
	}
	
	static dispatch def MusketType calculateType(IntVal exp){
		return MusketType.INT
	}
	
	static dispatch def MusketType calculateType(DoubleVal exp){
		return MusketType.DOUBLE
	}
	
	static dispatch def MusketType calculateType(BoolVal exp){
		return MusketType.BOOL
	}
	
	static dispatch def MusketType calculateType(StringVal exp){
		return MusketType.STRING
	}
	
	static dispatch def MusketType calculateType(InternalFunctionCall exp){
		return new MusketType(exp.value)
	}
	
	static dispatch def MusketType calculateType(StandardFunctionCall exp){
		if (exp.value === null) return null;
		
		switch exp.value {
			case ATOF: return MusketType.DOUBLE
			case ATOI: return MusketType.INT
			case ATOL: return MusketType.INT
			case ATOLL: return MusketType.INT
			case STRTOD: return MusketType.DOUBLE
			case STRTOF: return MusketType.DOUBLE
			case STRTOL: return MusketType.INT
			case STRTOLD: return MusketType.DOUBLE
			case STRTOLL: return MusketType.INT
			case STRTOUL: return MusketType.INT
			case STRTOULL: return MusketType.INT
			case RAND: return MusketType.INT
			case SRAND: return null
			case CALLOC: return null
			case FREE: return null
			case MALLOW: return null
			case REALLOC: return null
			case ABORT: return null
			case ATEXIT: return null
			case AT_QUICK_EXIT: return null
			case EXIT: return null
			case GETENV: return MusketType.STRING
			case QUICK_EXIT: return null
			case SYSTEM: return null
			case BSEARCH: return MusketType.INT
			case QSORT: return null
			case ABS: return MusketType.INT
			case DIV: return MusketType.INT
			case LABS: return MusketType.INT
			case LDIV: return MusketType.INT
			case LLABS: return MusketType.INT
			case LLDIV: return MusketType.INT
			case MBLEN: return MusketType.INT
			case MBTOWC: return MusketType.INT
			case WCTOMB: return MusketType.INT
			case MBSTOWCS: return MusketType.INT
			case WCSTOMBS: return MusketType.INT
			case REMOVE: return MusketType.INT
			case RENAME: return MusketType.INT
			case TMPFILE: return null
			case TMPNAM: return MusketType.STRING
			case FCLOSE: return MusketType.INT
			case FFLUSH: return MusketType.INT
			case FOPEN: return null
			case FREOPEN: return null
			case SETBUF: return null
			case SETVBUF: return MusketType.INT
			case FPRINTF: return MusketType.INT
			case FSCANF: return MusketType.INT
			case PRINTF: return MusketType.INT
			case SCANF: return MusketType.INT
			case SNPRINTF: return MusketType.INT
			case SPRINTF: return MusketType.INT
			case SSCANF: return MusketType.INT
			case VFPRINTF: return MusketType.INT
			case VFSCANF: return MusketType.INT
			case VPRINTF: return MusketType.INT
			case VSCANF: return MusketType.INT
			case VSNPRINTF: return MusketType.INT
			case VSPRINTF: return MusketType.INT
			case VSSCANF: return MusketType.INT
			case FGETC: return MusketType.INT
			case FGETS: return MusketType.STRING
			case FPUTC: return MusketType.INT
			case FPUTS: return MusketType.INT
			case GETC: return MusketType.INT
			case GETCHAR: return MusketType.INT
			case GETS: return MusketType.STRING
			case PUTC: return MusketType.INT
			case PUTCHAR: return MusketType.INT
			case PUTS: return MusketType.INT
			case UNGETC: return MusketType.INT
			case FREAD: return MusketType.INT
			case FWRITE: return MusketType.INT
			case FGETPOS: return MusketType.INT
			case FSEEK: return MusketType.INT
			case FSETPOS: return MusketType.INT
			case FTELL: return MusketType.INT
			case REWIND: return null
			case CLEARERR: return null
			case FEOF: return MusketType.INT
			case FERROR: return MusketType.INT
			case PERROR: return null
			case COS: return MusketType.DOUBLE
			case SIN: return MusketType.DOUBLE
			case TAN: return MusketType.DOUBLE
			case ACOS: return MusketType.DOUBLE
			case ASIN: return MusketType.DOUBLE
			case ATAN: return MusketType.DOUBLE
			case ATAN2: return MusketType.DOUBLE
			case COSH: return MusketType.DOUBLE
			case SINH: return MusketType.DOUBLE
			case TANH: return MusketType.DOUBLE
			case ACOSH: return MusketType.DOUBLE
			case ASINH: return MusketType.DOUBLE
			case ATANH: return MusketType.DOUBLE
			case EXP: return MusketType.DOUBLE
			case FREXP: return MusketType.DOUBLE
			case LDEXP: return MusketType.DOUBLE
			case LOG: return MusketType.DOUBLE
			case LOG10: return MusketType.DOUBLE
			case MODF: return MusketType.DOUBLE
			case EXP2: return MusketType.DOUBLE
			case EXPM1: return MusketType.DOUBLE
			case ILOGB: return MusketType.INT
			case LOG1P: return MusketType.DOUBLE
			case LOG2: return MusketType.DOUBLE
			case LOGB: return MusketType.DOUBLE
			case SCALBN: return MusketType.DOUBLE
			case SCALBLN: return MusketType.DOUBLE
			case POW: return MusketType.DOUBLE
			case SQRT: return MusketType.DOUBLE
			case CBRT: return MusketType.DOUBLE
			case HYPOT: return MusketType.DOUBLE
			case ERF: return MusketType.DOUBLE
			case ERFC: return MusketType.DOUBLE
			case TGAMMA: return MusketType.DOUBLE
			case LGAMMA: return MusketType.DOUBLE
			case CEIL: return MusketType.DOUBLE
			case FLOOR: return MusketType.DOUBLE
			case FMOD: return MusketType.DOUBLE
			case TRUNC: return MusketType.DOUBLE
			case ROUND: return MusketType.DOUBLE
			case LROUND: return MusketType.INT
			case LLROUND: return MusketType.INT
			case RINT: return MusketType.DOUBLE
			case LRINT: return MusketType.INT
			case LLRINT: return MusketType.INT
			case NEARBYINT: return MusketType.DOUBLE
			case REMAINDER: return MusketType.DOUBLE
			case REMQUO: return MusketType.DOUBLE
			case COPYSIGN: return MusketType.DOUBLE
			case NAN: return MusketType.DOUBLE
			case NEXTAFTER: return MusketType.DOUBLE
			case NEXTTOWARD: return MusketType.DOUBLE
			case FDIM: return MusketType.DOUBLE
			case FMAX: return MusketType.DOUBLE
			case FMIN: return MusketType.DOUBLE
			case FABS: return MusketType.DOUBLE
			case FMA: return MusketType.DOUBLE
			case FPCLASSIFY: return MusketType.INT
			case ISFINITE: return MusketType.BOOL
			case ISINF: return MusketType.BOOL
			case ISNAN: return MusketType.BOOL
			case ISNORMAL: return MusketType.BOOL
			case SIGNBIT: return MusketType.BOOL
			case ISGREATER: return MusketType.BOOL
			case ISGREATEREQUAL: return MusketType.BOOL
			case ISLESS: return MusketType.BOOL
			case ISLESSEQUAL: return MusketType.BOOL
			case ISLESSGREATER: return MusketType.BOOL
			case ISUNORDERED: return MusketType.BOOL
			case MEMCPY: return null
			case MEMMOVE: return null
			case STRCPY: return MusketType.STRING
			case STRNCPY: return MusketType.STRING
			case STRCAT: return MusketType.STRING
			case STRNCAT: return MusketType.STRING
			case MEMCMP: return MusketType.INT
			case STRCMP: return MusketType.INT
			case STRCOLL: return MusketType.INT
			case STRNCMP: return MusketType.INT
			case STRXFRM: return MusketType.INT
			case MEMCHR: return null
			case STRCHR: return null
			case STRCSPN: return MusketType.INT
			case STRPBRK: return null
			case STRRCHR: return null
			case STRSPN: return MusketType.INT
			case STRSTR: return null
			case STRTOK: return null
			case MEMSET: return null
			case STRERROR: return MusketType.STRING
			case STRLEN: return MusketType.INT
		}
	}
	
	static dispatch def MusketType calculateType(MusketFunctionCall exp){
		if (exp.value === null) return null;
		
		switch exp.value {
			case PRINT: return MusketType.STRING
			case RAND: return MusketType.DOUBLE
		}
	}
	
	static dispatch def MusketType calculateType(IntConstant exp){
		return MusketType.INT
	}
	
	static dispatch def MusketType calculateType(DoubleConstant exp){
		return MusketType.DOUBLE
	}
	
	static dispatch def MusketType calculateType(BoolConstant exp){
		return MusketType.BOOL
	}
	
	static dispatch def MusketType calculateType(MusketIntVariable exp){
		return MusketType.INT
	}
	
	static dispatch def MusketType calculateType(MusketDoubleVariable exp){
		return MusketType.DOUBLE
	}
	
	static dispatch def MusketType calculateType(MusketBoolVariable exp){
		return MusketType.BOOL
	}
	
	static dispatch def MusketType calculateType(MusketStructVariable exp){
		return new MusketType(exp.type)
	}
	
	static dispatch def MusketType calculateType(IntVariable exp){
		return MusketType.INT
	}
	
	static dispatch def MusketType calculateType(DoubleVariable exp){
		return MusketType.DOUBLE
	}
	
	static dispatch def MusketType calculateType(StructVariable exp){
		return new MusketType(exp.type)
	}
	
	static dispatch def MusketType calculateType(BoolVariable exp){
		return MusketType.BOOL
	}
	
	static dispatch def MusketType calculateType(IntParameter exp){
		return MusketType.INT
	}
	
	static dispatch def MusketType calculateType(DoubleParameter exp){
		return MusketType.DOUBLE
	}
	
	static dispatch def MusketType calculateType(BoolParameter exp){
		return MusketType.BOOL
	}
	
	static dispatch def MusketType calculateType(StructParameter exp){
		return new MusketType(exp.type)
	}
	
	static dispatch def MusketType calculateType(IntArrayParameter exp){
		return MusketType.INT_ARRAY
	}
	
	static dispatch def MusketType calculateType(DoubleArrayParameter exp){
		return MusketType.DOUBLE_ARRAY
	}
	
	static dispatch def MusketType calculateType(BoolArrayParameter exp){
		return MusketType.BOOL_ARRAY
	}
	
	static dispatch def MusketType calculateType(StructArrayParameter exp){
		return new MusketType(exp.type).toArray
	}
	
	static dispatch def MusketType calculateType(IntMatrixParameter exp){
		return MusketType.INT_MATRIX
	}
	
	static dispatch def MusketType calculateType(DoubleMatrixParameter exp){
		return MusketType.DOUBLE_MATRIX
	}
	
	static dispatch def MusketType calculateType(BoolMatrixParameter exp){
		return MusketType.BOOL_MATRIX
	}
	
	static dispatch def MusketType calculateType(StructMatrixParameter exp){
		return new MusketType(exp.type).toMatrix
	}
	
	static dispatch def MusketType calculateType(IntArray exp){
		return MusketType.INT_ARRAY
	}
	
	static dispatch def MusketType calculateType(DoubleArray exp){
		return MusketType.DOUBLE_ARRAY
	}
	
	static dispatch def MusketType calculateType(BoolArray exp){
		return MusketType.BOOL_ARRAY
	}
	
	static dispatch def MusketType calculateType(StructArray exp){
		return new MusketType(exp.type).toArray
	}
	
	static dispatch def MusketType calculateType(IntMatrix exp){
		return MusketType.INT_MATRIX
	}
	
	static dispatch def MusketType calculateType(DoubleMatrix exp){
		return MusketType.DOUBLE_MATRIX
	}
	
	static dispatch def MusketType calculateType(BoolMatrix exp){
		return MusketType.BOOL_MATRIX
	}
	
	static dispatch def MusketType calculateType(StructMatrix exp){
		return new MusketType(exp.type).toMatrix
	}
	
	static dispatch def MusketType calculateType(Ref exp){
		// Go down nested reference structure
		if(exp.tail !== null) return exp.tail.calculateType
		
		// Type of collection _element_ 
		if(exp.isCollectionRef) return exp.value.calculateCollectionType
		
		// Other object references
		return exp.value.calculateType
	}
	
	static dispatch def MusketType calculateType(SignedArithmetic exp){
		return exp.expression.calculateType
	}
	
	static dispatch def MusketType calculateType(TypeCast exp){
		return new MusketType(exp.targetType)
	}
	
	static dispatch def MusketType calculateType(PostIncrement exp){
		return exp.value.calculateType
	}
	
	static dispatch def MusketType calculateType(PostDecrement exp){
		return exp.value.calculateType
	}
	
	static dispatch def MusketType calculateType(PreIncrement exp){
		return exp.value.calculateType
	}
	
	static dispatch def MusketType calculateType(PreDecrement exp){
		return exp.value.calculateType
	}
	
	static dispatch def MusketType calculateType(Addition exp){
		if(exp.right !== null) {
			if(!exp.left.calculateType?.isNumeric || !exp.right.calculateType?.isNumeric) return null // Calculation error
			if(exp.right.calculateType == MusketType.DOUBLE) return MusketType.DOUBLE // anything plus double results in double
		}
		return exp.left.calculateType 
	}
	
	static dispatch def MusketType calculateType(Subtraction exp){
		if(exp.right !== null) {
			if(!exp.left.calculateType?.isNumeric || !exp.right.calculateType?.isNumeric) return null // Calculation error
			if(exp.right.calculateType == MusketType.DOUBLE) return MusketType.DOUBLE // anything minus double results in double
		}
		return exp.left.calculateType 
	}
	
	static dispatch def MusketType calculateType(Multiplication exp){
		if(exp.right !== null) {
			if(!exp.left.calculateType?.isNumeric || !exp.right.calculateType?.isNumeric) return null // Calculation error
			if(exp.right.calculateType == MusketType.DOUBLE) return MusketType.DOUBLE // anything times double results in double
		}
		return exp.left.calculateType 
	}
	
	static dispatch def MusketType calculateType(Division exp){
		if(exp.right !== null) {
			if(!exp.left.calculateType?.isNumeric || !exp.right.calculateType?.isNumeric) return null // Calculation error
			return exp.right.calculateType // division result depends on right side 
		}
		return exp.left.calculateType 
	}
	
	static dispatch def MusketType calculateType(Modulo exp){
		if(exp.right !== null) {
			if(exp.left.calculateType != MusketType.INT || exp.right.calculateType != MusketType.INT) return null // Modulo requires two ints 
		}
		return exp.left.calculateType
	}
	
	static dispatch def MusketType calculateType(Not exp){
		return MusketType.BOOL
	}
	
	static dispatch def MusketType calculateType(And exp){
		if(exp.rightExpression !== null) return MusketType.BOOL
		return exp.leftExpression.calculateType 
	}
	
	static dispatch def MusketType calculateType(Or exp){
		if(exp.rightExpression !== null) return MusketType.BOOL
		return exp.leftExpression.calculateType
	}
	
	static dispatch def MusketType calculateType(CompareExpression exp){
		if(exp.eqRight !== null) return MusketType.BOOL
		return exp.eqLeft?.calculateType
	}
	
	static dispatch def MusketType calculateType(Function exp){
		return new MusketType(exp)
	}
	
	static dispatch def MusketType calculateType(ReturnStatement exp){
		return exp.value.calculateType
	}
	
	static dispatch def MusketType calculateType(EObject exp){ // Else case
		println("try to calculate type for unknown object " + exp)
		return null
	}
}