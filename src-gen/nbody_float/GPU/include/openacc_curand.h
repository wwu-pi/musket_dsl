 /* Copyright 2010-2015 NVIDIA Corporation.  All rights reserved.
  *
  * NOTICE TO LICENSEE:
  *
  * The source code and/or documentation ("Licensed Deliverables") are
  * subject to NVIDIA intellectual property rights under U.S. and
  * international Copyright laws.
  *
  * The Licensed Deliverables contained herein are PROPRIETARY and
  * CONFIDENTIAL to NVIDIA and are being provided under the terms and
  * conditions of a form of NVIDIA software license agreement by and
  * between NVIDIA and Licensee ("License Agreement") or electronically
  * accepted by Licensee.  Notwithstanding any terms or conditions to
  * the contrary in the License Agreement, reproduction or disclosure
  * of the Licensed Deliverables to any third party without the express
  * written consent of NVIDIA is prohibited.
  *
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
  * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
  * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
  * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
  * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
  * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
  * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
  * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
  * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
  * OF THESE LICENSED DELIVERABLES.
  *
  * U.S. Government End Users.  These Licensed Deliverables are a
  * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
  * 1995), consisting of "commercial computer software" and "commercial
  * computer software documentation" as such terms are used in 48
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
  * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
  * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
  * U.S. Government End Users acquire the Licensed Deliverables with
  * only those rights set forth herein.
  *
  * Any use of the Licensed Deliverables in individual and commercial
  * software must include, in the user documentation and internal
  * comments to the code, the above Disclaimer and U.S. Government End
  * Users Notice.
  */


#if !defined(CURAND_KERNEL_H_)
#define CURAND_KERNEL_H_

/**
 * \defgroup DEVICE Device API
 *
 * @{
 */

#if !defined(QUALIFIERS)
#define QUALIFIERS static __forceinline__
#endif



/* Test RNG */
/* This generator uses the formula:
   x_n = x_(n-1) + 1 mod 2^32
   x_0 = (unsigned int)seed * 3
   Subsequences are spaced 31337 steps apart.
*/
struct curandStateTest {
    unsigned int v;
};

/** \cond UNHIDE_TYPEDEFS */
typedef struct curandStateTest curandStateTest_t;
/** \endcond */

/* XORSHIFT FAMILY RNGs */
/* These generators are a family proposed by Marsaglia.  They keep state
   in 32 bit chunks, then use repeated shift and xor operations to scramble
   the bits.  The following generators are a combination of a simple Weyl
   generator with an N variable XORSHIFT generator.
*/

/* XORSHIFT RNG */
/* This generator uses the xorwow formula of
www.jstatsoft.org/v08/i14/paper page 5
Has period 2^192 - 2^32.
*/
/**
 * CURAND XORWOW state 
 * Implementation details not in reference documentation */
struct curandStateXORWOW {
    unsigned int d, v[5];
    int boxmuller_flag;
    int boxmuller_flag_double;
    float boxmuller_extra;
    double boxmuller_extra_double;
};

/*
 * CURAND XORWOW state 
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct curandStateXORWOW curandStateXORWOW_t;

#define EXTRA_FLAG_NORMAL         0x00000001
#define EXTRA_FLAG_LOG_NORMAL     0x00000002
/** \endcond */

/* Combined Multiple Recursive Generators */
/* These generators are a family proposed by L'Ecuyer.  They keep state
   in sets of doubles, then use repeated modular arithmetic multiply operations 
   to scramble the bits in each set, and combine the result.
*/

/* MRG32k3a RNG */
/* This generator uses the MRG32k3A formula of
http://www.iro.umontreal.ca/~lecuyer/myftp/streams00/c++/streams4.pdf
Has period 2^191.
*/

/* moduli for the recursions */
/** \cond UNHIDE_DEFINES */
#define MRG32K3A_MOD1 4294967087.
#define MRG32K3A_MOD2 4294944443.

/* Constants used in generation */

#define MRG32K3A_A12  1403580.
#define MRG32K3A_A13N 810728.
#define MRG32K3A_A21  527612.
#define MRG32K3A_A23N 1370589.
#define MRG32K3A_NORM 2.328306549295728e-10
//
// #define MRG32K3A_BITS_NORM ((double)((POW32_DOUBLE-1.0)/MOD1))
//  above constant, used verbatim, rounds differently on some host systems.
#define MRG32K3A_BITS_NORM 1.000000048662


/* Constants for address manipulation */

#define MRG32K3A_SKIPUNITS_DOUBLES   (sizeof(struct sMRG32k3aSkipUnits)/sizeof(double))
#define MRG32K3A_SKIPSUBSEQ_DOUBLES  (sizeof(struct sMRG32k3aSkipSubSeq)/sizeof(double))
#define MRG32K3A_SKIPSEQ_DOUBLES     (sizeof(struct sMRG32k3aSkipSeq)/sizeof(double))
/** \endcond */




/**
 * CURAND MRG32K3A state 
 */
struct curandStateMRG32k3a;

/* Implementation details not in reference documentation */
struct curandStateMRG32k3a {
    double s1[3];
    double s2[3];
    int boxmuller_flag;
    int boxmuller_flag_double;
    float boxmuller_extra;
    double boxmuller_extra_double;
};

/*
 * CURAND MRG32K3A state 
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct curandStateMRG32k3a curandStateMRG32k3a_t;
/** \endcond */

/*
 * Taken from curand_philox4x32_x.h
 */
struct curandStatePhilox4_32_10 {
        unsigned int ctr;
        unsigned int output;
        unsigned short key;
        unsigned int STATE;
        int boxmuller_flag;
        int boxmuller_flag_double;
        float boxmuller_extra;
        double boxmuller_extra_double;
};

typedef struct curandStatePhilox4_32_10 curandStatePhilox4_32_10_t;

/* SOBOL QRNG */
/**
 * CURAND Sobol32 state 
 */
struct curandStateSobol32;

/* Implementation details not in reference documentation */
struct curandStateSobol32 {
    unsigned int i, x, c;
    unsigned int direction_vectors[32];
};

/*
 * CURAND Sobol32 state 
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct curandStateSobol32 curandStateSobol32_t;
/** \endcond */

/**
 * CURAND Scrambled Sobol32 state 
 */
struct curandStateScrambledSobol32;

/* Implementation details not in reference documentation */
struct curandStateScrambledSobol32 {
    unsigned int i, x, c;
    unsigned int direction_vectors[32];
};

/*
 * CURAND Scrambled Sobol32 state 
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct curandStateScrambledSobol32 curandStateScrambledSobol32_t;
/** \endcond */

/**
 * CURAND Sobol64 state 
 */
struct curandStateSobol64;

/* Implementation details not in reference documentation */
struct curandStateSobol64 {
    unsigned long long i, x, c;
    unsigned long long direction_vectors[64];
};

/*
 * CURAND Sobol64 state 
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct curandStateSobol64 curandStateSobol64_t;
/** \endcond */

/**
 * CURAND Scrambled Sobol64 state 
 */
struct curandStateScrambledSobol64;

/* Implementation details not in reference documentation */
struct curandStateScrambledSobol64 {
    unsigned long long i, x, c;
    unsigned long long direction_vectors[64];
};

/*
 * CURAND Scrambled Sobol64 state 
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct curandStateScrambledSobol64 curandStateScrambledSobol64_t;
/** \endcond */

/*
 * Default RNG
 */
/** \cond UNHIDE_TYPEDEFS */
typedef struct curandStateXORWOW curandState_t;
typedef struct curandStateXORWOW curandState;
/** \endcond */


#if defined(__cplusplus)
extern "C" {
#endif

/* These default to XORWOW */
#define curand_init                 __pgicudalib_curandInitXORWOW
#define curand_log_normal           __pgicudalib_curandLogNormalXORWOW
#define curand_normal               __pgicudalib_curandNormalXORWOW
#define curand_uniform              __pgicudalib_curandUniformXORWOW
#define curand_log_normal_double    __pgicudalib_curandLogNormalDoubleXORWOW
#define curand_normal_double        __pgicudalib_curandNormalDoubleXORWOW
#define curand_uniform_double       __pgicudalib_curandUniformDoubleXORWOW

/* ---------------------------------------------------------------------- */

#define curandInitXORWOW            __pgicudalib_curandInitXORWOW
#define curandGetXORWOW             __pgicudalib_curandGetXORWOW
#define curandNormalXORWOW          __pgicudalib_curandNormalXORWOW
#define curandNormalDoubleXORWOW    __pgicudalib_curandNormalDoubleXORWOW
#define curandLogNormalXORWOW       __pgicudalib_curandLogNormalXORWOW
#define curandLogNormalDoubleXORWOW __pgicudalib_curandLogNormalDoubleXORWOW
#define curandUniformXORWOW         __pgicudalib_curandUniformXORWOW
#define curandUniformDoubleXORWOW   __pgicudalib_curandUniformDoubleXORWOW

#pragma acc routine(__pgicudalib_curandInitXORWOW) seq
extern void __pgicudalib_curandInitXORWOW(unsigned long long, unsigned long long, unsigned long long, curandStateXORWOW_t *);
#pragma acc routine(__pgicudalib_curandGetXORWOW) seq
extern int __pgicudalib_curandGetXORWOW(curandStateXORWOW_t *);
#pragma acc routine(__pgicudalib_curandNormalXORWOW) seq
extern float __pgicudalib_curandNormalXORWOW(curandStateXORWOW_t *);
#pragma acc routine(__pgicudalib_curandNormalDoubleXORWOW) seq
extern double __pgicudalib_curandNormalDoubleXORWOW(curandStateXORWOW_t *);
#pragma acc routine(__pgicudalib_curandLogNormalXORWOW) seq
extern float __pgicudalib_curandLogNormalXORWOW(curandStateXORWOW_t *);
#pragma acc routine(__pgicudalib_curandLogNormalDoubleXORWOW) seq
extern double __pgicudalib_curandLogNormalDoubleXORWOW(curandStateXORWOW_t *);
#pragma acc routine(__pgicudalib_curandUniformXORWOW) seq
extern float __pgicudalib_curandUniformXORWOW(curandStateXORWOW_t *);
#pragma acc routine(__pgicudalib_curandUniformDoubleXORWOW) seq
extern double __pgicudalib_curandUniformDoubleXORWOW(curandStateXORWOW_t *);

/* ---------------------------------------------------------------------- */

#define curandInitMRG32k3a            __pgicudalib_curandInitMRG32k3a
#define curandGetMRG32k3a             __pgicudalib_curandGetMRG32k3a
#define curandNormalMRG32k3a          __pgicudalib_curandNormalMRG32k3a
#define curandNormalDoubleMRG32k3a    __pgicudalib_curandNormalDoubleMRG32k3a
#define curandLogNormalMRG32k3a       __pgicudalib_curandLogNormalMRG32k3a
#define curandLogNormalDoubleMRG32k3a __pgicudalib_curandLogNormalDoubleMRG32k3a
#define curandUniformMRG32k3a         __pgicudalib_curandUniformMRG32k3a
#define curandUniformDoubleMRG32k3a   __pgicudalib_curandUniformDoubleMRG32k3a

#pragma acc routine(__pgicudalib_curandInitMRG32k3a) seq
extern int __pgicudalib_curandInitMRG32k3a(unsigned long long, unsigned long long, unsigned long long, curandStateMRG32k3a_t *);
#pragma acc routine(__pgicudalib_curandGetMRG32k3a) seq
extern int __pgicudalib_curandGetMRG32k3a(curandStateMRG32k3a_t *);
#pragma acc routine(__pgicudalib_curandNormalMRG32k3a) seq
extern float __pgicudalib_curandNormalMRG32k3a(curandStateMRG32k3a_t *);
#pragma acc routine(__pgicudalib_curandNormalDoubleMRG32k3a) seq
extern double __pgicudalib_curandNormalDoubleMRG32k3a(curandStateMRG32k3a_t *);
#pragma acc routine(__pgicudalib_curandLogNormalMRG32k3a) seq
extern float __pgicudalib_curandLogNormalMRG32k3a(curandStateMRG32k3a_t *);
#pragma acc routine(__pgicudalib_curandLogNormalDoubleMRG32k3a) seq
extern double __pgicudalib_curandLogNormalDoubleMRG32k3a(curandStateMRG32k3a_t *);
#pragma acc routine(__pgicudalib_curandUniformMRG32k3a) seq
extern float __pgicudalib_curandUniformMRG32k3a(curandStateMRG32k3a_t *);
#pragma acc routine(__pgicudalib_curandUniformDoubleMRG32k3a) seq
extern double __pgicudalib_curandUniformDoubleMRG32k3a(curandStateMRG32k3a_t *);

/* ---------------------------------------------------------------------- */

#define curandInitPhilox4_32_10            __pgicudalib_curandInitPhilox4_32_10
#define curandGetPhilox4_32_10             __pgicudalib_curandGetPhilox4_32_10
#define curandNormalPhilox4_32_10          __pgicudalib_curandNormalPhilox4_32_10
#define curandNormalDoublePhilox4_32_10    __pgicudalib_curandNormalDoublePhilox4_32_10
#define curandLogNormalPhilox4_32_10       __pgicudalib_curandLogNormalPhilox4_32_10
#define curandLogNormalDoublePhilox4_32_10 __pgicudalib_curandLogNormalDoublePhilox4_32_10
#define curandUniformPhilox4_32_10         __pgicudalib_curandUniformPhilox4_32_10
#define curandUniformDoublePhilox4_32_10   __pgicudalib_curandUniformDoublePhilox4_32_10

#pragma acc routine(__pgicudalib_curandInitPhilox4_32_10) seq
extern int __pgicudalib_curandInitPhilox4_32_10(unsigned long long, unsigned long long, unsigned long long, curandStatePhilox4_32_10_t *);
#pragma acc routine(__pgicudalib_curandGetPhilox4_32_10) seq
extern int __pgicudalib_curandGetPhilox4_32_10(curandStatePhilox4_32_10_t *);
#pragma acc routine(__pgicudalib_curandNormalPhilox4_32_10) seq
extern float __pgicudalib_curandNormalPhilox4_32_10(curandStatePhilox4_32_10_t *);
#pragma acc routine(__pgicudalib_curandNormalDoublePhilox4_32_10) seq
extern double __pgicudalib_curandNormalDoublePhilox4_32_10(curandStatePhilox4_32_10_t *);
#pragma acc routine(__pgicudalib_curandLogNormalPhilox4_32_10) seq
extern float __pgicudalib_curandLogNormalPhilox4_32_10(curandStatePhilox4_32_10_t *);
#pragma acc routine(__pgicudalib_curandLogNormalDoublePhilox4_32_10) seq
extern double __pgicudalib_curandLogNormalDoublePhilox4_32_10(curandStatePhilox4_32_10_t *);
#pragma acc routine(__pgicudalib_curandUniformPhilox4_32_10) seq
extern float __pgicudalib_curandUniformPhilox4_32_10(curandStatePhilox4_32_10_t *);
#pragma acc routine(__pgicudalib_curandUniformDoublePhilox4_32_10) seq
extern double __pgicudalib_curandUniformDoublePhilox4_32_10(curandStatePhilox4_32_10_t *);

/* ---------------------------------------------------------------------- */

#define curandInitSobol32            __pgicudalib_curandInitSobol32
#define curandGetSobol32             __pgicudalib_curandGetSobol32
#define curandNormalSobol32          __pgicudalib_curandNormalSobol32
#define curandNormalDoubleSobol32    __pgicudalib_curandNormalDoubleSobol32
#define curandLogNormalSobol32       __pgicudalib_curandLogNormalSobol32
#define curandLogNormalDoubleSobol32 __pgicudalib_curandLogNormalDoubleSobol32
#define curandUniformSobol32         __pgicudalib_curandUniformSobol32
#define curandUniformDoubleSobol32   __pgicudalib_curandUniformDoubleSobol32

#pragma acc routine(__pgicudalib_curandInitSobol32) seq
extern int __pgicudalib_curandInitSobol32(unsigned long long, unsigned long long, unsigned long long, curandStateSobol32_t *);
#pragma acc routine(__pgicudalib_curandGetSobol32) seq
extern int __pgicudalib_curandGetSobol32(curandStateSobol32_t *);
#pragma acc routine(__pgicudalib_curandNormalSobol32) seq
extern float __pgicudalib_curandNormalSobol32(curandStateSobol32_t *);
#pragma acc routine(__pgicudalib_curandNormalDoubleSobol32) seq
extern double __pgicudalib_curandNormalDoubleSobol32(curandStateSobol32_t *);
#pragma acc routine(__pgicudalib_curandLogNormalSobol32) seq
extern float __pgicudalib_curandLogNormalSobol32(curandStateSobol32_t *);
#pragma acc routine(__pgicudalib_curandLogNormalDoubleSobol32) seq
extern double __pgicudalib_curandLogNormalDoubleSobol32(curandStateSobol32_t *);
#pragma acc routine(__pgicudalib_curandUniformSobol32) seq
extern float __pgicudalib_curandUniformSobol32(curandStateSobol32_t *);
#pragma acc routine(__pgicudalib_curandUniformDoubleSobol32) seq
extern double __pgicudalib_curandUniformDoubleSobol32(curandStateSobol32_t *);

/* ---------------------------------------------------------------------- */

#define curandInitScrambledSobol32            __pgicudalib_curandInitScrambledSobol32
#define curandGetScrambledSobol32             __pgicudalib_curandGetScrambledSobol32
#define curandNormalScrambledSobol32          __pgicudalib_curandNormalScrambledSobol32
#define curandNormalDoubleScrambledSobol32    __pgicudalib_curandNormalDoubleScrambledSobol32
#define curandLogNormalScrambledSobol32       __pgicudalib_curandLogNormalScrambledSobol32
#define curandLogNormalDoubleScrambledSobol32 __pgicudalib_curandLogNormalDoubleScrambledSobol32
#define curandUniformScrambledSobol32         __pgicudalib_curandUniformScrambledSobol32
#define curandUniformDoubleScrambledSobol32   __pgicudalib_curandUniformDoubleScrambledSobol32

#pragma acc routine(__pgicudalib_curandInitScrambledSobol32) seq
extern int __pgicudalib_curandInitScrambledSobol32(unsigned long long, unsigned long long, unsigned long long, curandStateScrambledSobol32_t *);
#pragma acc routine(__pgicudalib_curandGetScrambledSobol32) seq
extern int __pgicudalib_curandGetScrambledSobol32(curandStateScrambledSobol32_t *);
#pragma acc routine(__pgicudalib_curandNormalScrambledSobol32) seq
extern float __pgicudalib_curandNormalScrambledSobol32(curandStateScrambledSobol32_t *);
#pragma acc routine(__pgicudalib_curandNormalDoubleScrambledSobol32) seq
extern double __pgicudalib_curandNormalDoubleScrambledSobol32(curandStateScrambledSobol32_t *);
#pragma acc routine(__pgicudalib_curandLogNormalScrambledSobol32) seq
extern float __pgicudalib_curandLogNormalScrambledSobol32(curandStateScrambledSobol32_t *);
#pragma acc routine(__pgicudalib_curandLogNormalDoubleScrambledSobol32) seq
extern double __pgicudalib_curandLogNormalDoubleScrambledSobol32(curandStateScrambledSobol32_t *);
#pragma acc routine(__pgicudalib_curandUniformScrambledSobol32) seq
extern float __pgicudalib_curandUniformScrambledSobol32(curandStateScrambledSobol32_t *);
#pragma acc routine(__pgicudalib_curandUniformDoubleScrambledSobol32) seq
extern double __pgicudalib_curandUniformDoubleScrambledSobol32(curandStateScrambledSobol32_t *);

/* ---------------------------------------------------------------------- */

#define curandInitSobol64            __pgicudalib_curandInitSobol64
#define curandGetSobol64             __pgicudalib_curandGetSobol64
#define curandNormalSobol64          __pgicudalib_curandNormalSobol64
#define curandNormalDoubleSobol64    __pgicudalib_curandNormalDoubleSobol64
#define curandLogNormalSobol64       __pgicudalib_curandLogNormalSobol64
#define curandLogNormalDoubleSobol64 __pgicudalib_curandLogNormalDoubleSobol64
#define curandUniformSobol64         __pgicudalib_curandUniformSobol64
#define curandUniformDoubleSobol64   __pgicudalib_curandUniformDoubleSobol64

#pragma acc routine(__pgicudalib_curandInitSobol64) seq
extern int __pgicudalib_curandInitSobol64(unsigned long long, unsigned long long, unsigned long long, curandStateSobol64_t *);
#pragma acc routine(__pgicudalib_curandGetSobol64) seq
extern int __pgicudalib_curandGetSobol64(curandStateSobol64_t *);
#pragma acc routine(__pgicudalib_curandNormalSobol64) seq
extern float __pgicudalib_curandNormalSobol64(curandStateSobol64_t *);
#pragma acc routine(__pgicudalib_curandNormalDoubleSobol64) seq
extern double __pgicudalib_curandNormalDoubleSobol64(curandStateSobol64_t *);
#pragma acc routine(__pgicudalib_curandLogNormalSobol64) seq
extern float __pgicudalib_curandLogNormalSobol64(curandStateSobol64_t *);
#pragma acc routine(__pgicudalib_curandLogNormalDoubleSobol64) seq
extern double __pgicudalib_curandLogNormalDoubleSobol64(curandStateSobol64_t *);
#pragma acc routine(__pgicudalib_curandUniformSobol64) seq
extern float __pgicudalib_curandUniformSobol64(curandStateSobol64_t *);
#pragma acc routine(__pgicudalib_curandUniformDoubleSobol64) seq
extern double __pgicudalib_curandUniformDoubleSobol64(curandStateSobol64_t *);

/* ---------------------------------------------------------------------- */

#define curandInitScrambledSobol64            __pgicudalib_curandInitScrambledSobol64
#define curandGetScrambledSobol64             __pgicudalib_curandGetScrambledSobol64
#define curandNormalScrambledSobol64          __pgicudalib_curandNormalScrambledSobol64
#define curandNormalDoubleScrambledSobol64    __pgicudalib_curandNormalDoubleScrambledSobol64
#define curandLogNormalScrambledSobol64       __pgicudalib_curandLogNormalScrambledSobol64
#define curandLogNormalDoubleScrambledSobol64 __pgicudalib_curandLogNormalDoubleScrambledSobol64
#define curandUniformScrambledSobol64         __pgicudalib_curandUniformScrambledSobol64
#define curandUniformDoubleScrambledSobol64   __pgicudalib_curandUniformDoubleScrambledSobol64

#pragma acc routine(__pgicudalib_curandInitScrambledSobol64) seq
extern int __pgicudalib_curandInitScrambledSobol64(unsigned long long, unsigned long long, unsigned long long, curandStateScrambledSobol64_t *);
#pragma acc routine(__pgicudalib_curandGetScrambledSobol64) seq
extern int __pgicudalib_curandGetScrambledSobol64(curandStateScrambledSobol64_t *);
#pragma acc routine(__pgicudalib_curandNormalScrambledSobol64) seq
extern float __pgicudalib_curandNormalScrambledSobol64(curandStateScrambledSobol64_t *);
#pragma acc routine(__pgicudalib_curandNormalDoubleScrambledSobol64) seq
extern double __pgicudalib_curandNormalDoubleScrambledSobol64(curandStateScrambledSobol64_t *);
#pragma acc routine(__pgicudalib_curandLogNormalScrambledSobol64) seq
extern float __pgicudalib_curandLogNormalScrambledSobol64(curandStateScrambledSobol64_t *);
#pragma acc routine(__pgicudalib_curandLogNormalDoubleScrambledSobol64) seq
extern double __pgicudalib_curandLogNormalDoubleScrambledSobol64(curandStateScrambledSobol64_t *);
#pragma acc routine(__pgicudalib_curandUniformScrambledSobol64) seq
extern float __pgicudalib_curandUniformScrambledSobol64(curandStateScrambledSobol64_t *);
#pragma acc routine(__pgicudalib_curandUniformDoubleScrambledSobol64) seq
extern double __pgicudalib_curandUniformDoubleScrambledSobol64(curandStateScrambledSobol64_t *);

#if defined(__cplusplus)
}
#endif


#endif // !defined(CURAND_KERNEL_H_)
