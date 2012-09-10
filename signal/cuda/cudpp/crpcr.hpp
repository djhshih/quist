// Adapted from cudpp library

// -------------------------------------------------------------
// CUDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: 4400 $
// $Date: 2008-08-04 10:58:14 -0700 (Mon, 04 Aug 2008) $
// ------------------------------------------------------------- 
// This source code is distributed under the terms of license.txt 
// in the root directory of this source distribution.
// ------------------------------------------------------------- 

/**
 * @file
 * tridiagonal_app.cu
 *
 * @brief CUDPP application-level tridiagonal solver routines
 */

/** \addtogroup cudpp_app
  * @{
  */
/** @name Tridiagonal functions
 * @{
 */

#include <numeric/math.hpp>

#include "../util.hpp"
#include "tridiag_crpcr_kernel.hpp"

template <typename T>
inline unsigned int crpcrSharedSize(unsigned int systemSizeOriginal)
{
    const unsigned int systemSize = math::pow2ceil(systemSizeOriginal);
    const unsigned int restSystemSize = systemSize/2;
    return (systemSize + 1 + restSystemSize) * 5 * sizeof(T);
}

/**
 * @brief Hybrid CR-PCR solver (CRPCR)
 *
 * This is a wrapper function for the GPU CR-PCR kernel.
 *
 * @param[out] d_x Solution vector
 * @param[in] d_a Lower diagonal
 * @param[in] d_b Main diagonal
 * @param[in] d_c Upper diagonal
 * @param[in] d_d Right hand side
 * @param[in] systemSizeOriginal The size of the linear system
 * @param[in] numSystems The number of systems to be solved
 */
template <typename T>
void crpcr(T *d_a, 
           T *d_b, 
           T *d_c, 
           T *d_d, 
           T *d_x, 
           unsigned int systemSizeOriginal, 
           unsigned int numSystems)
{
    const unsigned int systemSize = math::pow2ceil(systemSizeOriginal);
    const unsigned int num_threads_block = systemSize/2;
    const unsigned int restSystemSize = systemSize/2;
    const unsigned int iterations = math::log2(restSystemSize/2);
  
    // setup execution parameters
    dim3  grid(numSystems, 1, 1);
    dim3  threads(num_threads_block, 1, 1);
    const unsigned int smemSize = crpcrSharedSize<T>(systemSizeOriginal);

    crpcrKernel<<< grid, threads, smemSize>>>(d_a, 
                                              d_b, 
                                              d_c, 
                                              d_d, 
                                              d_x, 
                                              systemSizeOriginal,
                                              iterations);

    CUDA_CHECK_ERROR("crpcr");
}
