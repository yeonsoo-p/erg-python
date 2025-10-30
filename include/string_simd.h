#ifndef STRING_SIMD_H
#define STRING_SIMD_H

#include <stddef.h>

/**
 * SIMD-optimized string operations using AVX2
 * These provide faster alternatives to standard C library functions
 * for use in performance-critical code paths
 */

/**
 * SIMD-optimized strlen
 * Processes 32 bytes at a time using AVX2
 *
 * @param str The string to measure
 * @return Length of the string (not including null terminator)
 */
size_t strlen_simd(const char *str);

/**
 * SIMD-optimized memcpy (aligned)
 * Uses AVX2 for 32-byte chunks, with alignment optimization
 * Best for destinations that are 32-byte aligned
 *
 * @param dest Destination buffer
 * @param src Source buffer
 * @param n Number of bytes to copy
 * @return dest pointer
 */
void *memcpy_simd(void *dest, const void *src, size_t n);

/**
 * SIMD-optimized memcpy (unaligned)
 * Uses AVX2 for 32-byte chunks without alignment requirements
 * Best for arena allocations which may not be aligned
 *
 * @param dest Destination buffer (can be unaligned)
 * @param src Source buffer (can be unaligned)
 * @param n Number of bytes to copy
 * @return dest pointer
 */
void *memcpy_simd_unaligned(void *dest, const void *src, size_t n);

#endif /* STRING_SIMD_H */
