#ifndef INFOFILE_H
#define INFOFILE_H

#include <stddef.h>
#include <utils.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * InfoFile Error Codes (200-299 range)
 */
typedef enum {
    INFOFILE_OK = 0,          /* Success */
    INFOFILE_MMAP_ERROR = 200, /* Memory mapping failed - check infofile_get_mmap_error() */
    INFOFILE_PARSE_ERROR = 201 /* Parsing failed (reserved for future use) */
} InfoFileError;

/**
 * Represents a parsed info file
 * Uses C++ hash table for O(1) lookups
 */
typedef struct {
    void* impl; /* Internal C++ implementation */
} InfoFile;

/**
 * Get human-readable error message for InfoFileError
 */
const char* infofile_error_string(InfoFileError error);

/**
 * Get the underlying error code
 * @param info Pointer to InfoFile structure
 * @return Error code (MMapError if INFOFILE_MMAP_ERROR, 0 if OK)
 */
Error infofile_get_error(const InfoFile* info);

/**
 * Initialize an InfoFile structure
 */
void infofile_init(InfoFile* info);

/**
 * Parse an info file from a file path
 * Returns error code on failure
 *
 * @param filename Path to info file
 * @param info Pointer to InfoFile structure
 * @return INFOFILE_OK on success, error code on failure
 */
InfoFileError infofile_parse_file(const char* filename, InfoFile* info);

/**
 * Parse an info file from a string buffer
 * This function does not fail - malformed lines are skipped
 *
 * @param data Pointer to string buffer
 * @param len Length of buffer
 * @param info Pointer to InfoFile structure
 */
void infofile_parse_string(const char* data, size_t len, InfoFile* info);

/**
 * Get a value by key (returns NULL if not found)
 */
const char* infofile_get(const InfoFile* info, const char* key);

/**
 * Get the number of entries in the info file
 */
size_t infofile_count(const InfoFile* info);

/**
 * Free all memory associated with an InfoFile structure
 */
void infofile_free(InfoFile* info);

#ifdef __cplusplus
}
#endif

#endif /* INFOFILE_H */
