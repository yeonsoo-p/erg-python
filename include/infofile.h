#ifndef INFOFILE_H
#define INFOFILE_H

#include <stddef.h>
#include <arena.h>

/**
 * Dual arena allocator for optimized cache locality
 * Keys (hot) and values (cold) stored separately
 */
typedef struct {
    Arena key_arena;    /* Hot data - accessed during lookups */
    Arena value_arena;  /* Cold data - accessed only when found */
} DualArena;

/**
 * Represents a key-value pair
 */
typedef struct {
    const char *key;    /* Points into key_arena */
    const char *value;  /* Points into value_arena */
} InfoFileEntry;

/**
 * Represents a parsed info file
 * Uses zero-copy parsing, SIMD whitespace trimming,
 * SIMD character class detection, and dual-arena layout
 */
typedef struct {
    InfoFileEntry *entries;  /* Array of entries */
    size_t count;            /* Number of entries */
    size_t capacity;         /* Allocated capacity */
    DualArena arena;         /* Dual arena for keys/values */
} InfoFile;

/**
 * Initialize an InfoFile structure
 */
void infofile_init(InfoFile *info);

/**
 * Parse an info file from a file path
 * Exits on error with descriptive message
 */
void infofile_parse_file(const char *filename, InfoFile *info);

/**
 * Parse an info file from a string buffer
 * Exits on error with descriptive message
 */
void infofile_parse_string(const char *data, size_t len, InfoFile *info);

/**
 * Get a value by key (returns NULL if not found)
 */
const char *infofile_get(const InfoFile *info, const char *key);

/**
 * Free all memory associated with an InfoFile structure
 */
void infofile_free(InfoFile *info);

#endif /* INFOFILE_H */
