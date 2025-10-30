#ifndef ARENA_H
#define ARENA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Cross-platform arena allocator for efficient memory management
 * Compatible with both Windows and Linux
 *
 * An arena allocator uses a linked list of memory chunks. When a chunk fills up,
 * a new larger chunk is allocated and linked. This eliminates the overhead of
 * individual malloc/free calls and improves cache locality.
 *
 * Features:
 * - Chain of memory chunks (no realloc, no pointer invalidation!)
 * - O(1) allocation time
 * - No per-allocation overhead
 * - Bulk deallocation (free entire chain at once)
 * - Cross-platform (Windows/Linux/macOS)
 * - **SAFE**: Existing pointers never invalidated by new allocations
 */

/* Opaque pointer to internal chunk structure */
typedef struct ArenaChunk ArenaChunk;

typedef struct {
    ArenaChunk *first;   /* First chunk in chain */
    ArenaChunk *current; /* Current chunk for allocation */
    size_t chunk_size;   /* Size of new chunks to allocate */
} Arena;

/**
 * Initialize an arena with a given initial size
 * Exits on allocation failure
 *
 * @param arena Pointer to arena structure
 * @param initial_size Initial capacity in bytes
 */
void arena_init(Arena *arena, size_t initial_size);

/**
 * Reserve capacity in the arena
 * Pre-allocates chunks to minimize allocations during parsing
 * Does NOT invalidate existing pointers
 * Exits on allocation failure
 *
 * @param arena Pointer to arena structure
 * @param total_needed Total capacity to reserve in bytes
 */
void arena_reserve(Arena *arena, size_t total_needed);

/**
 * Allocate memory from the arena
 * Automatically allocates new chunk if needed (safe - never invalidates pointers)
 * Exits on allocation failure
 *
 * @param arena Pointer to arena structure
 * @param size Number of bytes to allocate
 * @return Pointer to allocated memory (never NULL, never invalidated by future allocations)
 */
char *arena_alloc(Arena *arena, size_t size);

/**
 * Duplicate a string in the arena
 * Exits on allocation failure or if str is NULL
 *
 * @param arena Pointer to arena structure
 * @param str String to duplicate (must not be NULL)
 * @return Pointer to duplicated string in arena (never NULL)
 */
char *arena_strdup(Arena *arena, const char *str);

/**
 * Duplicate up to n characters of a string in the arena
 * Exits on allocation failure or if str is NULL
 *
 * @param arena Pointer to arena structure
 * @param str String to duplicate (must not be NULL)
 * @param n Maximum number of characters to copy
 * @return Pointer to duplicated string in arena (never NULL, null-terminated)
 */
char *arena_strndup(Arena *arena, const char *str, size_t n);

/**
 * Reset the arena (mark all memory as available without freeing)
 * Existing pointers into the arena become invalid
 *
 * @param arena Pointer to arena structure
 */
void arena_reset(Arena *arena);

/**
 * Free all memory associated with the arena
 *
 * @param arena Pointer to arena structure
 */
void arena_free(Arena *arena);

/**
 * Get the current memory usage of the arena
 * Walks the chunk chain to sum up used bytes
 *
 * @param arena Pointer to arena structure
 * @return Number of bytes currently used
 */
size_t arena_get_used(const Arena *arena);

/**
 * Get the total capacity of the arena
 * Walks the chunk chain to sum up capacities
 *
 * @param arena Pointer to arena structure
 * @return Total capacity in bytes
 */
size_t arena_get_capacity(const Arena *arena);

#ifdef __cplusplus
}
#endif

#endif /* ARENA_H */
