#ifndef UTILS_H
#define UTILS_H


#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>


#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef __cplusplus
#include <functional>
#include <string_view>
extern "C" {
#endif
/* ============================================================================
 * Unified Error Type
 * ============================================================================ */

/**
 * Unified error type that can hold any error code
 * All error enums use distinct ranges:
 * - 0: Success (all types)
 * - 100-199: MMapError
 * - 200-299: InfoFileError
 * - 300-399: ERGError
 */
typedef int Error;

/* ============================================================================
 * Memory Mapping Error Codes (100-199 range)
 * ============================================================================ */

typedef enum {
    MMAP_OK                    = 0,   /* Success */
    MMAP_FILE_NOT_FOUND        = 100, /* Failed to open file */
    MMAP_FILE_SIZE_FAILED      = 101, /* Failed to get file size */
    MMAP_CREATE_MAPPING_FAILED = 102, /* Failed to create file mapping (Windows) */
    MMAP_MAP_VIEW_FAILED       = 103  /* Failed to map view of file */
} MMapError;

/* Get human-readable error message for MMapError */
const char* mmap_error_string(MMapError error);

/* ============================================================================
 * Memory-Mapped File Utilities (C API)
 * ============================================================================ */

/**
 * Memory-mapped file handle for cross-platform file mapping
 * Pure C structure - accessible from both C and C++ code
 */
typedef struct {
    void*  data; /* Pointer to mapped memory */
    size_t size; /* Size of mapped region */

#ifdef _WIN32
    HANDLE file_handle;
    HANDLE mapping_handle;
#else
    int file_descriptor;
#endif
} MappedFile;

/**
 * Initialize a MappedFile structure to default values
 */
static inline void mmap_init(MappedFile* mapped) {
    mapped->data = NULL;
    mapped->size = 0;
#ifdef _WIN32
    mapped->file_handle = INVALID_HANDLE_VALUE;
    mapped->mapping_handle = NULL;
#else
    mapped->file_descriptor = -1;
#endif
}

#ifdef __cplusplus
}

/* Transparent hash for heterogeneous string lookup in unordered_map */
struct StringHash {
    using is_transparent = void;
    size_t operator()(std::string_view sv) const {
        return std::hash<std::string_view>{}(sv);
    }
};

/* Transparent equality for heterogeneous string lookup in unordered_map */
struct StringEqual {
    using is_transparent = void;
    bool operator()(std::string_view lhs, std::string_view rhs) const {
        return lhs == rhs;
    }
};

/* Transparent comparator for heterogeneous string lookup in ordered map */
struct StringLess {
    using is_transparent = void;
    bool operator()(std::string_view lhs, std::string_view rhs) const {
        return lhs < rhs;
    }
};

/**
 * Open and memory-map a file for reading
 * Returns error code on failure
 *
 * @param filename Path to file to map
 * @param mapped Output MappedFile structure
 * @return MMAP_OK on success, error code on failure
 */
inline MMapError mmap_open(const char* filename, MappedFile* mapped) {
#ifdef _WIN32
    /* Windows memory mapping */
    mapped->file_handle = CreateFileA(
        filename,
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL);

    if (mapped->file_handle == INVALID_HANDLE_VALUE) {
        return MMAP_FILE_NOT_FOUND;
    }

    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(mapped->file_handle, &file_size)) {
        CloseHandle(mapped->file_handle);
        return MMAP_FILE_SIZE_FAILED;
    }

    mapped->size = (size_t)file_size.QuadPart;

    mapped->mapping_handle = CreateFileMappingA(
        mapped->file_handle,
        NULL,
        PAGE_READONLY,
        0,
        0, /* Map entire file */
        NULL);

    if (!mapped->mapping_handle) {
        CloseHandle(mapped->file_handle);
        return MMAP_CREATE_MAPPING_FAILED;
    }

    mapped->data = MapViewOfFile(
        mapped->mapping_handle,
        FILE_MAP_READ,
        0,
        0,
        0 /* Map entire file */
    );

    if (!mapped->data) {
        CloseHandle(mapped->mapping_handle);
        CloseHandle(mapped->file_handle);
        return MMAP_MAP_VIEW_FAILED;
    }

#else
    /* POSIX memory mapping */
    mapped->file_descriptor = open(filename, O_RDONLY);
    if (mapped->file_descriptor == -1) {
        return MMAP_FILE_NOT_FOUND;
    }

    struct stat st;
    if (fstat(mapped->file_descriptor, &st) == -1) {
        close(mapped->file_descriptor);
        return MMAP_FILE_SIZE_FAILED;
    }

    mapped->size = (size_t)st.st_size;

    mapped->data = mmap(
        NULL,
        mapped->size,
        PROT_READ,
        MAP_PRIVATE,
        mapped->file_descriptor,
        0);

    if (mapped->data == MAP_FAILED) {
        close(mapped->file_descriptor);
        return MMAP_MAP_VIEW_FAILED;
    }
#endif

    return MMAP_OK;
}

/**
 * Close and unmap a memory-mapped file
 *
 * @param mapped MappedFile structure to close
 */
inline void mmap_close(MappedFile* mapped) {
    if (mapped->data) {
#ifdef _WIN32
        UnmapViewOfFile(mapped->data);
        if (mapped->mapping_handle) {
            CloseHandle(mapped->mapping_handle);
        }
        if (mapped->file_handle != INVALID_HANDLE_VALUE) {
            CloseHandle(mapped->file_handle);
        }
#else
        munmap(mapped->data, mapped->size);
        if (mapped->file_descriptor != -1) {
            close(mapped->file_descriptor);
        }
#endif
        mapped->data = NULL;
        mapped->size = 0;
    }
}

#endif

#endif /* UTILS_H */
