#ifndef ERG_H
#define ERG_H

#include <infofile.h>
#include <stddef.h>
#include <stdint.h>
#include <utils.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ERG_HEADER_SIZE 16

/**
 * ERG (CarMaker binary results) file parser
 *
 * Reads binary ERG files containing simulation results from CarMaker.
 * Each ERG file has a companion .erg.info file with metadata.
 */

/**
 * Data type mapping from CarMaker to C types
 */
typedef enum {
    ERG_FLOAT,     // 4 bytes
    ERG_DOUBLE,    // 8 bytes
    ERG_LONGLONG,  // 8 bytes signed
    ERG_ULONGLONG, // 8 bytes unsigned
    ERG_INT,       // 4 bytes signed
    ERG_UINT,      // 4 bytes unsigned
    ERG_SHORT,     // 2 bytes signed
    ERG_USHORT,    // 2 bytes unsigned
    ERG_CHAR,      // 1 byte signed
    ERG_UCHAR,     // 1 byte unsigned
    ERG_1BYTE,     // 1 byte
    ERG_2BYTES,    // 2 bytes
    ERG_3BYTES,    // 3 bytes
    ERG_4BYTES,    // 4 bytes
    ERG_5BYTES,    // 5 bytes
    ERG_6BYTES,    // 6 bytes
    ERG_7BYTES,    // 7 bytes
    ERG_8BYTES,    // 8 bytes
    ERG_UNKNOWN
} ERGDataType;

/**
 * ERG Error Codes (300-399 range)
 */
typedef enum {
    ERG_OK                     = 0,   /* Success */
    ERG_INFOFILE_ERROR         = 300, /* Info file parsing failed - check erg_get_infofile_error() */
    ERG_MMAP_ERROR             = 301, /* Memory mapping failed - check erg_get_mmap_error() */
    ERG_INVALID_FORMAT         = 302, /* Missing required metadata (e.g., File.ByteOrder) */
    ERG_UNSUPPORTED_BYTE_ORDER = 303, /* Big-endian files not supported */
    ERG_UNKNOWN_DATA_TYPE      = 304, /* Signal has unrecognized data type */
    ERG_NO_SIGNALS_FOUND       = 305, /* No signals in ERG file */
    ERG_FILE_TOO_SMALL         = 306, /* File smaller than minimum header size */
    ERG_DATA_CORRUPTION        = 307, /* Data size / row size mismatch */
    ERG_ALLOCATION_FAILED      = 308, /* Memory allocation failed */
    ERG_INTERNAL_ERROR         = 309  /* Unexpected internal state */
} ERGError;

/**
 * Metadata for a single signal/channel
 */
typedef struct {
    const char* name;        /* Signal name (e.g., "Time", "Car.v") */
    int         index;       /* Signal index in the file */
    ERGDataType type;        /* Data type */
    size_t      type_size;   /* Size in bytes */
    const char* unit;        /* Unit string (e.g., "m/s", "s") */
    double      factor;      /* Scaling factor */
    double      offset;      /* Scaling offset */
    size_t      data_offset; /* Byte offset in each row */
} ERGSignal;

/**
 * Main ERG file structure
 * Uses memory-mapped I/O for efficient access without keeping entire file in memory
 * Uses PMR allocator for metadata strings for better performance
 */
typedef struct {
    void* impl; /* Internal C++ implementation */

    char*      erg_path;    /* Path to .erg file */
    InfoFile   info;        /* Parsed .erg.info file */
    MappedFile mapped_file; /* Memory-mapped ERG file data */

    ERGSignal* signals;      /* Array of signal metadata */
    size_t     signal_count; /* Number of signals */

    size_t data_size;    /* Size of data in file */
    size_t sample_count; /* Number of samples/rows */

    int    little_endian; /* 1 if little-endian, 0 if big-endian */
    size_t row_size;      /* Size of one data row in bytes */
} ERG;

/**
 * Get human-readable error message for ERGError
 */
const char* erg_error_string(ERGError error);

/**
 * Get the underlying error code
 * @param erg Pointer to ERG structure
 * @return Error code (can be MMapError, InfoFileError, or 0 if OK)
 *
 * Usage:
 *   ERGError err = erg_parse(&erg);
 *   if (err == ERG_MMAP_ERROR) {
 *       Error underlying = erg_get_error(&erg);
 *       printf("MMap error: %s\n", mmap_error_string((MMapError)underlying));
 *   } else if (err == ERG_INFOFILE_ERROR) {
 *       Error underlying = erg_get_error(&erg);
 *       printf("InfoFile error: %s\n", infofile_error_string((InfoFileError)underlying));
 *   }
 */
Error erg_get_error(const ERG* erg);

/**
 * Initialize an ERG structure
 * Does not load data - call erg_parse() to load
 *
 * @param erg Pointer to ERG structure
 * @param erg_file_path Path to .erg file
 */
void erg_init(ERG* erg, const char* erg_file_path);

/**
 * Parse the ERG file and load all data
 * Reads both .erg and .erg.info files
 * Returns error code on failure
 *
 * @param erg Pointer to ERG structure
 * @param print_perf_metrics (optional) If non-zero, prints detailed performance breakdown to stdout
 * @return ERG_OK on success, error code on failure
 *
 * Usage:
 *   ERGError err = erg_parse(&erg);              // C: defaults to print_perf_metrics=0
 *   ERGError err = erg_parse(&erg, 1);           // C: enable performance metrics
 *   if (err != ERG_OK) {
 *       fprintf(stderr, "Parse failed: %s\n", erg_error_string(err));
 *   }
 */
ERGError erg_parse_impl(ERG* erg, int print_perf_metrics);

/* Variadic macro to support optional print_perf_metrics parameter */
#define erg_parse_get_macro(_1, _2, NAME, ...) NAME
#define erg_parse(...)                         erg_parse_get_macro(__VA_ARGS__, erg_parse_impl, erg_parse_default)(__VA_ARGS__)
#define erg_parse_default(erg)                 erg_parse_impl(erg, 0)

/**
 * Get signal data by name (returns raw typed data)
 * Returns data in its native type (float*, double*, int*, etc.)
 * Uses memory-mapped I/O for efficient zero-copy access
 * Automatically uses OpenMP parallelization for large datasets (>5000 samples)
 * Allocates new array - caller must free
 *
 * @param erg Pointer to ERG structure
 * @param signal_name Name of signal (e.g., "Time", "Car.v")
 * @return Pointer to newly allocated typed array, NULL if signal not found
 *         Length is erg->sample_count
 *         Cast to appropriate type based on signal->type:
 *         ERG_FLOAT -> float*, ERG_DOUBLE -> double*, ERG_INT -> int*, etc.
 */
void* erg_get_signal(const ERG* erg, const char* signal_name);

/**
 * Get signal metadata by name
 *
 * @param erg Pointer to ERG structure
 * @param signal_name Name of signal
 * @return Pointer to signal metadata, NULL if not found
 */
const ERGSignal* erg_get_signal_info(const ERG* erg, const char* signal_name);

/**
 * Get index of signal by name
 *
 * @param erg Pointer to ERG structure
 * @param signal_name Name of signal
 * @return Index of signal, or -1 if not found
 */
int erg_find_signal_index(const ERG* erg, const char* signal_name);

/**
 * Free all memory associated with ERG structure
 */
void erg_free(ERG* erg);

#ifdef __cplusplus
}
#endif

#endif /* ERG_H */
