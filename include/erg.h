#ifndef ERG_H
#define ERG_H

#include <stddef.h>
#include <stdint.h>
#include <arena.h>
#include <infofile.h>

#ifdef __cplusplus
extern "C" {
#endif

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
    ERG_FLOAT,      // 4 bytes
    ERG_DOUBLE,     // 8 bytes
    ERG_LONGLONG,   // 8 bytes signed
    ERG_ULONGLONG,  // 8 bytes unsigned
    ERG_INT,        // 4 bytes signed
    ERG_UINT,       // 4 bytes unsigned
    ERG_SHORT,      // 2 bytes signed
    ERG_USHORT,     // 2 bytes unsigned
    ERG_CHAR,       // 1 byte signed
    ERG_UCHAR,      // 1 byte unsigned
    ERG_BYTES,      // Raw bytes (1-8)
    ERG_UNKNOWN
} ERGDataType;

/**
 * Metadata for a single signal/channel
 */
typedef struct {
    char*        name;      /* Signal name (e.g., "Time", "Car.v") */
    ERGDataType  type;      /* Data type */
    size_t       type_size; /* Size in bytes */
    char*        unit;      /* Unit string (e.g., "m/s", "s") */
    double       factor;    /* Scaling factor */
    double       offset;    /* Scaling offset */
} ERGSignal;

/**
 * Main ERG file structure
 * Uses memory-mapped I/O for efficient access without keeping entire file in memory
 * Uses arena allocator for metadata strings for better performance
 */
typedef struct {
    char*         erg_path;       /* Path to .erg file */
    InfoFile*     info;           /* Parsed .erg.info file */

    ERGSignal*    signals;        /* Array of signal metadata */
    size_t        signal_count;   /* Number of signals */

    size_t        data_offset;    /* Offset to data in file (after header) */
    size_t        data_size;      /* Size of data in file */
    size_t        sample_count;   /* Number of samples/rows */

    int           little_endian;  /* 1 if little-endian, 0 if big-endian */
    size_t        row_size;       /* Size of one data row in bytes */

    Arena         metadata_arena; /* Arena for all string allocations */

    /* Memory-mapped file data */
    void*         mapped_data;    /* Memory-mapped file data (NULL if not mapped) */
    size_t        mapped_size;    /* Size of mapped region */
#ifdef _WIN32
    void*         file_handle;    /* Windows file handle (HANDLE) */
    void*         mapping_handle; /* Windows mapping handle (HANDLE) */
#else
    int           file_descriptor;/* POSIX file descriptor */
#endif
} ERG;

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
 * Exits on error with descriptive message
 */
void erg_parse(ERG* erg);

/**
 * Get signal data by name (returns raw typed data)
 * Returns data in its native type (float*, double*, int*, etc.)
 * Uses memory-mapped I/O with automatic multi-threading when beneficial
 * Thread count is determined adaptively in erg_init() based on sample count
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
 * Get signal data converted to double with scaling applied
 * Convenience function that reads raw data and applies factor/offset
 * Allocates new array - caller must free
 *
 * @param erg Pointer to ERG structure
 * @param signal_name Name of signal (e.g., "Time", "Car.v")
 * @return Pointer to newly allocated double array, NULL if signal not found
 *         Length is erg->sample_count
 */
double* erg_get_signal_as_double(const ERG* erg, const char* signal_name);

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

/**
 * Batch extract multiple signals (sequential)
 * Returns raw typed data for each signal
 *
 * @param erg Pointer to ERG structure
 * @param signal_names Array of signal names to extract
 * @param num_signals Number of signals to extract
 * @param out_signals Array of output pointers (will be filled with newly allocated arrays)
 *                    Caller must free each non-NULL pointer
 *                    Each pointer will be NULL if signal not found
 */
void erg_get_signals_batch(const ERG* erg, const char** signal_names,
                           size_t num_signals, void** out_signals);

/**
 * Batch extract multiple signals as double (sequential)
 * Converts all signals to double and applies scaling
 *
 * @param erg Pointer to ERG structure
 * @param signal_names Array of signal names to extract
 * @param num_signals Number of signals to extract
 * @param out_signals Array of output double* pointers
 *                    Caller must free each non-NULL pointer
 *                    Each pointer will be NULL if signal not found
 */
void erg_get_signals_batch_as_double(const ERG* erg, const char** signal_names,
                                     size_t num_signals, double** out_signals);

#ifdef __cplusplus
}
#endif

#endif /* ERG_H */
