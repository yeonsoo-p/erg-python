#ifndef ERG_MODULE_H
#define ERG_MODULE_H

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "erg.h"
#include <Python.h>

#define ERG_FASTCALL(func) static PyObject* func(ERGObject* self, PyObject* const* args, Py_ssize_t nargs)
#define ERG_NOARGS(func)   static PyObject* func(ERGObject* self, PyObject* Py_UNUSED(args))

#define ERG_CHECK_PARSED                                                \
    do {                                                                \
        if (!self->parsed) {                                            \
            PyErr_SetString(PyExc_RuntimeError, "ERG file not loaded"); \
            return NULL;                                                \
        }                                                               \
    } while (0)

#define ERG_CHECK_NARGS(expected)                                                                                                                     \
    do {                                                                                                                                              \
        if (nargs != (expected)) {                                                                                                                    \
            PyErr_Format(PyExc_TypeError, "%s() takes exactly %d argument%s (%zd given)", __func__, (expected), ((expected) == 1 ? "" : "s"), nargs); \
            return NULL;                                                                                                                              \
        }                                                                                                                                             \
    } while (0)

#define ERG_CHECK_ARGTYPE(index, check_func, type_name)                                    \
    do {                                                                                   \
        if (!check_func(args[index])) {                                                    \
            PyErr_Format(PyExc_TypeError, "argument %d must be %s", (index), (type_name)); \
            return NULL;                                                                   \
        }                                                                                  \
    } while (0)

#define ERG_GET_STRING_ARG(var, index)                                       \
    do {                                                                 \
        var = PyUnicode_AsUTF8(args[index]);                             \
        if (var == NULL) {                                               \
            PyErr_SetString(PyExc_RuntimeError, "Failed to decode string argument"); \
            return NULL;                                                 \
        }                                                                \
    } while (0)

#define ERG_NEW_LIST(var, size)                                         \
    do {                                                                \
        var = PyList_New(size);                                         \
        if (var == NULL) {                                              \
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate list"); \
            return NULL;                                                \
        }                                                               \
    } while (0)

#define ERG_NEW_DICT(var)                                               \
    do {                                                                \
        var = PyDict_New();                                             \
        if (var == NULL) {                                              \
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate dict"); \
            return NULL;                                                \
        }                                                               \
    } while (0)

typedef struct {
    PyObject_HEAD ERG erg;
    int     initialized;
    int     parsed;
    size_t  supported_signal_count; /* Number of signals excluding raw byte types */
    size_t* supported_signal_indices; /* Indices of supported signals */
} ERGObject;

#endif /* ERG_MODULE_H */