#include "erg_module.h"

#include <numpy/arrayobject.h>
#include <stdio.h>
#include <string.h>
#include <structmember.h>


/* Forward declarations */
static PyTypeObject ERGType;

/* Helper function to check if a signal type is a raw byte type */
static int is_raw_byte_type(ERGDataType type) {
    return (type == ERG_1BYTE || type == ERG_2BYTES || type == ERG_3BYTES ||
            type == ERG_4BYTES || type == ERG_5BYTES || type == ERG_6BYTES ||
            type == ERG_7BYTES || type == ERG_8BYTES);
}

/* ERG.__new__ */
static PyObject* ERG_new(PyTypeObject* type, PyObject* Py_UNUSED(args), PyObject* Py_UNUSED(kwds)) {
    ERGObject* self;
    self = (ERGObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->initialized               = 0;
        self->parsed                    = 0;
        self->supported_signal_count    = 0;
        self->supported_signal_indices  = NULL;
        memset(&self->erg, 0, sizeof(ERG));
    }
    return (PyObject*)self;
}

/* ERG.__init__ */
static int ERG_init(ERGObject* self, PyObject* args, PyObject* kwds) {
    PyObject*    filepath_obj;
    const char*  filepath;
    static char* kwlist[] = {"filepath", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &filepath_obj)) {
        return -1;
    }

    /* Accept both string and Path objects */
    if (PyUnicode_Check(filepath_obj)) {
        filepath = PyUnicode_AsUTF8(filepath_obj);
        if (filepath == NULL) {
            return -1;
        }
    } else {
        /* Try to convert to string using __fspath__ protocol (for Path objects) */
        PyObject* fspath = PyOS_FSPath(filepath_obj);
        if (fspath == NULL) {
            PyErr_SetString(PyExc_TypeError,
                            "filepath must be a string or path-like object");
            return -1;
        }
        filepath = PyUnicode_AsUTF8(fspath);
        Py_DECREF(fspath);
        if (filepath == NULL) {
            return -1;
        }
    }

    /* Initialize ERG structure */
    erg_init(&self->erg, filepath);
    self->initialized = 1;

    /* Auto-parse the file */
    ERGError err;
    Py_BEGIN_ALLOW_THREADS;
    err = erg_parse(&self->erg);
    Py_END_ALLOW_THREADS;

    if (err != ERG_OK) {
        const char* err_msg = erg_error_string(err);

        /* Get underlying error details if available */
        if (err == ERG_MMAP_ERROR || err == ERG_INFOFILE_ERROR) {
            Error underlying = erg_get_error(&self->erg);
            if (err == ERG_MMAP_ERROR) {
                PyErr_Format(PyExc_RuntimeError,
                             "Failed to parse ERG file: %s (MMap error: %s)",
                             err_msg, mmap_error_string((MMapError)underlying));
            } else {
                PyErr_Format(PyExc_RuntimeError,
                             "Failed to parse ERG file: %s (InfoFile error: %s)",
                             err_msg, infofile_error_string((InfoFileError)underlying));
            }
        } else {
            PyErr_Format(PyExc_RuntimeError, "Failed to parse ERG file: %s", err_msg);
        }
        return -1;
    }

    self->parsed = 1;

    /* Count supported signals (exclude raw byte types) */
    self->supported_signal_count = 0;
    for (size_t i = 0; i < self->erg.signal_count; i++) {
        if (!is_raw_byte_type(self->erg.signals[i].type)) {
            self->supported_signal_count++;
        }
    }

    /* Allocate and populate supported signal indices array */
    self->supported_signal_indices = (size_t*)malloc(self->supported_signal_count * sizeof(size_t));
    if (self->supported_signal_indices == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate signal indices array");
        return -1;
    }

    size_t idx = 0;
    for (size_t i = 0; i < self->erg.signal_count; i++) {
        if (!is_raw_byte_type(self->erg.signals[i].type)) {
            self->supported_signal_indices[idx++] = i;
        }
    }

    return 0;
}

/* ERG.__del__ */
static void ERG_dealloc(ERGObject* self) {
    if (self->initialized) {
        erg_free(&self->erg);
    }
    if (self->supported_signal_indices != NULL) {
        free(self->supported_signal_indices);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/* ERG.get_signal(name) -> numpy array */
ERG_FASTCALL(ERG_get_signal) {
    const char*      signal_name;
    void*            data;
    const ERGSignal* signal_info;
    int              numpy_type;
    npy_intp         dims[1];
    PyObject*        array;

    ERG_CHECK_PARSED;
    ERG_CHECK_NARGS(1);
    ERG_CHECK_ARGTYPE(0, PyUnicode_Check, "a string");
    ERG_GET_STRING_ARG(signal_name, 0);

    /* Get signal info to determine type and check if signal exists */
    signal_info = erg_get_signal_info(&self->erg, signal_name);
    if (signal_info == NULL) {
        PyErr_Format(PyExc_KeyError, "Signal '%s' not found", signal_name);
        return NULL;
    }

    /* Check if this is a raw byte type (not supported) */
    if (is_raw_byte_type(signal_info->type)) {
        PyErr_Format(PyExc_ValueError, "Signal '%s' has unsupported raw byte type", signal_name);
        return NULL;
    }

    /* Get signal data (returns void* in native type with scaling applied) */
    data = erg_get_signal(&self->erg, signal_name);
    if (data == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to extract signal data");
        return NULL;
    }

    /* Map ERG type to NumPy type */
    switch (signal_info->type) {
    case ERG_FLOAT:
        numpy_type = NPY_FLOAT;
        break;
    case ERG_DOUBLE:
        numpy_type = NPY_DOUBLE;
        break;
    case ERG_INT:
        numpy_type = NPY_INT32;
        break;
    case ERG_UINT:
        numpy_type = NPY_UINT32;
        break;
    case ERG_SHORT:
        numpy_type = NPY_INT16;
        break;
    case ERG_USHORT:
        numpy_type = NPY_UINT16;
        break;
    case ERG_CHAR:
        numpy_type = NPY_INT8;
        break;
    case ERG_UCHAR:
        numpy_type = NPY_UINT8;
        break;
    case ERG_LONGLONG:
        numpy_type = NPY_INT64;
        break;
    case ERG_ULONGLONG:
        numpy_type = NPY_UINT64;
        break;
    default:
        free(data);
        PyErr_SetString(PyExc_RuntimeError, "Unsupported signal data type");
        return NULL;
    }

    /* Create NumPy array directly from C data with native type */
    dims[0] = (npy_intp)self->erg.sample_count;
    array   = PyArray_SimpleNewFromData(1, dims, numpy_type, data);

    if (array == NULL) {
        free(data);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create NumPy array");
        return NULL;
    }

    /* Set the array to own the data so it gets freed when array is destroyed */
    PyArray_ENABLEFLAGS((PyArrayObject*)array, NPY_ARRAY_OWNDATA);

    return array;
}


/* ERG.get_signal_unit(name) */
ERG_FASTCALL(ERG_get_signal_unit) {
    const char*      signal_name;
    const ERGSignal* signal;

    ERG_CHECK_PARSED;
    ERG_CHECK_NARGS(1);
    ERG_CHECK_ARGTYPE(0, PyUnicode_Check, "a string");
    ERG_GET_STRING_ARG(signal_name, 0);

    signal = erg_get_signal_info(&self->erg, signal_name);
    if (signal == NULL) {
        PyErr_Format(PyExc_KeyError, "Signal '%s' not found", signal_name);
        return NULL;
    }

    return PyUnicode_FromString(signal->unit ? signal->unit : "");
}

/* ERG.get_signal_type(name) - returns numpy dtype */
ERG_FASTCALL(ERG_get_signal_type) {
    const char*      signal_name;
    const ERGSignal* signal;
    int              numpy_type;
    PyArray_Descr*   dtype;

    ERG_CHECK_PARSED;
    ERG_CHECK_NARGS(1);
    ERG_CHECK_ARGTYPE(0, PyUnicode_Check, "a string");
    ERG_GET_STRING_ARG(signal_name, 0);

    signal = erg_get_signal_info(&self->erg, signal_name);
    if (signal == NULL) {
        PyErr_Format(PyExc_KeyError, "Signal '%s' not found", signal_name);
        return NULL;
    }

    /* Map ERG type to NumPy type */
    switch (signal->type) {
    case ERG_FLOAT:
        numpy_type = NPY_FLOAT;
        break;
    case ERG_DOUBLE:
        numpy_type = NPY_DOUBLE;
        break;
    case ERG_INT:
        numpy_type = NPY_INT32;
        break;
    case ERG_UINT:
        numpy_type = NPY_UINT32;
        break;
    case ERG_SHORT:
        numpy_type = NPY_INT16;
        break;
    case ERG_USHORT:
        numpy_type = NPY_UINT16;
        break;
    case ERG_CHAR:
        numpy_type = NPY_INT8;
        break;
    case ERG_UCHAR:
        numpy_type = NPY_UINT8;
        break;
    case ERG_LONGLONG:
        numpy_type = NPY_INT64;
        break;
    case ERG_ULONGLONG:
        numpy_type = NPY_UINT64;
        break;
    default:
        PyErr_SetString(PyExc_RuntimeError, "Unsupported signal data type");
        return NULL;
    }

    dtype = PyArray_DescrFromType(numpy_type);
    if (dtype == NULL) {
        return NULL;
    }

    return (PyObject*)dtype;
}

/* ERG.get_all_signals() -> structured numpy array (zero-copy view) */
ERG_NOARGS(ERG_get_all_signals) {
    size_t         i;
    PyObject*      names_list;
    PyObject*      formats_list;
    PyObject*      offsets_list;
    PyObject*      dtype_dict;
    PyArray_Descr* dtype = NULL;
    PyObject*      array;
    npy_intp       shape[1];
    const char*    data_ptr;

    ERG_CHECK_PARSED;

    /* Build dtype using dict format: {'names': [...], 'formats': [...], 'offsets': [...]} */
    names_list   = PyList_New(self->supported_signal_count);
    formats_list = PyList_New(self->supported_signal_count);
    offsets_list = PyList_New(self->supported_signal_count);

    if (!names_list || !formats_list || !offsets_list) {
        Py_XDECREF(names_list);
        Py_XDECREF(formats_list);
        Py_XDECREF(offsets_list);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate lists");
        return NULL;
    }

    for (i = 0; i < self->supported_signal_count; i++) {
        const ERGSignal* signal = &self->erg.signals[self->supported_signal_indices[i]];
        PyObject*        name;
        PyObject*        format_str;
        PyObject*        offset_int;
        const char*      dtype_code;

        /* Convert ERG type to NumPy dtype string */
        switch (signal->type) {
        case ERG_FLOAT:
            dtype_code = "<f4";
            break;
        case ERG_DOUBLE:
            dtype_code = "<f8";
            break;
        case ERG_INT:
            dtype_code = "<i4";
            break;
        case ERG_UINT:
            dtype_code = "<u4";
            break;
        case ERG_SHORT:
            dtype_code = "<i2";
            break;
        case ERG_USHORT:
            dtype_code = "<u2";
            break;
        case ERG_CHAR:
            dtype_code = "<i1";
            break;
        case ERG_UCHAR:
            dtype_code = "<u1";
            break;
        case ERG_LONGLONG:
            dtype_code = "<i8";
            break;
        case ERG_ULONGLONG:
            dtype_code = "<u8";
            break;
        default:
            Py_DECREF(names_list);
            Py_DECREF(formats_list);
            Py_DECREF(offsets_list);
            printf("Unsupported signal type %d for signal %s\n", signal->type, signal->name);
            PyErr_SetString(PyExc_RuntimeError, "Unsupported signal data type");
            return NULL;
        }

        /* Add name, format, and offset */
        name       = PyUnicode_FromString(signal->name);
        format_str = PyUnicode_FromString(dtype_code);
        offset_int = PyLong_FromSize_t(signal->data_offset);

        if (!name || !format_str || !offset_int) {
            Py_XDECREF(name);
            Py_XDECREF(format_str);
            Py_XDECREF(offset_int);
            Py_DECREF(names_list);
            Py_DECREF(formats_list);
            Py_DECREF(offsets_list);
            return NULL;
        }

        PyList_SET_ITEM(names_list, i, name);
        PyList_SET_ITEM(formats_list, i, format_str);
        PyList_SET_ITEM(offsets_list, i, offset_int);
    }

    /* Create dtype dict */
    dtype_dict = PyDict_New();
    if (!dtype_dict) {
        Py_DECREF(names_list);
        Py_DECREF(formats_list);
        Py_DECREF(offsets_list);
        return NULL;
    }

    PyDict_SetItemString(dtype_dict, "names", names_list);
    PyDict_SetItemString(dtype_dict, "formats", formats_list);
    PyDict_SetItemString(dtype_dict, "offsets", offsets_list);
    PyDict_SetItemString(dtype_dict, "itemsize", PyLong_FromSize_t(self->erg.row_size));

    Py_DECREF(names_list);
    Py_DECREF(formats_list);
    Py_DECREF(offsets_list);

    /* Convert dict to dtype */
    if (PyArray_DescrConverter(dtype_dict, &dtype) != NPY_SUCCEED) {
        Py_DECREF(dtype_dict);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create NumPy dtype");
        return NULL;
    }
    Py_DECREF(dtype_dict);

    /* Create zero-copy view of memory-mapped data */
    shape[0] = (npy_intp)self->erg.sample_count;
    data_ptr = (const char*)self->erg.mapped_file.data + 16; /* Skip 16-byte header */

    array = PyArray_NewFromDescr(&PyArray_Type, dtype, 1, shape, NULL,
                                 (void*)data_ptr, NPY_ARRAY_DEFAULT, NULL);
    if (array == NULL) {
        return NULL;
    }

    /* Set base object to prevent NumPy from freeing the memory-mapped data */
    if (PyArray_SetBaseObject((PyArrayObject*)array, (PyObject*)self) < 0) {
        Py_DECREF(array);
        return NULL;
    }
    Py_INCREF(self);

    return array;
}

/* ERG.get_signal_names() */
ERG_NOARGS(ERG_get_signal_names) {
    PyObject* list;
    size_t    i;

    ERG_CHECK_PARSED;
    ERG_NEW_LIST(list, self->supported_signal_count);

    for (i = 0; i < self->supported_signal_count; i++) {
        size_t    signal_idx = self->supported_signal_indices[i];
        PyObject* name       = PyUnicode_FromString(self->erg.signals[signal_idx].name);
        if (name == NULL) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, i, name);
    }

    return list;
}

/* ERG.get_signal_units() */
ERG_NOARGS(ERG_get_signal_units) {
    PyObject* dict;
    size_t    i;

    ERG_CHECK_PARSED;
    ERG_NEW_DICT(dict);

    for (i = 0; i < self->supported_signal_count; i++) {
        const ERGSignal* signal = &self->erg.signals[self->supported_signal_indices[i]];

        PyObject* unit = PyUnicode_FromString(signal->unit ? signal->unit : "");
        if (unit == NULL) {
            Py_DECREF(dict);
            return NULL;
        }
        if (PyDict_SetItemString(dict, signal->name, unit) < 0) {
            Py_DECREF(unit);
            Py_DECREF(dict);
            return NULL;
        }
        Py_DECREF(unit);
    }

    return dict;
}

/* ERG.get_signal_types() - returns dict mapping signal names to numpy dtypes */
ERG_NOARGS(ERG_get_signal_types) {
    PyObject* dict;
    size_t    i;

    ERG_CHECK_PARSED;
    ERG_NEW_DICT(dict);

    for (i = 0; i < self->erg.signal_count; i++) {
        const ERGSignal* signal = &self->erg.signals[i];
        PyArray_Descr*   dtype;
        int              numpy_type;

        /* Skip raw byte types */
        if (is_raw_byte_type(signal->type)) {
            continue;
        }

        /* Map ERG type to NumPy type */
        switch (signal->type) {
        case ERG_FLOAT:
            numpy_type = NPY_FLOAT;
            break;
        case ERG_DOUBLE:
            numpy_type = NPY_DOUBLE;
            break;
        case ERG_INT:
            numpy_type = NPY_INT32;
            break;
        case ERG_UINT:
            numpy_type = NPY_UINT32;
            break;
        case ERG_SHORT:
            numpy_type = NPY_INT16;
            break;
        case ERG_USHORT:
            numpy_type = NPY_UINT16;
            break;
        case ERG_CHAR:
            numpy_type = NPY_INT8;
            break;
        case ERG_UCHAR:
            numpy_type = NPY_UINT8;
            break;
        case ERG_LONGLONG:
            numpy_type = NPY_INT64;
            break;
        case ERG_ULONGLONG:
            numpy_type = NPY_UINT64;
            break;
        default:
            Py_DECREF(dict);
            PyErr_SetString(PyExc_RuntimeError, "Unsupported signal data type");
            return NULL;
        }

        dtype = PyArray_DescrFromType(numpy_type);
        if (dtype == NULL) {
            Py_DECREF(dict);
            return NULL;
        }

        if (PyDict_SetItemString(dict, signal->name, (PyObject*)dtype) < 0) {
            Py_DECREF(dtype);
            Py_DECREF(dict);
            return NULL;
        }
        Py_DECREF(dtype);
    }

    return dict;
}

/* ERG.get_signal_factors() - returns dict mapping signal names to scaling factors */
ERG_NOARGS(ERG_get_signal_factors) {
    PyObject* dict;
    size_t    i;

    ERG_CHECK_PARSED;
    ERG_NEW_DICT(dict);

    for (i = 0; i < self->supported_signal_count; i++) {
        const ERGSignal* signal = &self->erg.signals[self->supported_signal_indices[i]];

        PyObject* factor = PyFloat_FromDouble(signal->factor);
        if (factor == NULL) {
            Py_DECREF(dict);
            return NULL;
        }
        if (PyDict_SetItemString(dict, signal->name, factor) < 0) {
            Py_DECREF(factor);
            Py_DECREF(dict);
            return NULL;
        }
        Py_DECREF(factor);
    }

    return dict;
}

/* ERG.get_signal_factor(name) - returns scaling factor for a signal */
ERG_FASTCALL(ERG_get_signal_factor) {
    const char*      signal_name;
    const ERGSignal* signal;

    ERG_CHECK_PARSED;
    ERG_CHECK_NARGS(1);
    ERG_CHECK_ARGTYPE(0, PyUnicode_Check, "a string");
    ERG_GET_STRING_ARG(signal_name, 0);

    signal = erg_get_signal_info(&self->erg, signal_name);
    if (signal == NULL) {
        PyErr_Format(PyExc_KeyError, "Signal '%s' not found", signal_name);
        return NULL;
    }

    return PyFloat_FromDouble(signal->factor);
}

/* ERG.get_signal_offsets() - returns dict mapping signal names to scaling offsets */
ERG_NOARGS(ERG_get_signal_offsets) {
    PyObject* dict;
    size_t    i;

    ERG_CHECK_PARSED;
    ERG_NEW_DICT(dict);

    for (i = 0; i < self->supported_signal_count; i++) {
        const ERGSignal* signal = &self->erg.signals[self->supported_signal_indices[i]];

        PyObject* offset = PyFloat_FromDouble(signal->offset);
        if (offset == NULL) {
            Py_DECREF(dict);
            return NULL;
        }
        if (PyDict_SetItemString(dict, signal->name, offset) < 0) {
            Py_DECREF(offset);
            Py_DECREF(dict);
            return NULL;
        }
        Py_DECREF(offset);
    }

    return dict;
}

/* ERG.get_signal_offset(name) - returns scaling offset for a signal */
ERG_FASTCALL(ERG_get_signal_offset) {
    const char*      signal_name;
    const ERGSignal* signal;

    ERG_CHECK_PARSED;
    ERG_CHECK_NARGS(1);
    ERG_CHECK_ARGTYPE(0, PyUnicode_Check, "a string");
    ERG_GET_STRING_ARG(signal_name, 0);

    signal = erg_get_signal_info(&self->erg, signal_name);
    if (signal == NULL) {
        PyErr_Format(PyExc_KeyError, "Signal '%s' not found", signal_name);
        return NULL;
    }

    return PyFloat_FromDouble(signal->offset);
}

/* ERG.get_signal_index(name) - returns index of signal by name */
ERG_FASTCALL(ERG_get_signal_index) {
    const char* signal_name;
    int         index;

    ERG_CHECK_PARSED;
    ERG_CHECK_NARGS(1);
    ERG_CHECK_ARGTYPE(0, PyUnicode_Check, "a string");
    ERG_GET_STRING_ARG(signal_name, 0);

    index = erg_find_signal_index(&self->erg, signal_name);
    if (index == -1) {
        PyErr_Format(PyExc_KeyError, "Signal '%s' not found", signal_name);
        return NULL;
    }

    return PyLong_FromLong(index);
}

/* Method definitions */
static PyMethodDef ERG_methods[] = {
    {"get_signal", (PyCFunction)(void (*)(void))ERG_get_signal, METH_FASTCALL,
     "Get signal data by name as numpy array"},
    {"get_all_signals", (PyCFunction)ERG_get_all_signals, METH_NOARGS,
     "Get all signals as structured numpy array (zero-copy view)"},
    {"get_signal_names", (PyCFunction)ERG_get_signal_names, METH_NOARGS,
     "Get list of all signal names"},
    {"get_signal_units", (PyCFunction)ERG_get_signal_units, METH_NOARGS,
     "Get dictionary mapping signal names to units"},
    {"get_signal_unit", (PyCFunction)(void (*)(void))ERG_get_signal_unit, METH_FASTCALL,
     "Get unit string for a signal by name"},
    {"get_signal_types", (PyCFunction)ERG_get_signal_types, METH_NOARGS,
     "Get dictionary mapping signal names to numpy dtypes"},
    {"get_signal_type", (PyCFunction)(void (*)(void))ERG_get_signal_type, METH_FASTCALL,
     "Get numpy dtype for a signal by name"},
    {"get_signal_factors", (PyCFunction)ERG_get_signal_factors, METH_NOARGS,
     "Get dictionary mapping signal names to scaling factors"},
    {"get_signal_factor", (PyCFunction)(void (*)(void))ERG_get_signal_factor, METH_FASTCALL,
     "Get scaling factor for a signal by name"},
    {"get_signal_offsets", (PyCFunction)ERG_get_signal_offsets, METH_NOARGS,
     "Get dictionary mapping signal names to scaling offsets"},
    {"get_signal_offset", (PyCFunction)(void (*)(void))ERG_get_signal_offset, METH_FASTCALL,
     "Get scaling offset for a signal by name"},
    {"get_signal_index", (PyCFunction)(void (*)(void))ERG_get_signal_index, METH_FASTCALL,
     "Get index of signal by name"},
    {NULL}
};

/* Type definition */
static PyTypeObject ERGType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "erg_python.ERG",
    .tp_doc                                = "ERG file reader for CarMaker binary results",
    .tp_basicsize                          = sizeof(ERGObject),
    .tp_itemsize                           = 0,
    .tp_flags                              = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new                                = ERG_new,
    .tp_init                               = (initproc)ERG_init,
    .tp_dealloc                            = (destructor)ERG_dealloc,
    .tp_methods                            = ERG_methods,
};

/* Module definition */
static PyModuleDef erg_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "erg_python",
    .m_doc  = "Python bindings for ERG (CarMaker binary results) file reader",
    .m_size = -1,
};

/* Module initialization */
PyMODINIT_FUNC PyInit_erg_python(void) {
    PyObject* m;

    /* Initialize NumPy C API */
    import_array();

    if (PyType_Ready(&ERGType) < 0)
        return NULL;

    m = PyModule_Create(&erg_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&ERGType);
    if (PyModule_AddObject(m, "ERG", (PyObject*)&ERGType) < 0) {
        Py_DECREF(&ERGType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
