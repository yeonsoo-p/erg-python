#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "erg.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <string.h>
#include <structmember.h>

/* ERG object structure */
typedef struct {
    PyObject_HEAD ERG erg;
    int               initialized;
    int               parsed;
} ERGObject;

/* Forward declarations */
static PyTypeObject ERGType;

/* ERG.__new__ */
static PyObject* ERG_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    ERGObject* self;
    (void)args;
    (void)kwds;
    self = (ERGObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->initialized = 0;
        self->parsed      = 0;
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
    return 0;
}

/* ERG.__del__ */
static void ERG_dealloc(ERGObject* self) {
    if (self->initialized) {
        erg_free(&self->erg);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/* Internal helper function to get signal data without argument parsing */
static PyObject* ERG_get_signal_internal(ERGObject* self, const char* signal_name) {
    void*            data;
    const ERGSignal* signal_info;
    int              numpy_type;

    if (!self->parsed) {
        PyErr_SetString(PyExc_RuntimeError, "ERG file not loaded");
        return NULL;
    }

    /* Get signal info to determine type */
    signal_info = erg_get_signal_info(&self->erg, signal_name);
    if (signal_info == NULL) {
        PyErr_Format(PyExc_KeyError, "Signal '%s' not found", signal_name);
        return NULL;
    }

    /* Get signal data (returns void* in native type with scaling applied) */
    data = erg_get_signal(&self->erg, signal_name);
    if (data == NULL) {
        PyErr_Format(PyExc_KeyError, "Signal '%s' not found", signal_name);
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
    npy_intp  dims[1] = {(npy_intp)self->erg.sample_count};
    PyObject* array   = PyArray_SimpleNewFromData(1, dims, numpy_type, data);

    if (array == NULL) {
        free(data);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create NumPy array");
        return NULL;
    }

    /* Set the array to own the data so it gets freed when array is destroyed */
    PyArray_ENABLEFLAGS((PyArrayObject*)array, NPY_ARRAY_OWNDATA);

    return array;
}

/* ERG.get_signal(name) -> numpy array - FASTCALL wrapper for internal function */
static PyObject* ERG_get_signal(ERGObject* self, PyObject* const* args, Py_ssize_t nargs) {
    const char* signal_name;

    /* METH_FASTCALL requires exactly 1 argument */
    if (nargs != 1) {
        PyErr_Format(PyExc_TypeError, "get_signal() takes exactly 1 argument (%zd given)", nargs);
        return NULL;
    }

    /* Check if argument is a string */
    if (!PyUnicode_Check(args[0])) {
        PyErr_SetString(PyExc_TypeError, "signal name must be a string");
        return NULL;
    }

    signal_name = PyUnicode_AsUTF8(args[0]);
    if (signal_name == NULL) {
        return NULL;
    }

    return ERG_get_signal_internal(self, signal_name);
}


/* ERG.get_signal_unit(name) */
static PyObject* ERG_get_signal_unit(ERGObject* self, PyObject* args) {
    const char*      signal_name;
    const ERGSignal* signal;

    if (!self->parsed) {
        PyErr_SetString(PyExc_RuntimeError, "ERG file not loaded");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "s", &signal_name)) {
        return NULL;
    }

    signal = erg_get_signal_info(&self->erg, signal_name);
    if (signal == NULL) {
        PyErr_Format(PyExc_KeyError, "Signal '%s' not found", signal_name);
        return NULL;
    }

    return PyUnicode_FromString(signal->unit ? signal->unit : "");
}

/* ERG.get_signal_type(name) - returns numpy dtype */
static PyObject* ERG_get_signal_type(ERGObject* self, PyObject* args) {
    const char*      signal_name;
    const ERGSignal* signal;
    int              numpy_type;
    PyArray_Descr*   dtype;

    if (!self->parsed) {
        PyErr_SetString(PyExc_RuntimeError, "ERG file not loaded");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "s", &signal_name)) {
        return NULL;
    }

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
static PyObject* ERG_get_all_signals(ERGObject* self, PyObject* Py_UNUSED(args)) {
    size_t         i;
    PyObject*      names_list;
    PyObject*      formats_list;
    PyObject*      offsets_list;
    PyObject*      dtype_dict;
    PyArray_Descr* dtype = NULL;
    PyObject*      array;
    npy_intp       shape[1];
    const char*    data_ptr;

    if (!self->parsed) {
        PyErr_SetString(PyExc_RuntimeError, "ERG file not loaded");
        return NULL;
    }

    /* Build dtype using dict format: {'names': [...], 'formats': [...], 'offsets': [...]} */
    names_list = PyList_New(self->erg.signal_count);
    formats_list = PyList_New(self->erg.signal_count);
    offsets_list = PyList_New(self->erg.signal_count);

    if (!names_list || !formats_list || !offsets_list) {
        Py_XDECREF(names_list);
        Py_XDECREF(formats_list);
        Py_XDECREF(offsets_list);
        return NULL;
    }

    for (i = 0; i < self->erg.signal_count; i++) {
        const ERGSignal* signal = &self->erg.signals[i];
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
            PyErr_SetString(PyExc_RuntimeError, "Unsupported signal data type");
            return NULL;
        }

        /* Add name, format, and offset */
        name = PyUnicode_FromString(signal->name);
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
static PyObject* ERG_get_signal_names(ERGObject* self, PyObject* Py_UNUSED(args)) {
    PyObject* list;
    size_t    i;

    if (!self->parsed) {
        PyErr_SetString(PyExc_RuntimeError, "ERG file not loaded");
        return NULL;
    }

    list = PyList_New(self->erg.signal_count);
    if (list == NULL) {
        return NULL;
    }

    for (i = 0; i < self->erg.signal_count; i++) {
        PyObject* name = PyUnicode_FromString(self->erg.signals[i].name);
        if (name == NULL) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, i, name);
    }

    return list;
}

/* ERG.get_signal_units() */
static PyObject* ERG_get_signal_units(ERGObject* self, PyObject* Py_UNUSED(args)) {
    PyObject* dict;
    size_t    i;

    if (!self->parsed) {
        PyErr_SetString(PyExc_RuntimeError, "ERG file not loaded");
        return NULL;
    }

    dict = PyDict_New();
    if (dict == NULL) {
        return NULL;
    }

    for (i = 0; i < self->erg.signal_count; i++) {
        const ERGSignal* signal = &self->erg.signals[i];
        PyObject*        unit   = PyUnicode_FromString(signal->unit ? signal->unit : "");
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
static PyObject* ERG_get_signal_types(ERGObject* self, PyObject* Py_UNUSED(args)) {
    PyObject* dict;
    size_t    i;

    if (!self->parsed) {
        PyErr_SetString(PyExc_RuntimeError, "ERG file not loaded");
        return NULL;
    }

    dict = PyDict_New();
    if (dict == NULL) {
        return NULL;
    }

    for (i = 0; i < self->erg.signal_count; i++) {
        const ERGSignal* signal = &self->erg.signals[i];
        PyArray_Descr*   dtype;
        int              numpy_type;

        /* Map ERG type to NumPy type */
        switch (signal->type) {
        case ERG_FLOAT:    numpy_type = NPY_FLOAT; break;
        case ERG_DOUBLE:   numpy_type = NPY_DOUBLE; break;
        case ERG_INT:      numpy_type = NPY_INT32; break;
        case ERG_UINT:     numpy_type = NPY_UINT32; break;
        case ERG_SHORT:    numpy_type = NPY_INT16; break;
        case ERG_USHORT:   numpy_type = NPY_UINT16; break;
        case ERG_CHAR:     numpy_type = NPY_INT8; break;
        case ERG_UCHAR:    numpy_type = NPY_UINT8; break;
        case ERG_LONGLONG: numpy_type = NPY_INT64; break;
        case ERG_ULONGLONG: numpy_type = NPY_UINT64; break;
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
static PyObject* ERG_get_signal_factors(ERGObject* self, PyObject* Py_UNUSED(args)) {
    PyObject* dict;
    size_t    i;

    if (!self->parsed) {
        PyErr_SetString(PyExc_RuntimeError, "ERG file not loaded");
        return NULL;
    }

    dict = PyDict_New();
    if (dict == NULL) {
        return NULL;
    }

    for (i = 0; i < self->erg.signal_count; i++) {
        const ERGSignal* signal = &self->erg.signals[i];
        PyObject*        factor = PyFloat_FromDouble(signal->factor);
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
static PyObject* ERG_get_signal_factor(ERGObject* self, PyObject* args) {
    const char*      signal_name;
    const ERGSignal* signal;

    if (!self->parsed) {
        PyErr_SetString(PyExc_RuntimeError, "ERG file not loaded");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "s", &signal_name)) {
        return NULL;
    }

    signal = erg_get_signal_info(&self->erg, signal_name);
    if (signal == NULL) {
        PyErr_Format(PyExc_KeyError, "Signal '%s' not found", signal_name);
        return NULL;
    }

    return PyFloat_FromDouble(signal->factor);
}

/* ERG.get_signal_offsets() - returns dict mapping signal names to scaling offsets */
static PyObject* ERG_get_signal_offsets(ERGObject* self, PyObject* Py_UNUSED(args)) {
    PyObject* dict;
    size_t    i;

    if (!self->parsed) {
        PyErr_SetString(PyExc_RuntimeError, "ERG file not loaded");
        return NULL;
    }

    dict = PyDict_New();
    if (dict == NULL) {
        return NULL;
    }

    for (i = 0; i < self->erg.signal_count; i++) {
        const ERGSignal* signal = &self->erg.signals[i];
        PyObject*        offset = PyFloat_FromDouble(signal->offset);
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
static PyObject* ERG_get_signal_offset(ERGObject* self, PyObject* args) {
    const char*      signal_name;
    const ERGSignal* signal;

    if (!self->parsed) {
        PyErr_SetString(PyExc_RuntimeError, "ERG file not loaded");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "s", &signal_name)) {
        return NULL;
    }

    signal = erg_get_signal_info(&self->erg, signal_name);
    if (signal == NULL) {
        PyErr_Format(PyExc_KeyError, "Signal '%s' not found", signal_name);
        return NULL;
    }

    return PyFloat_FromDouble(signal->offset);
}

/* ERG.get_signal_index(name) - returns index of signal by name */
static PyObject* ERG_get_signal_index(ERGObject* self, PyObject* args) {
    const char* signal_name;
    int         index;

    if (!self->parsed) {
        PyErr_SetString(PyExc_RuntimeError, "ERG file not loaded");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "s", &signal_name)) {
        return NULL;
    }

    index = erg_find_signal_index(&self->erg, signal_name);
    if (index == -1) {
        PyErr_Format(PyExc_KeyError, "Signal '%s' not found", signal_name);
        return NULL;
    }

    return PyLong_FromLong(index);
}

/* Method definitions */
static PyMethodDef ERG_methods[] = {
    {"get_signal", (PyCFunction)(void(*)(void))ERG_get_signal, METH_FASTCALL,
     "Get signal data by name as numpy array"},
    {"get_all_signals", (PyCFunction)ERG_get_all_signals, METH_NOARGS,
     "Get all signals as structured numpy array (zero-copy view)"},
    {"get_signal_names", (PyCFunction)ERG_get_signal_names, METH_NOARGS,
     "Get list of all signal names"},
    {"get_signal_units", (PyCFunction)ERG_get_signal_units, METH_NOARGS,
     "Get dictionary mapping signal names to units"},
    {"get_signal_unit", (PyCFunction)ERG_get_signal_unit, METH_VARARGS,
     "Get unit string for a signal by name"},
    {"get_signal_types", (PyCFunction)ERG_get_signal_types, METH_NOARGS,
     "Get dictionary mapping signal names to numpy dtypes"},
    {"get_signal_type", (PyCFunction)ERG_get_signal_type, METH_VARARGS,
     "Get numpy dtype for a signal by name"},
    {"get_signal_factors", (PyCFunction)ERG_get_signal_factors, METH_NOARGS,
     "Get dictionary mapping signal names to scaling factors"},
    {"get_signal_factor", (PyCFunction)ERG_get_signal_factor, METH_VARARGS,
     "Get scaling factor for a signal by name"},
    {"get_signal_offsets", (PyCFunction)ERG_get_signal_offsets, METH_NOARGS,
     "Get dictionary mapping signal names to scaling offsets"},
    {"get_signal_offset", (PyCFunction)ERG_get_signal_offset, METH_VARARGS,
     "Get scaling offset for a signal by name"},
    {"get_signal_index", (PyCFunction)ERG_get_signal_index, METH_VARARGS,
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
