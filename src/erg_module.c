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
    char         info_path[4096];
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

    /* Check if .erg file exists */
    FILE* erg_file = fopen(filepath, "rb");
    if (erg_file == NULL) {
        PyErr_Format(PyExc_FileNotFoundError, "ERG file not found: '%s'", filepath);
        return -1;
    }
    fclose(erg_file);

    /* Check if .erg.info file exists */
    snprintf(info_path, sizeof(info_path), "%s.info", filepath);
    FILE* info_file = fopen(info_path, "r");
    if (info_file == NULL) {
        PyErr_Format(PyExc_FileNotFoundError, "ERG info file not found: '%s'",
                     info_path);
        return -1;
    }
    fclose(info_file);

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

/* ERG.get_signal(name) -> numpy array */
static PyObject* ERG_get_signal(ERGObject* self, PyObject* args) {
    const char*      signal_name;
    void*            data;
    const ERGSignal* signal_info;
    int              numpy_type;

    if (!self->parsed) {
        PyErr_SetString(PyExc_RuntimeError, "ERG file not loaded");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "s", &signal_name)) {
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

/* ERG.__getitem__(name) - for dictionary-style access */
static PyObject* ERG_getitem(ERGObject* self, PyObject* key) {
    const char* signal_name;

    if (!PyUnicode_Check(key)) {
        PyErr_SetString(PyExc_TypeError, "Signal name must be a string");
        return NULL;
    }

    signal_name = PyUnicode_AsUTF8(key);
    if (signal_name == NULL) {
        return NULL;
    }

    return ERG_get_signal(self, Py_BuildValue("(s)", signal_name));
}

/* Method definitions */
static PyMethodDef ERG_methods[] = {
    {"get_signal", (PyCFunction)ERG_get_signal, METH_VARARGS,
     "Get signal data by name as numpy array"},
    {"get_signal_names", (PyCFunction)ERG_get_signal_names, METH_NOARGS,
     "Get list of all signal names"},
    {"get_signal_units", (PyCFunction)ERG_get_signal_units, METH_NOARGS,
     "Get dictionary mapping signal names to units"},
    {"get_signal_unit", (PyCFunction)ERG_get_signal_unit, METH_VARARGS,
     "Get unit string for a signal by name"},
    {"get_signal_type", (PyCFunction)ERG_get_signal_type, METH_VARARGS,
     "Get numpy dtype for a signal by name"},
    {NULL}
};

/* Mapping protocol for dictionary-style access */
static PyMappingMethods ERG_as_mapping = {
    .mp_subscript = (binaryfunc)ERG_getitem,
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
    .tp_as_mapping                         = &ERG_as_mapping,
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
