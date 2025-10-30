#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include "erg.h"
#include <string.h>

/* ERG object structure */
typedef struct {
    PyObject_HEAD
    ERG erg;
    int initialized;
    int parsed;
} ERGObject;

/* Forward declarations */
static PyTypeObject ERGType;

/* ERG.__new__ */
static PyObject *ERG_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    ERGObject *self;
    self = (ERGObject *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->initialized = 0;
        self->parsed = 0;
        memset(&self->erg, 0, sizeof(ERG));
    }
    return (PyObject *)self;
}

/* ERG.__init__ */
static int ERG_init(ERGObject *self, PyObject *args, PyObject *kwds) {
    const char *filepath;
    static char *kwlist[] = {"filepath", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &filepath)) {
        return -1;
    }

    /* Initialize ERG structure */
    erg_init(&self->erg, filepath);
    self->initialized = 1;

    return 0;
}

/* ERG.__del__ */
static void ERG_dealloc(ERGObject *self) {
    if (self->initialized) {
        erg_free(&self->erg);
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* ERG.parse() */
static PyObject *ERG_parse(ERGObject *self, PyObject *Py_UNUSED(ignored)) {
    if (!self->initialized) {
        PyErr_SetString(PyExc_RuntimeError, "ERG object not initialized");
        return NULL;
    }

    /* Parse the ERG file */
    Py_BEGIN_ALLOW_THREADS
    erg_parse(&self->erg);
    Py_END_ALLOW_THREADS

    self->parsed = 1;
    Py_RETURN_NONE;
}

/* ERG.get_signal(name) -> numpy array or list */
static PyObject *ERG_get_signal(ERGObject *self, PyObject *args) {
    const char *signal_name;
    double *data;
    PyObject *list;
    size_t i;

    if (!self->parsed) {
        PyErr_SetString(PyExc_RuntimeError, "ERG file not parsed. Call parse() first");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "s", &signal_name)) {
        return NULL;
    }

    /* Get signal data as double */
    Py_BEGIN_ALLOW_THREADS
    data = erg_get_signal_as_double(&self->erg, signal_name);
    Py_END_ALLOW_THREADS

    if (data == NULL) {
        PyErr_Format(PyExc_KeyError, "Signal '%s' not found", signal_name);
        return NULL;
    }

    /* Try to import numpy and create array */
    PyObject *numpy = PyImport_ImportModule("numpy");
    if (numpy != NULL) {
        PyObject *asarray = PyObject_GetAttrString(numpy, "asarray");
        Py_DECREF(numpy);

        if (asarray != NULL) {
            /* Create list first */
            list = PyList_New(self->erg.sample_count);
            if (list == NULL) {
                free(data);
                Py_DECREF(asarray);
                return NULL;
            }

            for (i = 0; i < self->erg.sample_count; i++) {
                PyObject *val = PyFloat_FromDouble(data[i]);
                if (val == NULL) {
                    Py_DECREF(list);
                    free(data);
                    Py_DECREF(asarray);
                    return NULL;
                }
                PyList_SET_ITEM(list, i, val);
            }

            free(data);

            /* Convert to numpy array */
            PyObject *np_args = PyTuple_Pack(1, list);
            PyObject *result = PyObject_CallObject(asarray, np_args);
            Py_DECREF(np_args);
            Py_DECREF(list);
            Py_DECREF(asarray);
            return result;
        }
        Py_XDECREF(asarray);
    }

    /* Fallback to list if numpy not available */
    PyErr_Clear();

    list = PyList_New(self->erg.sample_count);
    if (list == NULL) {
        free(data);
        return NULL;
    }

    for (i = 0; i < self->erg.sample_count; i++) {
        PyObject *val = PyFloat_FromDouble(data[i]);
        if (val == NULL) {
            Py_DECREF(list);
            free(data);
            return NULL;
        }
        PyList_SET_ITEM(list, i, val);
    }

    free(data);
    return list;
}

/* ERG.get_signals(names) -> dict */
static PyObject *ERG_get_signals(ERGObject *self, PyObject *args) {
    PyObject *name_list;
    PyObject *result;
    Py_ssize_t num_signals;
    const char **signal_names = NULL;
    double **out_signals = NULL;
    Py_ssize_t i;

    if (!self->parsed) {
        PyErr_SetString(PyExc_RuntimeError, "ERG file not parsed. Call parse() first");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &name_list)) {
        return NULL;
    }

    num_signals = PyList_Size(name_list);
    if (num_signals == 0) {
        return PyDict_New();
    }

    /* Allocate arrays */
    signal_names = (const char **)malloc(num_signals * sizeof(char *));
    out_signals = (double **)malloc(num_signals * sizeof(double *));

    if (signal_names == NULL || out_signals == NULL) {
        free(signal_names);
        free(out_signals);
        return PyErr_NoMemory();
    }

    /* Extract signal names from list */
    for (i = 0; i < num_signals; i++) {
        PyObject *item = PyList_GetItem(name_list, i);
        if (!PyUnicode_Check(item)) {
            free(signal_names);
            free(out_signals);
            PyErr_SetString(PyExc_TypeError, "Signal names must be strings");
            return NULL;
        }
        signal_names[i] = PyUnicode_AsUTF8(item);
    }

    /* Get signals in batch */
    Py_BEGIN_ALLOW_THREADS
    erg_get_signals_batch_as_double(&self->erg, signal_names, num_signals, out_signals);
    Py_END_ALLOW_THREADS

    /* Create result dictionary */
    result = PyDict_New();
    if (result == NULL) {
        goto cleanup;
    }

    /* Check if numpy is available */
    PyObject *numpy = PyImport_ImportModule("numpy");
    PyObject *asarray = NULL;
    if (numpy != NULL) {
        asarray = PyObject_GetAttrString(numpy, "asarray");
        Py_DECREF(numpy);
    }
    PyErr_Clear();

    for (i = 0; i < num_signals; i++) {
        if (out_signals[i] != NULL) {
            PyObject *list = PyList_New(self->erg.sample_count);
            if (list == NULL) {
                goto cleanup;
            }

            for (size_t j = 0; j < self->erg.sample_count; j++) {
                PyObject *val = PyFloat_FromDouble(out_signals[i][j]);
                if (val == NULL) {
                    Py_DECREF(list);
                    goto cleanup;
                }
                PyList_SET_ITEM(list, j, val);
            }

            PyObject *data;
            if (asarray != NULL) {
                /* Convert to numpy array */
                PyObject *np_args = PyTuple_Pack(1, list);
                data = PyObject_CallObject(asarray, np_args);
                Py_DECREF(np_args);
                Py_DECREF(list);
                if (data == NULL) {
                    goto cleanup;
                }
            } else {
                data = list;
            }

            PyDict_SetItemString(result, signal_names[i], data);
            Py_DECREF(data);
            free(out_signals[i]);
        }
    }

    Py_XDECREF(asarray);
    free(signal_names);
    free(out_signals);
    return result;

cleanup:
    Py_XDECREF(asarray);
    for (i = 0; i < num_signals; i++) {
        if (out_signals[i] != NULL) {
            free(out_signals[i]);
        }
    }
    free(signal_names);
    free(out_signals);
    Py_XDECREF(result);
    return NULL;
}

/* ERG.signal_names property */
static PyObject *ERG_get_signal_names(ERGObject *self, void *closure) {
    PyObject *list;
    size_t i;

    if (!self->parsed) {
        PyErr_SetString(PyExc_RuntimeError, "ERG file not parsed. Call parse() first");
        return NULL;
    }

    list = PyList_New(self->erg.signal_count);
    if (list == NULL) {
        return NULL;
    }

    for (i = 0; i < self->erg.signal_count; i++) {
        PyObject *name = PyUnicode_FromString(self->erg.signals[i].name);
        if (name == NULL) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, i, name);
    }

    return list;
}

/* ERG.sample_count property */
static PyObject *ERG_get_sample_count(ERGObject *self, void *closure) {
    if (!self->parsed) {
        PyErr_SetString(PyExc_RuntimeError, "ERG file not parsed. Call parse() first");
        return NULL;
    }
    return PyLong_FromSize_t(self->erg.sample_count);
}

/* ERG.get_signal_info(name) */
static PyObject *ERG_get_signal_info(ERGObject *self, PyObject *args) {
    const char *signal_name;
    const ERGSignal *signal;
    PyObject *dict;

    if (!self->parsed) {
        PyErr_SetString(PyExc_RuntimeError, "ERG file not parsed. Call parse() first");
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

    dict = PyDict_New();
    if (dict == NULL) {
        return NULL;
    }

    PyDict_SetItemString(dict, "name", PyUnicode_FromString(signal->name));
    PyDict_SetItemString(dict, "unit", PyUnicode_FromString(signal->unit ? signal->unit : ""));
    PyDict_SetItemString(dict, "factor", PyFloat_FromDouble(signal->factor));
    PyDict_SetItemString(dict, "offset", PyFloat_FromDouble(signal->offset));
    PyDict_SetItemString(dict, "type_size", PyLong_FromSize_t(signal->type_size));

    const char *type_name;
    switch (signal->type) {
        case ERG_FLOAT: type_name = "float"; break;
        case ERG_DOUBLE: type_name = "double"; break;
        case ERG_LONGLONG: type_name = "longlong"; break;
        case ERG_ULONGLONG: type_name = "ulonglong"; break;
        case ERG_INT: type_name = "int"; break;
        case ERG_UINT: type_name = "uint"; break;
        case ERG_SHORT: type_name = "short"; break;
        case ERG_USHORT: type_name = "ushort"; break;
        case ERG_CHAR: type_name = "char"; break;
        case ERG_UCHAR: type_name = "uchar"; break;
        case ERG_BYTES: type_name = "bytes"; break;
        default: type_name = "unknown"; break;
    }
    PyDict_SetItemString(dict, "type", PyUnicode_FromString(type_name));

    return dict;
}

/* Method definitions */
static PyMethodDef ERG_methods[] = {
    {"parse", (PyCFunction)ERG_parse, METH_NOARGS,
     "Parse the ERG file and load metadata"},
    {"get_signal", (PyCFunction)ERG_get_signal, METH_VARARGS,
     "Get signal data by name as numpy array or list"},
    {"get_signals", (PyCFunction)ERG_get_signals, METH_VARARGS,
     "Get multiple signals as dict (batch operation)"},
    {"get_signal_info", (PyCFunction)ERG_get_signal_info, METH_VARARGS,
     "Get signal metadata by name"},
    {NULL}
};

/* Property definitions */
static PyGetSetDef ERG_getsetters[] = {
    {"signal_names", (getter)ERG_get_signal_names, NULL,
     "List of all signal names", NULL},
    {"sample_count", (getter)ERG_get_sample_count, NULL,
     "Number of samples in the file", NULL},
    {NULL}
};

/* Type definition */
static PyTypeObject ERGType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "erg_python.ERG",
    .tp_doc = "ERG file reader for CarMaker binary results",
    .tp_basicsize = sizeof(ERGObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = ERG_new,
    .tp_init = (initproc)ERG_init,
    .tp_dealloc = (destructor)ERG_dealloc,
    .tp_methods = ERG_methods,
    .tp_getset = ERG_getsetters,
};

/* Module definition */
static PyModuleDef erg_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "erg_python",
    .m_doc = "Python bindings for ERG (CarMaker binary results) file reader",
    .m_size = -1,
};

/* Module initialization */
PyMODINIT_FUNC PyInit_erg_python(void) {
    PyObject *m;

    if (PyType_Ready(&ERGType) < 0)
        return NULL;

    m = PyModule_Create(&erg_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&ERGType);
    if (PyModule_AddObject(m, "ERG", (PyObject *)&ERGType) < 0) {
        Py_DECREF(&ERGType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
