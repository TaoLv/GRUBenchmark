
import theano
from theano import tensor, gof
from theano.tensor.blas import ldflags


class GRU_Inference(gof.Op):
    __props__ = ('hid', 'step', 'dim', 'return_sequences', 'max_len')

    def __init__(self, hid, step=None, dim=None, return_sequences=True, max_len=None):
        self.hid = hid
        self.step = step
        self.dim = dim
        self.return_sequences = return_sequences
        self.max_len=max_len
        super(GRU_Inference, self).__init__()

    def make_node(self, *inputs):
        """
        inputs: X, Wx, Wh, hid_init, bias. bias is optional.

        """
        if len(inputs) in (4, 5):
            inp = list(map(tensor.as_tensor_variable, inputs))
        else:
            raise ValueError('GRU: number of parameter is wrong.')

        if len(inputs) == 5:
            self.bias = True
            assert inp[-1].ndim is 1
        else:
            self.bias = False

        assert inp[0].ndim is 3
        assert inp[1].ndim is 2
        assert inp[2].ndim is 2
        assert inp[3].ndim is 2

        if self.return_sequences:
            out = [inp[0].type()]
        else:
            bcast = [inp[0].type.broadcastable[1], inp[0].type.broadcastable[2]]
            out = [tensor.tensor(dtype=inp[0].type.dtype, broadcastable=bcast)()]

        return gof.Apply(self, inp, out)

    def c_headers(self):
        headers = ['<mkl.h>', '<omp.h>']
        return headers

    def c_libraries(self):
        return ldflags()

    def c_support_code_struct(self, node, name):
        if node.inputs[0].type.dtype == 'float32':
            dtype = 'float'
        elif node.inputs[0].type.dtype == 'float64':
            dtype = 'double'
        else:
            raise TypeError('GRU: dtype %s is not supported.'
                            % (node.inputs[0].type.dtype))

        ccode = """
        %(dtype)s** A;
        %(dtype)s** B;
        %(dtype)s** C;

        MKL_INT lda[1];
        MKL_INT ldb[1];
        MKL_INT ldc[1];

        MKL_INT m[1];
        MKL_INT n[1];
        MKL_INT k[1];

        CBLAS_TRANSPOSE transA[1];
        CBLAS_TRANSPOSE transB[1];

        %(dtype)s alpha[1];
        %(dtype)s beta[1];
        MKL_INT size_per_grp[1];

        size_t time_step;
        size_t batch_size;
        size_t embed_dims;
        size_t max_len;

        %(dtype)s* temp;
        %(dtype)s* x_hzr;
        %(dtype)s* zt_ptr;
        %(dtype)s* rt_ptr;

        """ % locals()
        return ccode

    def c_init_code_struct(self, node, name, sub):

        ccode = """
        A = NULL;
        B = NULL;
        C = NULL;

        lda[0] = 0;
        ldb[0] = 0;
        ldc[0] = 0;

        m[0] = 0;
        n[0] = 0;
        k[0] = 0;

        alpha[0] = 1.0;
        beta[0] = 1.0;

        transA[0] = CblasNoTrans;
        transB[0] = CblasNoTrans;
        size_per_grp[0] = 1;

        time_step = 0;
        batch_size = 0;
        embed_dims = 0;
        max_len = 0;

        temp = NULL;
        x_hzr = NULL;
        zt_ptr = NULL;
        rt_ptr = NULL;

        """ % locals()
        return ccode

    def c_cleanup_code_struct(self, node, name):
        ccode = """
        // printf("c_clean_code_struct\\n");
        if (A) {
            mkl_free (A);
            A = NULL;
        }

        if (B) {
            mkl_free (B);
            B = NULL;
        }

        if (C) {
            mkl_free (C);
            C = NULL;
        }

        if (temp) {
            mkl_free(temp);
        }

        if (zt_ptr) {
            mkl_free(zt_ptr);
        }

        if (rt_ptr) {
            mkl_free(rt_ptr);
        }

        if (x_hzr) {
            mkl_free(x_hzr);
        }

        """
        return ccode

    def c_code(self, node, name, inputs, outputs, sub):
        if len(inputs) is 4:
            with_bias = 0
            X, W_x, W_h, hid_init = inputs
        elif len(inputs) is 5:
            with_bias = 1
            X, W_x, W_h, hid_init, b = inputs
        else:
            raise TypeError('GRU: wrong number of arguments, expecting for 4 or 5')

        z, = outputs
        hid = self.hid
        if self.return_sequences:
            return_sequences = 1
        else:
            return_sequences = 0

        if self.max_len:
            max_len = int(self.max_len)
        else:
            max_len = 0

        if node.inputs[0].type.dtype == 'float32':
            dtype = 's'
            d = 'float'
        elif node.inputs[0].type.dtype == 'float64':
            dtype = 'd'
            d = 'double'
        else:
            raise TypeError('GRU: dtype %s is not supported.'
                            % (node.inputs[0].type.dtype))

        ccode = """
            double tic = dsecnd();
            time_step  = PyArray_DIMS(%(X)s)[0];
            embed_dims = PyArray_DIMS(%(X)s)[1];
            batch_size = PyArray_DIMS(%(X)s)[2];

            assert (%(hid)s == PyArray_DIMS(%(W_x)s)[0] / 3);

            npy_intp dims[3] = {0, 0, 0};
            %(d)s* x_ptr     = NULL;
            %(d)s* w_x_ptr   = NULL;
            %(d)s* w_h_ptr   = NULL;
            %(d)s* b_ptr     = NULL;   // If no bias input, keep b_ptr as NULL.
            %(d)s* hinit_ptr = NULL;

            // if max_len is given, and it's small than time_step,
            // a runtime error will be threw out.
            if (%(max_len)s && %(max_len)s < time_step) {
                PyErr_Format(PyExc_RuntimeError,
                             \"GRU: length of input sequence is too long: %%d,\"
                             \" max length is %%d\",
                             time_step, %(max_len)s);
                goto gru_fail;
            }

            if (%(max_len)s) {
                max_len = %(max_len)s;
            }

            // vmlSetMode(vmlGetMode() & 0xFFFFFFF0 | VML_EP);

            // if max_len is not input as a constraint, we need to
            // recreate buffers and output objects if time_step increased
            // among iterations. Otherwise, we will reuse those buffers
            // and output objects.
            if (max_len < time_step) {
                printf(\"GRU: no max_len constraint, time_step increased: %%d -> %%d\\n\", max_len, time_step);
                mkl_free(A);            A = NULL;
                mkl_free(B);            B = NULL;
                mkl_free(C);            C = NULL;
                mkl_free(x_hzr);        x_hzr = NULL;
                max_len = time_step;
            }

            if (NULL == A) {
                A = (%(d)s**)mkl_malloc(3 * max_len * sizeof (%(d)s*), 64);
            }

            if (NULL == B) {
                B = (%(d)s**)mkl_malloc(3 * max_len * sizeof (%(d)s*), 64);
            }

            if (NULL == C) {
                C = (%(d)s**)mkl_malloc(3 * max_len * sizeof (%(d)s*), 64);
            }

            if (NULL == A || NULL == B || NULL == C) {
                PyErr_SetString(PyExc_MemoryError,
                                \"GRU: malloc A, B, C buffers for gemm_batch failed\");
                goto gru_fail;
            }

            PyArrayObject* x_src     = NULL;
            PyArrayObject* w_x_src   = NULL;
            PyArrayObject* w_h_src   = NULL;
            PyArrayObject* b_src     = NULL;   // If no bias input, keep it with NULL
            PyArrayObject* hinit_src = NULL;

            if (!PyArray_IS_C_CONTIGUOUS(%(X)s)) {
                printf(\"Warning: GRU need convert X to C-Contiguous\\n\");
                x_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(X)s,
                                            PyArray_TYPE(%(X)s),
                                            PyArray_NDIM(%(X)s),
                                            PyArray_NDIM(%(X)s));
                if (!x_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"GRU: fail to cast X to contiguous array\");
                    goto gru_fail;
                }
                x_ptr = (%(d)s*) PyArray_DATA(x_src);
            } else {
                x_ptr = (%(d)s*) PyArray_DATA(%(X)s);
            }

            //// step 1. Dot(X, (W_x))
            if (embed_dims != PyArray_DIMS(%(W_x)s)[1]) {
                PyErr_SetString(PyExc_RuntimeError, \"GRU: W_x * X size error\");
                goto gru_fail;
            }

            if (!PyArray_IS_C_CONTIGUOUS(%(W_x)s)) {
                printf(\"Warning: GRU need convert W_x to C-Contiguous\\n\");
                w_x_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(W_x)s,
                                            PyArray_TYPE(%(W_x)s),
                                            PyArray_NDIM(%(W_x)s),
                                            PyArray_NDIM(%(W_x)s));
                if (!w_x_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"GRU: fail to cast W_x to contiguous array\");
                    goto gru_fail;
                }
                w_x_ptr = (%(d)s*) PyArray_DATA(w_x_src);
            } else {
                w_x_ptr = (%(d)s*) PyArray_DATA(%(W_x)s);
            }

            if (!PyArray_IS_C_CONTIGUOUS(%(W_h)s)) {
                printf(\"Warning: GRU need convert W_h to C-Contiguous\\n\");
                w_h_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(W_h)s,
                                            PyArray_TYPE(%(W_h)s),
                                            PyArray_NDIM(%(W_h)s),
                                            PyArray_NDIM(%(W_h)s));
                if (!w_h_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"GRU: fail to cast W_h to contiguous array\");
                    goto gru_fail;
                }
                w_h_ptr = (%(d)s*) PyArray_DATA(w_h_src);
            } else {
                w_h_ptr = (%(d)s*) PyArray_DATA(%(W_h)s);
            }

            if (!PyArray_IS_C_CONTIGUOUS(%(hid_init)s)) {
                printf(\"Warning: GRU need convert hid_init to C-Contiguous\\n\");
                hinit_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(hid_init)s,
                                            PyArray_TYPE(%(hid_init)s),
                                            PyArray_NDIM(%(hid_init)s),
                                            PyArray_NDIM(%(hid_init)s));
                if (!hinit_src) {
                    PyErr_SetString(PyExc_RuntimeError, \"GRU: fail to cast hid_init to contiguous array\");
                    goto gru_fail;
                }
                hinit_ptr = (%(d)s*) PyArray_DATA(hinit_src);
            } else {
                hinit_ptr = (%(d)s*) PyArray_DATA(%(hid_init)s);
            }

            // x_hzr has shape (3 * max_len, batch_size, %(hid)s)
            if (NULL == x_hzr) {
                x_hzr = (%(d)s*)mkl_calloc(max_len * 3 * batch_size * %(hid)s, sizeof (%(d)s), 64);
            }

            if (NULL == x_hzr) {
                PyErr_SetString(PyExc_MemoryError,
                                \"GRU: fail to allocate buffer for x_hzr\");
                goto gru_fail;
            }

        """ % locals()

        if with_bias:
            ccode += """
                if (!PyArray_IS_C_CONTIGUOUS(%(b)s)) {
                    printf(\"Warning: Need convert bias to C-Contiguous\\n\");
                    b_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)%(b)s,
                                                PyArray_TYPE(%(b)s),
                                                PyArray_NDIM(%(b)s),
                                                PyArray_NDIM(%(b)s));
                    if (!b_src) {
                        PyErr_SetString(PyExc_RuntimeError, \"GRU: fail to case bias to contiguous array\");
                        goto gru_fail;
                    }
                    b_ptr = (%(d)s*) PyArray_DATA(b_src);
                } else {
                    b_ptr = (%(d)s*) PyArray_DATA(%(b)s);
                }

                #pragma omp parallel for
                for (int i = 0; i < time_step; i++) {
                    for (int j = 0; j < batch_size; j++) {
                        for (int k = 0; k < %(hid)s; k++) }
                            size_t ind1 = i * batch_size * %(hid)s + j * %(hid)s + k;
                            size_t ind2 = (i + time_stemp) * batch_size * %(hid)s + j * %(hid)s + k;
                            size_t ind3 = (i + 2 * time_step) * batch_size * %(hid)s + j * %(hid)s + k;

                            x_hzr[ind1] = b_ptr[k];
                            x_hzr[ind2] = b_ptr[k + time_stemp];
                            x_hzr[ind3] = b_ptr[k + 2 * time_step];
                        }
                    }
                }
            """ % locals()
        else:
            ccode += """    
                // x_hzr is allocated by mkl_calloc and set to zero already.
                // memset((char*)x_hzr, 0, max_len * 3 * batch_size * %(hid)s * sizeof (%(d)s));
            """ % locals()

        ccode += """
            // printf(\"head: %%.8f\\n\", (dsecnd() - tic) * 1000.0);

            m[0] = %(hid)s;
            k[0] = embed_dims;
            n[0] = batch_size;

            #pragma omp parallel for
            for (int i = 0; i < time_step; i++) {
                A[i] = w_x_ptr;                                            // w_xh
                A[i + time_step] = w_x_ptr + embed_dims * %(hid)s;         // w_xz
                A[i + 2 * time_step] = w_x_ptr + 2 * embed_dims * %(hid)s; // w_xr

                B[i] = x_ptr + i * k[0] * n[0];
                B[i + time_step] = B[i];
                B[i + 2 * time_step] = B[i];

                C[i] = x_hzr + i * m[0] * n[0];
                C[i + time_step] = x_hzr + (i + time_step) * m[0] * n[0];
                C[i + 2 * time_step] = x_hzr + (i + 2 * time_step) * m[0] * n[0];
            }

            if (%(with_bias)s) {
                beta[0] = 1.0;
            } else {
                beta[0] = 0.0;
            }

            size_per_grp[0] = 3 * time_step;
            tic = dsecnd();
            cblas_%(dtype)sgemm_batch(CblasRowMajor, transA, transB, m, n, k,
                                      alpha, A, k, B, n, beta, C, n, 1, size_per_grp);
            // printf(\"batchgemm: %%.8f\\n\", (dsecnd() - tic) * 1000.0);

            //// step 2. construct output
            tic = dsecnd();
            if ( NULL == %(z)s) {
                if (%(return_sequences)s) {
                    dims[0] = time_step;
                    dims[1] = %(hid)s;
                    dims[2] = batch_size;
                    %(z)s = (PyArrayObject*) PyArray_ZEROS(3, dims, PyArray_TYPE(%(X)s), 0);
                } else {
                    dims[0] = %(hid)s;
                    dims[1] = batch_size;
                    %(z)s = (PyArrayObject*) PyArray_ZEROS(2, dims, PyArray_TYPE(%(X)s), 0);
                }
            } else {
                if (%(return_sequences)s) {
                    if (PyArray_NDIM(%(z)s) != 3 ||
                        PyArray_DIMS(%(z)s)[0] != time_step ||
                        PyArray_DIMS(%(z)s)[1] != %(hid)s ||
                        PyArray_DIMS(%(z)s)[2] != batch_size) {
                        // printf(\"delete output and recreate it\\n\");
                        Py_DECREF(%(z)s);
                        dims[0] = time_step;
                        dims[1] = %(hid)s;
                        dims[2] = batch_size;
                        %(z)s = (PyArrayObject*) PyArray_ZEROS(3, dims, PyArray_TYPE(%(X)s), 0);
                    }
                } else {
                    if (PyArray_NDIM(%(z)s) != 2 ||
                        PyArray_DIMS(%(z)s)[0] != %(hid)s ||
                        PyArray_DIMS(%(z)s)[1] != batch_size) {
                        Py_DECREF(%(z)s);
                        dims[0] = %(hid)s;
                        dims[1] = batch_size;
                        %(z)s = (PyArrayObject*) PyArray_ZEROS(2, dims, PyArray_TYPE(%(X)s), 0);
                    }
                }
            }

            if (NULL == %(z)s) {
                PyErr_SetString(PyExc_RuntimeError, \"GRU: create output array failed\");
                goto gru_fail;
            }

            //// step 3: step on time_step
            // loop on step
            if (NULL == temp) {
                temp = (%(d)s*)mkl_malloc(batch_size * %(hid)s * sizeof (%(d)s), 64);
            }

            if (NULL == temp) {
                PyErr_SetString(PyExc_MemoryError,
                                \"GRU: fail to allocate buffer for temp\");
                goto gru_fail;
            }
            memset((void*)temp, 0, batch_size * %(hid)s * sizeof (%(d)s));

            if (NULL == zt_ptr) {
                zt_ptr = (%(d)s*)mkl_malloc(batch_size * %(hid)s * sizeof (%(d)s), 64);
            }

            if (NULL == zt_ptr) {
                PyErr_SetString(PyExc_MemoryError,
                                \"GRU: fail to allocate buffer for zt_ptr\");
                goto gru_fail;
            }

            if (NULL == rt_ptr) {
                rt_ptr = (%(d)s*)mkl_malloc(batch_size * %(hid)s * sizeof (%(d)s), 64);
            }

            if (NULL == rt_ptr) {
                PyErr_SetString(PyExc_MemoryError,
                                \"GRU: fail to allocate buffer for rt_ptr\");
                goto gru_fail;
            }

            // printf(\"alloc: %%.8f\\n\", (dsecnd() - tic) * 1000.0);
            m[0]            = %(hid)s;
            k[0]            = %(hid)s;
            n[0]            = batch_size;
            beta[0]         = 1.0;
            size_per_grp[0] = 2;

            %(d)s* wh_z = (%(d)s*)PyArray_DATA(%(W_h)s) + %(hid)s * %(hid)s;
            %(d)s* wh_r = (%(d)s*)PyArray_DATA(%(W_h)s) + 2 * %(hid)s * %(hid)s;
            %(d)s* wh_h = (%(d)s*)PyArray_DATA(%(W_h)s);

            A[0] = wh_z;
            A[1] = wh_r;
            A[2] = wh_h;
            B[0] = hinit_ptr;
            B[1] = hinit_ptr;
            B[2] = hinit_ptr;
 
            double acc1 = 0.0f;
            double acc2 = 0.0f;
            double acc3 = 0.0f;
            double acc4 = 0.0f;
            double acc5 = 0.0f;

            for (int i = 0; i < time_step; i++) {
                //z_t, r_t
                C[0] = x_hzr + (i + time_step) * m[0] * n[0];
                C[1] = x_hzr + (i + 2 * time_step) * m[0] * n[0];
                C[2] = temp;

                // do below two function with batch-gemm first, then sigmoid respectively
                // z_t = K.sigmoid(x_z + K.dot(h_tm1, self.W_hz) + self.b_z)
                // r_t = K.sigmoid(x_r + K.dot(h_tm1, self.W_hr) + self.b_r)

                tic = dsecnd();
                cblas_%(dtype)sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m[0], n[0], k[0],
                                    alpha[0], A[0], k[0], B[0], n[0], beta[0], C[0], n[0]);
                cblas_%(dtype)sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m[0], n[0], k[0],
                                    alpha[0], A[1], k[0], B[1], n[0], beta[0], C[1], n[0]);
                acc1 += dsecnd() - tic;

                // sigmoid(C[0]), sigmoid(C[1])
                // v%(dtype)sExp(m[0] * n[0], C[0], C[0]);
                // v%(dtype)sExp(m[0] * n[0], C[1], C[1]);
                size_t mn = m[0] * n[0];
                int t = 0;
                tic = dsecnd();
                #pragma omp parallel for
                for (t = 0; t < mn; t++) {
                    double exp_zt = exp((double)(C[0][t]));
                    double exp_rt = exp((double)(C[1][t]));

                    // Save rt and zt for workspace
                    zt_ptr[t] = (%(d)s)(exp_zt / ((double)1.0 + exp_zt));
                    rt_ptr[t] = (%(d)s)(exp_rt / ((double)1.0 + exp_rt));
                }
                acc2 += dsecnd() - tic;

                // GEMM -> Mul -> Add -> tanh
                // can_h_t = K.tanh(x_h + r_t * K.dot(h_tm1, self.W_hh) + self.b_h)
                tic = dsecnd();
                cblas_%(dtype)sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m[0], n[0], k[0],
                                    alpha[0], A[2], k[0], B[2], n[0], beta[0], C[2], n[0]);
                acc3 += dsecnd() - tic;

                // v%(dtype)sMul(batch_size * %(hid)s, rt_ptr, temp, temp);
                // v%(dtype)sAdd(batch_size * %(hid)s, x_hzr + i * m[0] * n[0], temp, temp);

                // tanh(temp), save hcan(t)
                // v%(dtype)sTanh(m[0] * n[0], temp, hcan_ptr);
                tic = dsecnd();
                #pragma omp parallel for
                for (t = 0; t < mn; t++) {
                    double foo = (double)(rt_ptr[t]) * (double)(temp[t]) + (double)((x_hzr + i * mn)[t]);
                    // foo = (double)(temp[t]) + foo;
                    temp[t] = tanh(foo);
                }

                acc4 += dsecnd() - tic;
                // h_t = (1. - z_t) * h_tm1 + z_t * can_h_t
                mn = batch_size * %(hid)s;
                %(d)s* z_ptr = NULL;
                if (%(return_sequences)s) {
                    z_ptr = (%(d)s*) PyArray_DATA(%(z)s) + i * batch_size * %(hid)s;
                } else {
                    z_ptr = (%(d)s*) PyArray_DATA(%(z)s);
                }

                // update
                tic = dsecnd();
                #pragma omp parallel for
                for (int j = 0; j < mn; j++) {
                    z_ptr[j] = (%(d)s)( ((double)1.0 - (double)(zt_ptr[j])) * (double)(A[0][j]) + (double)(zt_ptr[j]) * (double)(temp[j]));
                }
                acc5 += dsecnd() - tic;

                B[0] = z_ptr;
                B[1] = A[0];
                B[2] = A[0];
            }
            // printf(\"loop: %%.8f, %%.8f, %%.8f, %%.8f, %%.8f\\n\", acc1 * 1000.0, acc2 * 1000.0, acc3 * 1000.0, acc4 * 1000.0, acc5 * 1000.0);

            gru_fail:
            Py_XDECREF(x_src);
            Py_XDECREF(w_x_src);
            Py_XDECREF(w_h_src);
            Py_XDECREF(b_src);
            Py_XDECREF(hinit_src);
        """ % locals()
        return ccode

    def grad(self, inp, grads):
        raise NotImplemented('GRU_Inference: gradient function is not implemented.')

    def c_code_cache_version(self):
        return (1, 0, 1)
