#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

#define VOCAB "0123456789+-="
#define VOCAB_SIZE 13
#define VOCAB_SIZE_PAD 16
#define EMBED 16
#define HIDDEN 64
#define MAX_SEQ 32
#define BATCH 16
#define EPOCHS 200000
#define DIGITS 3
#define LR 0.05f
#define ADAM_LR 0.005f

float randf() { return (float)((double)rand() / (double)RAND_MAX - 0.5) * 0.1f; }

typedef struct {
    float *E;
    float *Wz, *Uz, *bz;
    float *Wr, *Ur, *br;
    float *Wh, *Uh, *bh;
    float *Wy, *by;
} GRU;

GRU* gru_new() {
    GRU* g = calloc(1, sizeof(GRU));
    size_t g_size   = VOCAB_SIZE * EMBED;
    size_t wh_size  = EMBED * HIDDEN;
    size_t uh_size  = HIDDEN * HIDDEN;
    size_t b_size   = HIDDEN;
    size_t wy_size  = HIDDEN * VOCAB_SIZE_PAD;
    size_t by_size  = VOCAB_SIZE_PAD;
    size_t total    = g_size + 3*wh_size + 3*uh_size + 3*b_size + wy_size + by_size;
    float* arena = calloc(total, sizeof(float));
    for (size_t i = 0; i < total; i++) arena[i] = randf();
    float* p = arena;
    g->E  = p; p += g_size;
    g->Wz = p; p += wh_size;
    g->Uz = p; p += uh_size;
    g->Wr = p; p += wh_size;
    g->Ur = p; p += uh_size;
    g->Wh = p; p += wh_size;
    g->Uh = p; p += uh_size;
    g->Wy = p; p += wy_size;
    g->bz = p; p += b_size;
    g->br = p; p += b_size;
    g->bh = p; p += b_size;
    g->by = p; p += by_size;
    memset(g->bz, 0, b_size * sizeof(float));
    memset(g->br, 0, b_size * sizeof(float));
    memset(g->bh, 0, b_size * sizeof(float));
    memset(g->by, 0, by_size * sizeof(float));
    return g;
}

int to_idx(char c) {
    char* p = strchr(VOCAB, c);
    return p ? (int)(p - VOCAB) : 0;
}

static inline float sgmd(float x) { return 1.0f / (1.0f + expf(-x)); }

void softmax(float* x, int n) {
    float m = x[0], s = 0;
    for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= (s + 1e-9f);
}

int gen_prob(int* seq) {
    int a = rand() % (int)pow(10, DIGITS);
    int b = rand() % (int)pow(10, DIGITS);
    int add = rand() % 2;
    int c = add ? a + b : abs(a - b);
    if (!add && b > a) { int t = a; a = b; b = t; c = a - b; }
    char s1[16], s2[16], sr[16];
    sprintf(s1, "%d", a);
    sprintf(s2, "%d", b);
    sprintf(sr, "%d", c);
    int l1 = strlen(s1), l2 = strlen(s2), lr = strlen(sr);
    int len = l1 + 1 + l2 + 1 + lr + 1, idx = 0;
    for (int i = l1-1; i >= 0; i--) seq[idx++] = to_idx(s1[i]);
    seq[idx++] = to_idx(add ? '+' : '-');
    for (int i = l2-1; i >= 0; i--) seq[idx++] = to_idx(s2[i]);
    seq[idx++] = to_idx('=');
    for (int i = lr-1; i >= 0; i--) seq[idx++] = to_idx(sr[i]);
    seq[idx++] = to_idx('=');
    return len;
}

//forward GRU step with fused z+r gate computation
//weights stored as [rows][HIDDEN] in column-major-for-SIMD layout:
//Wx[j][i] = Wx[j*HIDDEN + i]  contiguous across i
void gru_step(GRU* g, int token, float* h, float* z, float* r, float* n) {
    float x[EMBED];
    memcpy(x, &g->E[token * EMBED], sizeof(x));

    //fused z = sigmoid(Wz@x + Uz@h + bz), r = sigmoid(Wr@x + Ur@h + br)
    for (int off = 0; off < HIDDEN; off += 16) {
        __m512 yz = _mm512_loadu_ps(g->bz + off);
        __m512 yr = _mm512_loadu_ps(g->br + off);
        for (int j = 0; j < EMBED; j++) {
            __m512 xj = _mm512_set1_ps(x[j]);
            yz = _mm512_fmadd_ps(_mm512_loadu_ps(&g->Wz[j * HIDDEN + off]), xj, yz);
            yr = _mm512_fmadd_ps(_mm512_loadu_ps(&g->Wr[j * HIDDEN + off]), xj, yr);
        }
        for (int j = 0; j < HIDDEN; j++) {
            __m512 hj = _mm512_set1_ps(h[j]);
            yz = _mm512_fmadd_ps(_mm512_loadu_ps(&g->Uz[j * HIDDEN + off]), hj, yz);
            yr = _mm512_fmadd_ps(_mm512_loadu_ps(&g->Ur[j * HIDDEN + off]), hj, yr);
        }
        float tmpz[16], tmpr[16];
        _mm512_storeu_ps(tmpz, yz);
        _mm512_storeu_ps(tmpr, yr);
        for (int i = 0; i < 16; i++) { tmpz[i] = sgmd(tmpz[i]); tmpr[i] = sgmd(tmpr[i]); }
        _mm512_storeu_ps(z + off, _mm512_loadu_ps(tmpz));
        _mm512_storeu_ps(r + off, _mm512_loadu_ps(tmpr));
    }

    //compute r⊙h (element-wise) for the n gate
    float rh[HIDDEN];
    for (int off = 0; off < HIDDEN; off += 16) {
        __m512 rv = _mm512_loadu_ps(r + off);
        __m512 hv = _mm512_loadu_ps(h + off);
        _mm512_storeu_ps(rh + off, _mm512_mul_ps(rv, hv));
    }

    //n = tanh(Wh@x + Uh@(r⊙h) + bh)
    for (int off = 0; off < HIDDEN; off += 16) {
        __m512 y = _mm512_loadu_ps(g->bh + off);
        for (int j = 0; j < EMBED; j++) {
            __m512 w = _mm512_loadu_ps(&g->Wh[j * HIDDEN + off]);
            y = _mm512_fmadd_ps(w, _mm512_set1_ps(x[j]), y);
        }
        for (int j = 0; j < HIDDEN; j++) {
            __m512 u = _mm512_loadu_ps(&g->Uh[j * HIDDEN + off]);
            y = _mm512_fmadd_ps(u, _mm512_set1_ps(rh[j]), y);
        }
        float tmp[16];
        _mm512_storeu_ps(tmp, y);
        for (int i = 0; i < 16; i++) tmp[i] = tanhf(tmp[i]);
        _mm512_storeu_ps(n + off, _mm512_loadu_ps(tmp));
    }

    //h = z⊙h + (1-z)⊙n
    for (int off = 0; off < HIDDEN; off += 16) {
        __m512 zv = _mm512_loadu_ps(z + off);
        __m512 hv = _mm512_loadu_ps(h + off);
        __m512 nv = _mm512_loadu_ps(n + off);
        __m512 one = _mm512_set1_ps(1.0f);
        __m512 new_h = _mm512_add_ps(
            _mm512_mul_ps(zv, hv),
            _mm512_mul_ps(_mm512_sub_ps(one, zv), nv));
        _mm512_storeu_ps(h + off, new_h);
    }
}

float forward(GRU* g, int* seq, int len, int ans_start,
              float* hc, float* zc, float* rc, float* nc, float* yc) {
    float h[HIDDEN] = {0};
    float loss = 0;
    for (int t = 0; t < len; t++) {
        gru_step(g, seq[t], h, &zc[t * HIDDEN], &rc[t * HIDDEN], &nc[t * HIDDEN]);
        memcpy(&hc[t * HIDDEN], h, sizeof(h));

        //output layer: y = by + Wy^T @ h vectorized with padding
        float ypad[VOCAB_SIZE_PAD] = {0};
        for (int off = 0; off < VOCAB_SIZE_PAD; off += 16) {
            __m512 yv = _mm512_loadu_ps(g->by + off);
            for (int j = 0; j < HIDDEN; j++) {
                __m512 w = _mm512_loadu_ps(&g->Wy[j * VOCAB_SIZE_PAD + off]);
                yv = _mm512_fmadd_ps(_mm512_set1_ps(h[j]), w, yv);
            }
            _mm512_storeu_ps(ypad + off, yv);
        }
        float y[VOCAB_SIZE];
        for (int i = 0; i < VOCAB_SIZE; i++) y[i] = ypad[i];
        softmax(y, VOCAB_SIZE);
        memcpy(&yc[t * VOCAB_SIZE], y, sizeof(y));

        if (t >= ans_start && t < len - 1) {
            int tgt = seq[t+1];
            loss += -logf(y[tgt] + 1e-10f);
        }
    }
    return loss;
}

//y += W @ dy where W is [cols][HIDDEN] stored as W[col*HIDDEN + row]
//vectorized per-row dot product (contiguous row access)
static void backprop_mv(const float* W, int cols, const float* dy, float* y) {
    for (int j = 0; j < cols; j++) {
        __m512 acc = _mm512_setzero_ps();
        for (int c = 0; c < 4; c++) {
            __m512 w = _mm512_loadu_ps(&W[j * HIDDEN + c * 16]);
            __m512 d = _mm512_loadu_ps(&dy[c * 16]);
            acc = _mm512_fmadd_ps(w, d, acc);
        }
        y[j] += _mm512_reduce_add_ps(acc);
    }
}

void backward(GRU* g, int* seq, int len, int ans_start,
              float* hc, float* zc, float* rc, float* nc, float* yc,
              float* gE, float* gWz, float* gUz, float* gbz,
              float* gWr, float* gUr, float* gbr,
              float* gWh, float* gUh, float* gbh,
              float* gWy, float* gby) {
    float dh[HIDDEN] = {0};
    for (int t = len - 1; t >= 0; t--) {
        float h[HIDDEN]; memcpy(h, &hc[t * HIDDEN], sizeof(h));
        float z[HIDDEN]; memcpy(z, &zc[t * HIDDEN], sizeof(z));
        float r[HIDDEN]; memcpy(r, &rc[t * HIDDEN], sizeof(r));
        float n[HIDDEN]; memcpy(n, &nc[t * HIDDEN], sizeof(n));
        float x_emb[EMBED]; memcpy(x_emb, &g->E[seq[t]*EMBED], sizeof(x_emb));

        float dy[VOCAB_SIZE_PAD];
        if (t >= ans_start && t < len - 1) {
            int tgt = seq[t+1];
            memcpy(dy, &yc[t * VOCAB_SIZE], VOCAB_SIZE * sizeof(float));
            dy[tgt] -= 1.0f;
            memset(dy + VOCAB_SIZE, 0, (VOCAB_SIZE_PAD - VOCAB_SIZE) * sizeof(float));
        } else memset(dy, 0, sizeof(dy));

        //output layer: gradient w.r.t. Wy, by and backprop to dh
        for (int j = 0; j < HIDDEN; j++) {
            __m512 sumv = _mm512_setzero_ps();
            __m512 hj = _mm512_set1_ps(h[j]);
            for (int off = 0; off < VOCAB_SIZE_PAD; off += 16) {
                __m512 dyv = _mm512_loadu_ps(dy + off);
                __m512 wv = _mm512_loadu_ps(&g->Wy[j * VOCAB_SIZE_PAD + off]);
                sumv = _mm512_fmadd_ps(dyv, wv, sumv);
                __m512 gv = _mm512_loadu_ps(&gWy[j * VOCAB_SIZE_PAD + off]);
                gv = _mm512_fmadd_ps(hj, dyv, gv);
                _mm512_storeu_ps(&gWy[j * VOCAB_SIZE_PAD + off], gv);
            }
            dh[j] += _mm512_reduce_add_ps(sumv);
        }
        for (int i = 0; i < VOCAB_SIZE; i++) gby[i] += dy[i];

        float h_prev[HIDDEN];
        if (t == 0) memset(h_prev, 0, sizeof(h_prev));
        else memcpy(h_prev, &hc[(t-1) * HIDDEN], sizeof(h_prev));

        float dn[HIDDEN], dz[HIDDEN], dpre_n[HIDDEN], d_h_prev[HIDDEN], rh_prev[HIDDEN];
        for (int off = 0; off < HIDDEN; off += 16) {
            __m512 dhv = _mm512_loadu_ps(dh + off);
            __m512 zv  = _mm512_loadu_ps(z + off);
            __m512 nv  = _mm512_loadu_ps(n + off);
            __m512 hpv = _mm512_loadu_ps(h_prev + off);
            __m512 one = _mm512_set1_ps(1.0f);
            __m512 dnv = _mm512_mul_ps(dhv, _mm512_sub_ps(one, zv));
            __m512 dzv = _mm512_mul_ps(dhv, _mm512_sub_ps(hpv, nv));
            _mm512_storeu_ps(dn + off, dnv);
            _mm512_storeu_ps(dz + off, dzv);
            _mm512_storeu_ps(dpre_n + off, _mm512_mul_ps(dnv, _mm512_sub_ps(one, _mm512_mul_ps(nv, nv))));
            _mm512_storeu_ps(d_h_prev + off, _mm512_mul_ps(dhv, zv));
            __m512 rv = _mm512_loadu_ps(r + off);
            _mm512_storeu_ps(rh_prev + off, _mm512_mul_ps(rv, hpv));
        }

        //Uh gradient: vectorized outer product (contiguous row writes)
        for (int k = 0; k < HIDDEN; k++) {
            __m512 rh = _mm512_set1_ps(rh_prev[k]);
            for (int off = 0; off < HIDDEN; off += 16) {
                __m512 g = _mm512_loadu_ps(&gUh[k * HIDDEN + off]);
                __m512 d = _mm512_loadu_ps(&dpre_n[off]);
                _mm512_storeu_ps(&gUh[k * HIDDEN + off], _mm512_fmadd_ps(rh, d, g));
            }
        }

        //Wh gradient: gWh[j][chunk] += dpre_n[chunk] * x[j]
        for (int j = 0; j < EMBED; j++) {
            float sc = x_emb[j];
            for (int off = 0; off < HIDDEN; off += 16) {
                __m512 g = _mm512_loadu_ps(&gWh[j * HIDDEN + off]);
                __m512 d = _mm512_loadu_ps(&dpre_n[off]);
                g = _mm512_fmadd_ps(_mm512_set1_ps(sc), d, g);
                _mm512_storeu_ps(&gWh[j * HIDDEN + off], g);
            }
        }
        for (int off = 0; off < HIDDEN; off += 16) {
            __m512 d = _mm512_loadu_ps(&dpre_n[off]);
            __m512 g = _mm512_loadu_ps(gbh + off);
            _mm512_storeu_ps(gbh + off, _mm512_add_ps(g, d));
        }

        //d_h_prev from Uh: r[i] * sum_j Uh[i][j] * dpre_n[j]
        float uh_contrib[HIDDEN] = {0};
        backprop_mv(g->Uh, HIDDEN, dpre_n, uh_contrib);
        float dr[HIDDEN];
        for (int off = 0; off < HIDDEN; off += 16) {
            __m512 uc = _mm512_loadu_ps(uh_contrib + off);
            __m512 rv = _mm512_loadu_ps(r + off);
            __m512 hpv = _mm512_loadu_ps(h_prev + off);
            __m512 dhv = _mm512_loadu_ps(d_h_prev + off);
            _mm512_storeu_ps(d_h_prev + off, _mm512_fmadd_ps(uc, rv, dhv));
            _mm512_storeu_ps(dr + off, _mm512_mul_ps(uc, hpv));
        }

        //z gate
        float dpre_z[HIDDEN];
        for (int off = 0; off < HIDDEN; off += 16) {
            __m512 dzv = _mm512_loadu_ps(dz + off);
            __m512 zv  = _mm512_loadu_ps(z + off);
            __m512 one = _mm512_set1_ps(1.0f);
            _mm512_storeu_ps(dpre_z + off, _mm512_mul_ps(dzv, _mm512_mul_ps(zv, _mm512_sub_ps(one, zv))));
        }
        backprop_mv(g->Uz, HIDDEN, dpre_z, d_h_prev);
        for (int off = 0; off < HIDDEN; off += 16) {
            __m512 d = _mm512_loadu_ps(&dpre_z[off]);
            for (int j = 0; j < EMBED; j++) {
                __m512 g = _mm512_loadu_ps(&gWz[j * HIDDEN + off]);
                g = _mm512_fmadd_ps(_mm512_set1_ps(x_emb[j]), d, g);
                _mm512_storeu_ps(&gWz[j * HIDDEN + off], g);
            }
            for (int j = 0; j < HIDDEN; j++) {
                __m512 g = _mm512_loadu_ps(&gUz[j * HIDDEN + off]);
                g = _mm512_fmadd_ps(_mm512_set1_ps(h_prev[j]), d, g);
                _mm512_storeu_ps(&gUz[j * HIDDEN + off], g);
            }
            float tmp[16]; _mm512_storeu_ps(tmp, d);
            for (int k = 0; k < 16; k++) gbz[off + k] += tmp[k];
        }

        //r gate
        float dpre_r[HIDDEN];
        for (int off = 0; off < HIDDEN; off += 16) {
            __m512 drv = _mm512_loadu_ps(dr + off);
            __m512 rv  = _mm512_loadu_ps(r + off);
            __m512 one = _mm512_set1_ps(1.0f);
            _mm512_storeu_ps(dpre_r + off, _mm512_mul_ps(drv, _mm512_mul_ps(rv, _mm512_sub_ps(one, rv))));
        }
        backprop_mv(g->Ur, HIDDEN, dpre_r, d_h_prev);
        for (int off = 0; off < HIDDEN; off += 16) {
            __m512 d = _mm512_loadu_ps(&dpre_r[off]);
            for (int j = 0; j < EMBED; j++) {
                __m512 g = _mm512_loadu_ps(&gWr[j * HIDDEN + off]);
                g = _mm512_fmadd_ps(_mm512_set1_ps(x_emb[j]), d, g);
                _mm512_storeu_ps(&gWr[j * HIDDEN + off], g);
            }
            for (int j = 0; j < HIDDEN; j++) {
                __m512 g = _mm512_loadu_ps(&gUr[j * HIDDEN + off]);
                g = _mm512_fmadd_ps(_mm512_set1_ps(h_prev[j]), d, g);
                _mm512_storeu_ps(&gUr[j * HIDDEN + off], g);
            }
            float tmp[16]; _mm512_storeu_ps(tmp, d);
            for (int k = 0; k < 16; k++) gbr[off + k] += tmp[k];
        }

        //embedding gradient
        float dx[EMBED] = {0};
        backprop_mv(g->Wz, EMBED, dpre_z, dx);
        backprop_mv(g->Wr, EMBED, dpre_r, dx);
        backprop_mv(g->Wh, EMBED, dpre_n, dx);
        for (int k = 0; k < EMBED; k++)
            gE[seq[t]*EMBED + k] += dx[k];

        memcpy(dh, d_h_prev, sizeof(dh));
    }
}

void params(GRU* g) {
    int n = VOCAB_SIZE*EMBED + 3*(EMBED*HIDDEN + HIDDEN*HIDDEN + HIDDEN) + HIDDEN*VOCAB_SIZE + VOCAB_SIZE;
    printf("Params: %d (%.1fK)\n", n, n/1000.0f);
}

void solve(GRU* g, int a, int b, char op) {
    char s1[16], s2[16];
    sprintf(s1, "%d", a);
    sprintf(s2, "%d", b);
    int l1 = strlen(s1), l2 = strlen(s2);
    int seq[32], idx = 0;
    for (int i = l1-1; i >= 0; i--) seq[idx++] = to_idx(s1[i]);
    seq[idx++] = to_idx(op);
    for (int i = l2-1; i >= 0; i--) seq[idx++] = to_idx(s2[i]);
    seq[idx++] = to_idx('=');
    int pl = idx;
    float h[HIDDEN] = {0};
    float z[HIDDEN], r[HIDDEN], n[HIDDEN];
    for (int i = 0; i < pl; i++) gru_step(g, seq[i], h, z, r, n);
    printf("%d %c %d = ", a, op, b);
    char out_buf[16];
    int out_len = 0;
    for (int i = 0; i < 10; i++) {
        //output layer: y = by + Wy^T @ h vectorized with padding
        float ypad[VOCAB_SIZE_PAD] = {0};
        for (int off = 0; off < VOCAB_SIZE_PAD; off += 16) {
            __m512 yv = _mm512_loadu_ps(g->by + off);
            for (int j = 0; j < HIDDEN; j++) {
                __m512 w = _mm512_loadu_ps(&g->Wy[j * VOCAB_SIZE_PAD + off]);
                yv = _mm512_fmadd_ps(_mm512_set1_ps(h[j]), w, yv);
            }
            _mm512_storeu_ps(ypad + off, yv);
        }
        float y[VOCAB_SIZE];
        for (int k = 0; k < VOCAB_SIZE; k++) y[k] = ypad[k];
        softmax(y, VOCAB_SIZE);
        int best = 0;
        for (int j = 1; j < VOCAB_SIZE; j++) if (y[j] > y[best]) best = j;
        if (best == to_idx('=')) break;
        out_buf[out_len++] = VOCAB[best];
        gru_step(g, best, h, z, r, n);
    }
    for (int i = out_len - 1; i >= 0; i--) printf("%c", out_buf[i]);
    printf("\n");
}

int main() {
    setbuf(stdout, NULL);
    srand(12345);
    GRU* g = gru_new();
    params(g);
    printf("Digits: %d, Hidden: %d, Embed: %d [AVX-512]\n", DIGITS, HIDDEN, EMBED);
    printf("Training %d epochs, batch=%d...\n", EPOCHS, BATCH);

    float *hc = malloc(MAX_SEQ * HIDDEN * sizeof(float));
    float *zc = malloc(MAX_SEQ * HIDDEN * sizeof(float));
    float *rc = malloc(MAX_SEQ * HIDDEN * sizeof(float));
    float *nc = malloc(MAX_SEQ * HIDDEN * sizeof(float));
    float *yc = malloc(MAX_SEQ * VOCAB_SIZE * sizeof(float));

    size_t g_size   = VOCAB_SIZE * EMBED;
    size_t wh_size  = EMBED * HIDDEN;
    size_t uh_size  = HIDDEN * HIDDEN;
    size_t b_size   = HIDDEN;
    size_t wy_size  = HIDDEN * VOCAB_SIZE_PAD;
    size_t by_size  = VOCAB_SIZE_PAD;
    size_t total    = g_size + 3*wh_size + 3*uh_size + 3*b_size + wy_size + by_size;

    //arena for all gradient/optimizer arrays
    float* grad_arena = calloc(total * 4, sizeof(float));
    float* g_base  = grad_arena;
    float* m_base  = grad_arena + total;
    float* s_base  = grad_arena + total * 2;
    float* sg_base = grad_arena + total * 3;

    size_t off = 0;
    float *gE  = g_base + off; float *mE  = m_base + off; float *sE  = s_base + off; float *sgE  = sg_base + off; off += g_size;
    float *gWz = g_base + off; float *mWz = m_base + off; float *sWz = s_base + off; float *sgWz = sg_base + off; off += wh_size;
    float *gUz = g_base + off; float *mUz = m_base + off; float *sUz = s_base + off; float *sgUz = sg_base + off; off += uh_size;
    float *gbz = g_base + off; float *mbz = m_base + off; float *sbz = s_base + off; float *sgbz = sg_base + off; off += b_size;
    float *gWr = g_base + off; float *mWr = m_base + off; float *sWr = s_base + off; float *sgWr = sg_base + off; off += wh_size;
    float *gUr = g_base + off; float *mUr = m_base + off; float *sUr = s_base + off; float *sgUr = sg_base + off; off += uh_size;
    float *gbr = g_base + off; float *mbr = m_base + off; float *sbr = s_base + off; float *sgbr = sg_base + off; off += b_size;
    float *gWh = g_base + off; float *mWh = m_base + off; float *sWh = s_base + off; float *sgWh = sg_base + off; off += wh_size;
    float *gUh = g_base + off; float *mUh = m_base + off; float *sUh = s_base + off; float *sgUh = sg_base + off; off += uh_size;
    float *gbh = g_base + off; float *mbh = m_base + off; float *sbh = s_base + off; float *sgbh = sg_base + off; off += b_size;
    float *gWy = g_base + off; float *mWy = m_base + off; float *sWy = s_base + off; float *sgWy = sg_base + off; off += wy_size;
    float *gby = g_base + off; float *mby = m_base + off; float *sby = s_base + off; float *sgby = sg_base + off; off += by_size;

    float grok_alpha = 0.98f, grok_lamb = 1.0f;
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f, adam_lr = (float)ADAM_LR, wd = 1e-4f;
    float max_lr = adam_lr, min_lr = adam_lr * 0.01f;
    int adam_step = 0;
    float b1t = 1.0f, b2t = 1.0f;

    for (int ep = 0; ep < EPOCHS; ep++) {
        double loss_sum = 0;
        memset(g_base, 0, total * sizeof(float));

        for (int b = 0; b < BATCH; b++) {
            int seq[MAX_SEQ], len, ans_start;
            len = gen_prob(seq);
            ans_start = 0;
            while (ans_start < len && seq[ans_start] != to_idx('=')) ans_start++;
            float loss = forward(g, seq, len, ans_start, hc, zc, rc, nc, yc);
            backward(g, seq, len, ans_start, hc, zc, rc, nc, yc,
                     gE, gWz, gUz, gbz, gWr, gUr, gbr, gWh, gUh, gbh, gWy, gby);
            loss_sum += loss;
        }

        //compute gradient norm and clip raw gradients
        float gn = 0;
        for (int i = 0; i < VOCAB_SIZE*EMBED; i++) gn += gE[i]*gE[i];
        for (int i = 0; i < EMBED*HIDDEN; i++) gn += gWz[i]*gWz[i] + gWr[i]*gWr[i] + gWh[i]*gWh[i];
        for (int i = 0; i < HIDDEN*HIDDEN; i++) gn += gUz[i]*gUz[i] + gUr[i]*gUr[i] + gUh[i]*gUh[i];
        for (int i = 0; i < HIDDEN; i++) gn += gbz[i]*gbz[i] + gbr[i]*gbr[i] + gbh[i]*gbh[i];
        for (int i = 0; i < HIDDEN*VOCAB_SIZE_PAD; i++) gn += gWy[i]*gWy[i];
        for (int i = 0; i < VOCAB_SIZE_PAD; i++) gn += gby[i]*gby[i];
        gn = sqrtf(gn / BATCH);
        float clip_scale = 1.0f;
        if (gn > 5.0f) clip_scale = 5.0f / gn;
        
        //technique from "Grokfast: Accelerated Grokking by Amplifying Slow Gradients"
#define GROK_P(g, sg, n) do { for (int i = 0; i < n; i++) { float gc = g[i] * clip_scale; sg[i] = grok_alpha * sg[i] + (1.0f - grok_alpha) * gc; g[i] = gc + grok_lamb * sg[i]; } } while(0)
        GROK_P(gE,  sgE, VOCAB_SIZE*EMBED);
        GROK_P(gWz, sgWz, EMBED*HIDDEN);
        GROK_P(gUz, sgUz, HIDDEN*HIDDEN);
        GROK_P(gbz, sgbz, HIDDEN);
        GROK_P(gWr, sgWr, EMBED*HIDDEN);
        GROK_P(gUr, sgUr, HIDDEN*HIDDEN);
        GROK_P(gbr, sgbr, HIDDEN);
        GROK_P(gWh, sgWh, EMBED*HIDDEN);
        GROK_P(gUh, sgUh, HIDDEN*HIDDEN);
        GROK_P(gbh, sgbh, HIDDEN);
        GROK_P(gWy, sgWy, HIDDEN*VOCAB_SIZE_PAD);
        GROK_P(gby, sgby, VOCAB_SIZE_PAD);
#undef GROK_P

        adam_step++;
        float progress = (float)adam_step / (float)(EPOCHS);
        adam_lr = min_lr + 0.5f * (max_lr - min_lr) * (1.0f + cosf(progress * (float)M_PI));
        b1t *= beta1; b2t *= beta2;
        float b1c = 1.0f - b1t;
        float b2c = 1.0f - b2t;
#define ADAM_P(p, g, m, s, n) do { for (int i = 0; i < n; i++) { m[i] = beta1*m[i] + (1-beta1)*g[i]; s[i] = beta2*s[i] + (1-beta2)*g[i]*g[i]; p[i] -= adam_lr * ((m[i]/b1c) / (sqrtf(s[i]/b2c) + eps) + wd * p[i]); } } while(0)
        ADAM_P(g->E,  gE, mE, sE, VOCAB_SIZE*EMBED);
        ADAM_P(g->Wz, gWz, mWz, sWz, EMBED*HIDDEN);
        ADAM_P(g->Uz, gUz, mUz, sUz, HIDDEN*HIDDEN);
        ADAM_P(g->bz, gbz, mbz, sbz, HIDDEN);
        ADAM_P(g->Wr, gWr, mWr, sWr, EMBED*HIDDEN);
        ADAM_P(g->Ur, gUr, mUr, sUr, HIDDEN*HIDDEN);
        ADAM_P(g->br, gbr, mbr, sbr, HIDDEN);
        ADAM_P(g->Wh, gWh, mWh, sWh, EMBED*HIDDEN);
        ADAM_P(g->Uh, gUh, mUh, sUh, HIDDEN*HIDDEN);
        ADAM_P(g->bh, gbh, mbh, sbh, HIDDEN);
        ADAM_P(g->Wy, gWy, mWy, sWy, HIDDEN*VOCAB_SIZE_PAD);
        ADAM_P(g->by, gby, mby, sby, VOCAB_SIZE_PAD);
#undef ADAM_P

        if (ep % 1000 == 0) {
            printf("epoch %d: loss=%.4f  ", ep, (float)(loss_sum / BATCH));
            solve(g, 12+ep%10, 3+ep%5, '+');
            solve(g, 45+ep%10, 12+ep%3, '-');
        }
    }

    printf("\nFinal tests:\n");
    for (int a = 0; a <= 20; a += 5)
        for (int b = 0; b <= 10; b += 3)
            if (a + b > 0) solve(g, a, b, '+');
    printf("\n3-digit tests:\n");
    solve(g, 123, 456, '+');
    solve(g, 789, 100, '+');
    solve(g, 500, 499, '+');
    solve(g, 999, 1, '+');
    solve(g, 456, 123, '-');
    solve(g, 800, 300, '-');
    solve(g, 999, 1, '-');
    return 0;
}
