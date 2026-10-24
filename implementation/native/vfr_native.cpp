// vfr_native.cpp -- polynomial-level CKKS and BFV operations for protocol v3.1.
//
// Built on OpenFHE 1.5.1 (DCRTPoly arithmetic, NTTs, hybrid key switching, BFV evaluation).
// Everything that the specification fixes at the polynomial level is implemented here by hand:
//   * threshold key shares with a common random string (CRS) expanded by SHAKE-256,
//     joint rotation keys sum of two shares (Mouchet et al.), no relinearization key for CKKS;
//   * R's seeded symmetric upload c0 = -a s_R + e + Ecd(v), a = PRG(seed); O's conversion c0 -= a s_O;
//   * a CKKS encoder/decoder in double-double precision with an exact, explicitly chosen scale
//     (scales may exceed 2^60; they are tracked exactly, never as nominal powers of two);
//   * ct-pt products, additions, deterministic or stochastic rescaling (c - r)/q;
//   * EvalSum with explicit joint keys (OpenFHE's hybrid key switch);
//   * the release: re-randomization under the joint pk with fresh (u, e0, e1) and the noisy partial
//     decryption h_O = c0' + c1' s_O + eta with eta supplied by the caller (exact sampler in Python);
//   * R's combination v = h_O + c1' s_R, CRT to int128 and decoding with the exact final scale;
//   * BFV labeled-PSI evaluation (windowed powers, membership/label polynomials, masks) and
//     SanitizeBFV (modulus switch, + Enc_pk(0), + flooding noise supplied by the caller).
#include "openfhe.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/bfvrns/bfvrns-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"

#include <openssl/evp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <quadmath.h>
#include <omp.h>

#include <cmath>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace lbcrypto;
using i128 = __int128;
using u128 = unsigned __int128;
using ParamsT = DCRTPoly::Params;

// =====================================================================================
// double-double arithmetic (Dekker / QD library algorithms)
// =====================================================================================
struct dd {
    double hi, lo;
};
static inline dd two_sum(double a, double b) {
    double s = a + b, bb = s - a;
    return {s, (a - (s - bb)) + (b - bb)};
}
static inline dd quick_two_sum(double a, double b) {
    double s = a + b;
    return {s, b - (s - a)};
}
static inline dd two_prod(double a, double b) {
    double p = a * b;
    return {p, std::fma(a, b, -p)};
}
static inline dd dd_add(dd a, dd b) {
    dd s = two_sum(a.hi, b.hi), t = two_sum(a.lo, b.lo);
    s.lo += t.hi;
    s = quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    return quick_two_sum(s.hi, s.lo);
}
static inline dd dd_neg(dd a) { return {-a.hi, -a.lo}; }
static inline dd dd_sub(dd a, dd b) { return dd_add(a, dd_neg(b)); }
static inline dd dd_mul(dd a, dd b) {
    dd p = two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    return quick_two_sum(p.hi, p.lo);
}
static inline dd dd_mul_d(dd a, double b) {
    dd p = two_prod(a.hi, b);
    p.lo += a.lo * b;
    return quick_two_sum(p.hi, p.lo);
}
static inline dd dd_div(dd a, dd b) {
    double q1 = a.hi / b.hi;
    dd r      = dd_sub(a, dd_mul_d(b, q1));
    double q2 = r.hi / b.hi;
    r         = dd_sub(r, dd_mul_d(b, q2));
    double q3 = r.hi / b.hi;
    return dd_add(quick_two_sum(q1, q2), dd{q3, 0.0});
}
static inline dd dd_from_q(__float128 x) {
    double hi = (double)x;
    double lo = (double)(x - (__float128)hi);
    return {hi, lo};
}
static inline dd dd_from_i128(i128 v) {
    double hi = (double)v;
    i128 r1   = v - (i128)hi;
    double mid = (double)r1;
    i128 r2   = r1 - (i128)mid;
    return dd_add(dd_add(dd{hi, 0.0}, dd{mid, 0.0}), dd{(double)r2, 0.0});
}
// round to nearest integer (ties to even on the leading part), exact for |x| < 2^120
static inline i128 dd_round_i128(dd x) {
    double r  = std::nearbyint(x.hi);
    dd rem    = dd_add(dd{x.hi - r, 0.0}, dd{x.lo, 0.0});  // x.hi - r is exact
    double r2 = std::nearbyint(rem.hi + rem.lo);
    return (i128)r + (i128)r2;
}
struct cdd {
    dd re, im;
};
static inline cdd cadd(cdd a, cdd b) { return {dd_add(a.re, b.re), dd_add(a.im, b.im)}; }
static inline cdd csub(cdd a, cdd b) { return {dd_sub(a.re, b.re), dd_sub(a.im, b.im)}; }
static inline cdd cmul(cdd a, cdd b) {
    return {dd_sub(dd_mul(a.re, b.re), dd_mul(a.im, b.im)), dd_add(dd_mul(a.re, b.im), dd_mul(a.im, b.re))};
}
static inline cdd cconj(cdd a) { return {a.re, dd_neg(a.im)}; }

// =====================================================================================
// CKKS encoder in double-double precision.
// tau(f)_k = f(zeta^{h_k}), zeta = exp(i pi / N), h_k = 5^k mod 2N, k < N/2   (proof conventions)
// Ecd_Omega(z) = round(Omega * tau^{-1}(z)),  Dcd(v) = tau(v) / Omega.
// =====================================================================================
class Encoder {
public:
    uint32_t N, logN;
    std::vector<uint32_t> rot_m, conj_m, bitrev;
    std::vector<cdd> W;      // exp(2 pi i k / N), k < N/2
    std::vector<cdd> twist;  // exp(i pi n / N), n < N

    explicit Encoder(uint32_t N_) : N(N_) {
        logN = 0;
        while ((1u << logN) < N) ++logN;
        uint32_t M = 2 * N;
        rot_m.resize(N / 2);
        conj_m.resize(N / 2);
        uint64_t h = 1;
        for (uint32_t k = 0; k < N / 2; ++k) {
            rot_m[k]  = (uint32_t)((h - 1) / 2);
            conj_m[k] = (uint32_t)((M - h - 1) / 2);
            h         = (h * 5) % M;
        }
        W.resize(N / 2);
        twist.resize(N);
        const __float128 pi = acosq((__float128)-1);
        for (uint32_t k = 0; k < N / 2; ++k) {
            __float128 ang = 2 * pi * (__float128)k / (__float128)N;
            W[k]           = {dd_from_q(cosq(ang)), dd_from_q(sinq(ang))};
        }
        for (uint32_t n = 0; n < N; ++n) {
            __float128 ang = pi * (__float128)n / (__float128)N;
            twist[n]       = {dd_from_q(cosq(ang)), dd_from_q(sinq(ang))};
        }
        bitrev.resize(N);
        for (uint32_t i = 0; i < N; ++i) {
            uint32_t r = 0;
            for (uint32_t b = 0; b < logN; ++b)
                if (i & (1u << b)) r |= 1u << (logN - 1 - b);
            bitrev[i] = r;
        }
    }

    // in place; sign = +1: X[m] = sum_n x[n] w^{mn}; sign = -1: conjugate roots (no 1/N)
    void fft(std::vector<cdd>& a, int sign) const {
        for (uint32_t i = 0; i < N; ++i)
            if (i < bitrev[i]) std::swap(a[i], a[bitrev[i]]);
        for (uint32_t len = 2; len <= N; len <<= 1) {
            uint32_t half = len / 2, step = N / len;
            for (uint32_t i = 0; i < N; i += len) {
                for (uint32_t j = 0; j < half; ++j) {
                    cdd w = W[j * step];
                    if (sign < 0) w = cconj(w);
                    cdd u = a[i + j], v = cmul(a[i + j + half], w);
                    a[i + j]        = cadd(u, v);
                    a[i + j + half] = csub(u, v);
                }
            }
        }
    }

    // exact-scale encoding of real values (len <= N/2, remaining slots zero) -> integer coefficients
    std::vector<i128> encode(const double* z, size_t len, dd scale) const {
        if (len > N / 2) throw std::runtime_error("encode: too many values");
        std::vector<cdd> Y(N, cdd{{0, 0}, {0, 0}});
        for (size_t k = 0; k < len; ++k) {
            Y[rot_m[k]].re  = dd{z[k], 0.0};
            Y[conj_m[k]].re = dd{z[k], 0.0};
        }
        fft(Y, -1);
        const double invN = 1.0 / (double)N;  // exact power of two
        std::vector<i128> out(N);
        for (uint32_t n = 0; n < N; ++n) {
            // f_n = Re(g_n * conj(twist_n)) / N
            dd f   = dd_add(dd_mul(Y[n].re, twist[n].re), dd_mul(Y[n].im, twist[n].im));
            f      = dd_mul_d(f, invN);
            out[n] = dd_round_i128(dd_mul(f, scale));
        }
        return out;
    }

    // tau(v) / scale for integer coefficients v
    std::vector<std::complex<double>> decode(const std::vector<i128>& v, dd scale) const {
        std::vector<cdd> G(N);
        for (uint32_t n = 0; n < N; ++n) {
            dd x = dd_from_i128(v[n]);
            G[n] = {dd_mul(x, twist[n].re), dd_mul(x, twist[n].im)};
        }
        fft(G, +1);
        std::vector<std::complex<double>> out(N / 2);
        for (uint32_t k = 0; k < N / 2; ++k) {
            const cdd& g = G[rot_m[k]];
            dd re = dd_div(g.re, scale), im = dd_div(g.im, scale);
            out[k] = {re.hi + re.lo, im.hi + im.lo};
        }
        return out;
    }
};

// =====================================================================================
// helpers
// =====================================================================================
static std::vector<uint8_t> shake256(const std::string& input, size_t outlen) {
    std::vector<uint8_t> out(outlen);
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx || EVP_DigestInit_ex(ctx, EVP_shake256(), nullptr) != 1 ||
        EVP_DigestUpdate(ctx, input.data(), input.size()) != 1 || EVP_DigestFinalXOF(ctx, out.data(), outlen) != 1) {
        if (ctx) EVP_MD_CTX_free(ctx);
        throw std::runtime_error("SHAKE-256 failed");
    }
    EVP_MD_CTX_free(ctx);
    return out;
}

static inline uint64_t mod_i128(i128 x, uint64_t q) {
    i128 r = x % (i128)q;
    if (r < 0) r += q;
    return (uint64_t)r;
}
static inline uint64_t mulmod(uint64_t a, uint64_t b, uint64_t q) { return (uint64_t)(((u128)a * b) % q); }
static uint64_t powmod(uint64_t a, uint64_t e, uint64_t q) {
    uint64_t r = 1 % q;
    a %= q;
    while (e) {
        if (e & 1) r = mulmod(r, a, q);
        a = mulmod(a, a, q);
        e >>= 1;
    }
    return r;
}
static uint64_t invmod(uint64_t a, uint64_t q) { return powmod(a, q - 2, q); }  // q prime

// =====================================================================================
// CKKS core
// =====================================================================================
struct SecretShare {
    DCRTPoly s;  // over Q, EVALUATION
};
struct PubKey {
    DCRTPoly p0, a;  // over Q, EVALUATION
};
struct KeyMap {
    std::shared_ptr<std::map<uint32_t, EvalKey<DCRTPoly>>> keys;
};
struct Ctxt {
    Ciphertext<DCRTPoly> ct;
    dd scale;
};
struct Ptxt {
    DCRTPoly m;
    dd scale;
};

class CKKSCore {
public:
    CryptoContext<DCRTPoly> cc;
    std::shared_ptr<CryptoParametersCKKSRNS> cp;
    uint32_t N, M, nQ, nP, dnum, alpha;
    std::shared_ptr<ParamsT> pQ, pQP, pP;
    std::vector<std::shared_ptr<ParamsT>> pk_;  // pk_[k] = params with first k towers of Q
    std::vector<uint64_t> qs, ps;
    std::vector<uint32_t> sum_idx;
    std::unique_ptr<Encoder> enc;
    std::string tag = "vfr-joint";

    CKKSCore(uint32_t logN, uint32_t depth, uint32_t sms, uint32_t fms, uint32_t dnum_, bool sec128) {
        CCParams<CryptoContextCKKSRNS> p;
        p.SetMultiplicativeDepth(depth);
        p.SetScalingModSize(sms);
        p.SetFirstModSize(fms);
        p.SetRingDim(1u << logN);
        p.SetBatchSize(1u << (logN - 1));
        p.SetScalingTechnique(FIXEDMANUAL);
        p.SetKeySwitchTechnique(HYBRID);
        p.SetNumLargeDigits(dnum_);
        p.SetSecretKeyDist(UNIFORM_TERNARY);
        p.SetSecurityLevel(sec128 ? HEStd_128_classic : HEStd_NotSet);
        cc = GenCryptoContext(p);
        cc->Enable(PKE);
        cc->Enable(KEYSWITCH);
        cc->Enable(LEVELEDSHE);
        cc->Enable(MULTIPARTY);
        cp  = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
        pQ  = cp->GetElementParams();
        pQP = cp->GetParamsQP();
        pP  = cp->GetParamsP();
        N   = pQ->GetRingDimension();
        M   = 2 * N;
        nQ  = pQ->GetParams().size();
        nP  = pP->GetParams().size();
        dnum  = cp->GetNumPartQ();
        alpha = cp->GetNumPerPartQ();
        for (auto& t : pQ->GetParams()) qs.push_back(t->GetModulus().ConvertToInt<uint64_t>());
        for (auto& t : pP->GetParams()) ps.push_back(t->GetModulus().ConvertToInt<uint64_t>());
        pk_.resize(nQ + 1);
        for (uint32_t k = 1; k <= nQ; ++k) {
            DCRTPoly tmp(pQ, Format::EVALUATION, true);
            tmp.DropLastElements(nQ - k);
            pk_[k] = tmp.GetParams();
        }
        // EvalSum indices: g_i = 5^{2^i} mod M, i < log2(N/2)  (rotations by 2^i)
        uint64_t g = 5;
        for (uint32_t i = 0; i + 1 < logN; ++i) {
            sum_idx.push_back((uint32_t)g);
            g = (g * g) % M;
        }
        enc = std::make_unique<Encoder>(N);
    }

    std::shared_ptr<ParamsT> params_k(uint32_t k) const {
        if (k < 1 || k > nQ) throw std::runtime_error("bad tower count");
        return pk_[k];
    }

    // ------------------------------------------------------------------ conversions
    py::array_t<uint64_t> to_np(const DCRTPoly& p0) const {
        DCRTPoly p = p0;
        uint32_t k = p.GetNumOfElements();
        py::array_t<uint64_t> out({(size_t)k, (size_t)N});
        auto r = out.mutable_unchecked<2>();
        for (uint32_t i = 0; i < k; ++i) {
            const auto& t = p.GetElementAtIndex(i);
            for (uint32_t n = 0; n < N; ++n) r(i, n) = t[n].ConvertToInt<uint64_t>();
        }
        return out;
    }
    DCRTPoly from_np(py::array_t<uint64_t, py::array::c_style | py::array::forcecast> a,
                     std::shared_ptr<ParamsT> params, Format fmt) const {
        auto r     = a.unchecked<2>();
        uint32_t k = params->GetParams().size();
        if ((uint32_t)r.shape(0) != k || (uint32_t)r.shape(1) != N) throw std::runtime_error("from_np: bad shape");
        DCRTPoly p(params, fmt, true);
        for (uint32_t i = 0; i < k; ++i) {
            auto t        = p.GetElementAtIndex(i);
            uint64_t q    = params->GetParams()[i]->GetModulus().ConvertToInt<uint64_t>();
            for (uint32_t n = 0; n < N; ++n) {
                uint64_t v = r(i, n);
                if (v >= q) throw std::runtime_error("from_np: residue out of range");
                t[n] = NativeInteger(v);
            }
            p.SetElementAtIndex(i, std::move(t));
        }
        return p;
    }
    DCRTPoly from_i128(const std::vector<i128>& c, std::shared_ptr<ParamsT> params) const {
        uint32_t k = params->GetParams().size();
        DCRTPoly p(params, Format::COEFFICIENT, true);
        for (uint32_t i = 0; i < k; ++i) {
            auto t     = p.GetElementAtIndex(i);
            uint64_t q = params->GetParams()[i]->GetModulus().ConvertToInt<uint64_t>();
            for (uint32_t n = 0; n < N; ++n) t[n] = NativeInteger(mod_i128(c[n], q));
            p.SetElementAtIndex(i, std::move(t));
        }
        p.SetFormat(Format::EVALUATION);
        return p;
    }
    // uniform polynomial from a seed (SHAKE-256, rejection sampling), written directly in EVALUATION form
    DCRTPoly prg_poly(const std::string& seed, const std::string& label, std::shared_ptr<ParamsT> params) const {
        uint32_t k = params->GetParams().size();
        DCRTPoly p(params, Format::EVALUATION, true);
        for (uint32_t i = 0; i < k; ++i) {
            uint64_t q    = params->GetParams()[i]->GetModulus().ConvertToInt<uint64_t>();
            int bits      = 64 - __builtin_clzll(q);
            uint64_t mask = (bits == 64) ? ~0ULL : ((1ULL << bits) - 1);
            std::string in = "VFR31-PRG|" + label + "|" + std::to_string(i) + "|" + std::to_string(q) + "|" + seed;
            size_t need = (size_t)N * 8 * 3;
            std::vector<uint8_t> buf;
            auto t   = p.GetElementAtIndex(i);
            uint32_t filled = 0;
            while (filled < N) {
                buf = shake256(in, need);  // XOF: a longer output extends the shorter one
                filled = 0;
                for (size_t off = 0; off + 8 <= buf.size() && filled < N; off += 8) {
                    uint64_t v;
                    memcpy(&v, buf.data() + off, 8);
                    v &= mask;
                    if (v < q) t[filled++] = NativeInteger(v);
                }
                need *= 2;
            }
            p.SetElementAtIndex(i, std::move(t));
        }
        return p;
    }

    // ------------------------------------------------------------------ keys
    SecretShare sample_secret() const {
        DCRTPoly::TugType tug;
        return SecretShare{DCRTPoly(tug, pQ, Format::EVALUATION)};
    }
    DCRTPoly error_poly(std::shared_ptr<ParamsT> params) const {
        const auto& dgg = cp->GetDiscreteGaussianGenerator();
        return DCRTPoly(dgg, params, Format::EVALUATION);
    }
    py::array_t<uint64_t> pk_share(const SecretShare& s, const std::string& crs) const {
        DCRTPoly a = prg_poly(crs, "pk-a", pQ);
        DCRTPoly e = error_poly(pQ);
        DCRTPoly p0 = e - a * s.s;
        return to_np(p0);
    }
    PubKey make_pk(std::vector<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>> shares,
                   const std::string& crs) const {
        DCRTPoly p0(pQ, Format::EVALUATION, true);
        for (auto& sh : shares) p0 += from_np(sh, pQ, Format::EVALUATION);
        return PubKey{p0, prg_poly(crs, "pk-a", pQ)};
    }
    // s extended from Q to QP (small coefficients)
    DCRTPoly extend_QP(const DCRTPoly& sQ) const {
        DCRTPoly ext(pQP, Format::EVALUATION, true);
        auto s0 = sQ.GetElementAtIndex(0);
        s0.SetFormat(Format::COEFFICIENT);
        const auto& pp = pQP->GetParams();
        for (uint32_t i = 0; i < nQ + nP; ++i) {
            if (i < nQ) {
                ext.SetElementAtIndex(i, sQ.GetElementAtIndex(i));
            }
            else {
                auto tmp = s0;
                tmp.SwitchModulus(pp[i]->GetModulus(), pp[i]->GetRootOfUnity(), 0, 0);
                tmp.SetFormat(Format::EVALUATION);
                ext.SetElementAtIndex(i, std::move(tmp));
            }
        }
        return ext;
    }
    // rotation-key share for every EvalSum index: b_j = -a_j psi^{-1}(s) + e + P g_j s (OpenFHE hybrid layout)
    py::array_t<uint64_t> rotkey_share(const SecretShare& s, const std::string& crs) const {
        uint32_t nidx = sum_idx.size(), kQP = nQ + nP;
        py::array_t<uint64_t> out({(size_t)nidx, (size_t)dnum, (size_t)kQP, (size_t)N});
        auto r = out.mutable_unchecked<4>();
        const auto& PModq = cp->GetPModq();
        for (uint32_t ii = 0; ii < nidx; ++ii) {
            uint32_t idx = sum_idx[ii];
            uint32_t inv = NativeInteger(idx).ModInverse(M).ConvertToInt<uint32_t>();
            std::vector<uint32_t> vec(N);
            PrecomputeAutoMap(N, inv, &vec);
            DCRTPoly sPerm = s.s.AutomorphismTransform(inv, vec);
            DCRTPoly sNew  = extend_QP(sPerm);
            for (uint32_t j = 0; j < dnum; ++j) {
                DCRTPoly a = prg_poly(crs, "rk-a|" + std::to_string(idx) + "|" + std::to_string(j), pQP);
                DCRTPoly e = error_poly(pQP);
                uint32_t start = alpha * j, end = std::min(nQ, alpha * (j + 1));
                for (uint32_t i = 0; i < kQP; ++i) {
                    auto bi = (-a.GetElementAtIndex(i) * sNew.GetElementAtIndex(i)) + e.GetElementAtIndex(i);
                    if (i >= start && i < end) bi += s.s.GetElementAtIndex(i) * PModq[i];
                    for (uint32_t n = 0; n < N; ++n) r(ii, j, i, n) = bi[n].ConvertToInt<uint64_t>();
                }
            }
        }
        return out;
    }
    KeyMap make_rotkeys(std::vector<py::array_t<uint64_t, py::array::c_style | py::array::forcecast>> shares,
                        const std::string& crs) const {
        uint32_t nidx = sum_idx.size(), kQP = nQ + nP;
        auto keys = std::make_shared<std::map<uint32_t, EvalKey<DCRTPoly>>>();
        std::vector<uint64_t> mods;
        for (auto& t : pQP->GetParams()) mods.push_back(t->GetModulus().ConvertToInt<uint64_t>());
        for (auto& sh : shares) {
            auto r = sh.unchecked<4>();
            if ((uint32_t)r.shape(0) != nidx || (uint32_t)r.shape(1) != dnum || (uint32_t)r.shape(2) != kQP ||
                (uint32_t)r.shape(3) != N)
                throw std::runtime_error("rotation-key share has the wrong shape");
        }
        for (uint32_t ii = 0; ii < nidx; ++ii) {
            uint32_t idx = sum_idx[ii];
            std::vector<DCRTPoly> av(dnum), bv(dnum);
            for (uint32_t j = 0; j < dnum; ++j) {
                av[j] = prg_poly(crs, "rk-a|" + std::to_string(idx) + "|" + std::to_string(j), pQP);
                DCRTPoly b(pQP, Format::EVALUATION, true);
                for (uint32_t i = 0; i < kQP; ++i) {
                    auto t = b.GetElementAtIndex(i);
                    for (uint32_t n = 0; n < N; ++n) {
                        u128 acc = 0;
                        for (auto& sh : shares) acc += sh.unchecked<4>()(ii, j, i, n);
                        t[n] = NativeInteger((uint64_t)(acc % mods[i]));
                    }
                    b.SetElementAtIndex(i, std::move(t));
                }
                bv[j] = std::move(b);
            }
            auto ek = std::make_shared<EvalKeyRelinImpl<DCRTPoly>>(cc);
            ek->SetAVector(std::move(av));
            ek->SetBVector(std::move(bv));
            ek->SetKeyTag(tag);
            (*keys)[idx] = ek;
        }
        return KeyMap{keys};
    }

    // ------------------------------------------------------------------ encoding
    DCRTPoly encode_poly(py::array_t<double, py::array::c_style | py::array::forcecast> v, dd scale, uint32_t k) const {
        auto r  = v.unchecked<1>();
        auto c  = enc->encode(r.data(0), (size_t)r.shape(0), scale);
        return from_i128(c, params_k(k));
    }
    Ptxt encode_plain(py::array_t<double, py::array::c_style | py::array::forcecast> v, py::tuple scale, uint32_t k) const {
        dd sc{scale[0].cast<double>(), scale[1].cast<double>()};
        return Ptxt{encode_poly(v, sc, k), sc};
    }
    Ptxt encode_plain_v(const std::vector<double>& v, double hi, double lo, uint32_t k) const {
        dd sc{hi, lo};
        auto c = enc->encode(v.data(), v.size(), sc);
        return Ptxt{from_i128(c, params_k(k)), sc};
    }
    // exact integer encoding (for tests): returns the coefficients as Python ints
    py::list encode_coeffs(py::array_t<double, py::array::c_style | py::array::forcecast> v, py::tuple scale) const {
        dd sc{scale[0].cast<double>(), scale[1].cast<double>()};
        auto r = v.unchecked<1>();
        auto c = enc->encode(r.data(0), (size_t)r.shape(0), sc);
        py::list out;
        for (auto x : c) out.append(i128_to_py(x));
        return out;
    }
    static py::object i128_to_py(i128 x) {
        bool neg = x < 0;
        u128 u   = neg ? (u128)(-x) : (u128)x;
        py::int_ hi((uint64_t)(u >> 64)), lo((uint64_t)u);
        py::object v = (hi.attr("__lshift__")(64)).attr("__or__")(lo);
        return neg ? v.attr("__neg__")() : v;
    }
    py::array_t<std::complex<double>> decode_coeffs(py::list coeffs, py::tuple scale) const {
        dd sc{scale[0].cast<double>(), scale[1].cast<double>()};
        if ((uint32_t)py::len(coeffs) != N) throw std::runtime_error("decode_coeffs: need N coefficients");
        std::vector<i128> v(N);
        for (uint32_t n = 0; n < N; ++n) v[n] = py_to_i128(coeffs[n]);
        auto z = enc->decode(v, sc);
        py::array_t<std::complex<double>> out(N / 2);
        auto w = out.mutable_unchecked<1>();
        for (uint32_t k = 0; k < N / 2; ++k) w(k) = z[k];
        return out;
    }
    static i128 py_to_i128(py::handle h) {
        py::int_ x = py::reinterpret_borrow<py::int_>(h);
        bool neg   = x.attr("__lt__")(0).cast<bool>();
        py::object a = neg ? x.attr("__neg__")() : py::object(x);
        if (a.attr("bit_length")().cast<int>() > 126) throw std::runtime_error("integer too large for int128");
        uint64_t lo = a.attr("__and__")(py::int_(~0ULL)).cast<uint64_t>();
        uint64_t hi = a.attr("__rshift__")(64).cast<uint64_t>();
        i128 v      = (i128)(((u128)hi << 64) | lo);
        return neg ? -v : v;
    }

    // ------------------------------------------------------------------ upload and conversion
    py::array_t<uint64_t> encrypt_seeded(const SecretShare& s, py::array_t<double, py::array::c_style | py::array::forcecast> v,
                                         py::tuple scale, const std::string& seed) const {
        dd sc{scale[0].cast<double>(), scale[1].cast<double>()};
        DCRTPoly a  = prg_poly(seed, "upload-a", pQ);
        DCRTPoly e  = error_poly(pQ);
        DCRTPoly m  = encode_poly(v, sc, nQ);
        DCRTPoly c0 = e + m - a * s.s;
        return to_np(c0);
    }
    Ctxt make_ctxt(DCRTPoly c0, DCRTPoly c1, dd scale) const {
        auto ct = std::make_shared<CiphertextImpl<DCRTPoly>>(cc);
        uint32_t k = c0.GetNumOfElements();
        ct->SetElements({std::move(c0), std::move(c1)});
        ct->SetNoiseScaleDeg(1);
        ct->SetLevel(nQ - k);
        ct->SetScalingFactor(scale.hi);
        ct->SetSlots(N / 2);
        ct->SetEncodingType(CKKS_PACKED_ENCODING);
        ct->SetKeyTag(tag);
        return Ctxt{ct, scale};
    }
    Ctxt convert(py::array_t<uint64_t, py::array::c_style | py::array::forcecast> c0np, const std::string& seed,
                 const SecretShare& sO, py::tuple scale) const {
        dd sc{scale[0].cast<double>(), scale[1].cast<double>()};
        DCRTPoly a  = prg_poly(seed, "upload-a", pQ);
        DCRTPoly c0 = from_np(c0np, pQ, Format::EVALUATION);
        c0 -= a * sO.s;
        return make_ctxt(std::move(c0), std::move(a), sc);
    }

    // ------------------------------------------------------------------ evaluation
    static bool same_scale(dd a, dd b) { return a.hi == b.hi && a.lo == b.lo; }
    Ctxt mul_plain(const Ctxt& x, const Ptxt& p) const {
        const auto& cv = x.ct->GetElements();
        if (cv[0].GetNumOfElements() != p.m.GetNumOfElements()) throw std::runtime_error("mul_plain: level mismatch");
        return make_ctxt(cv[0] * p.m, cv[1] * p.m, dd_mul(x.scale, p.scale));
    }
    Ctxt add(const Ctxt& x, const Ctxt& y) const {
        if (!same_scale(x.scale, y.scale)) throw std::runtime_error("add: scales differ");
        const auto& a = x.ct->GetElements();
        const auto& b = y.ct->GetElements();
        if (a[0].GetNumOfElements() != b[0].GetNumOfElements()) throw std::runtime_error("add: level mismatch");
        return make_ctxt(a[0] + b[0], a[1] + b[1], x.scale);
    }
    void add_inplace(Ctxt& x, const Ctxt& y) const {
        if (!same_scale(x.scale, y.scale)) throw std::runtime_error("add: scales differ");
        auto& a       = x.ct->GetElements();
        const auto& b = y.ct->GetElements();
        if (a[0].GetNumOfElements() != b[0].GetNumOfElements()) throw std::runtime_error("add: level mismatch");
        a[0] += b[0];
        a[1] += b[1];
    }
    // (c - r)/q_last with r = c mod q_last, centred (deterministic) or chosen with E[(c-r)/q] = c/q (stochastic)
    DCRTPoly rescale_poly(const DCRTPoly& c, bool stochastic) const {
        uint32_t k = c.GetNumOfElements();
        if (k < 2) throw std::runtime_error("rescale: no tower left to drop");
        auto last  = c.GetElementAtIndex(k - 1);
        last.SetFormat(Format::COEFFICIENT);
        uint64_t ql = last.GetModulus().ConvertToInt<uint64_t>();
        std::vector<int64_t> rr(N);  // signed remainder, |r| < ql < 2^61
        if (stochastic) {
            DiscreteUniformGeneratorImpl<NativeVector> dug;
            dug.SetModulus(NativeInteger(ql));
            NativeVector U = dug.GenerateVector(N);
            for (uint32_t n = 0; n < N; ++n) {
                uint64_t r0 = last[n].ConvertToInt<uint64_t>();
                // P[r = r0 - ql] = r0/ql, P[r = r0] = 1 - r0/ql
                rr[n] = (U[n].ConvertToInt<uint64_t>() < r0) ? (int64_t)r0 - (int64_t)ql : (int64_t)r0;
            }
        }
        else {
            for (uint32_t n = 0; n < N; ++n) {
                uint64_t r0 = last[n].ConvertToInt<uint64_t>();
                rr[n]       = (r0 > ql / 2) ? (int64_t)r0 - (int64_t)ql : (int64_t)r0;
            }
        }
        DCRTPoly out = c;
        out.DropLastElement();
        for (uint32_t i = 0; i + 1 < k; ++i) {
            auto ti    = out.GetElementAtIndex(i);
            uint64_t q = ti.GetModulus().ConvertToInt<uint64_t>();
            NativePoly rp(ti.GetParams(), Format::COEFFICIENT, true);
            for (uint32_t n = 0; n < N; ++n) rp[n] = NativeInteger(mod_i128(rr[n], q));
            rp.SetFormat(Format::EVALUATION);
            ti -= rp;
            ti *= NativeInteger(invmod(ql % q, q));
            out.SetElementAtIndex(i, std::move(ti));
        }
        return out;
    }
    Ctxt rescale(const Ctxt& x, bool stochastic) const {
        const auto& cv = x.ct->GetElements();
        uint32_t k     = cv[0].GetNumOfElements();
        double ql      = (double)qs[k - 1];
        dd qdd         = dd_from_i128((i128)qs[k - 1]);
        (void)ql;
        return make_ctxt(rescale_poly(cv[0], stochastic), rescale_poly(cv[1], stochastic), dd_div(x.scale, qdd));
    }
    Ctxt eval_sum(const Ctxt& x, const KeyMap& km) const {
        Ctxt acc = x;
        acc.ct   = x.ct->Clone();
        for (uint32_t idx : sum_idx) {
            auto rot = cc->GetScheme()->EvalAutomorphism(acc.ct, idx, *km.keys);
            auto& a  = acc.ct->GetElements();
            const auto& b = rot->GetElements();
            a[0] += b[0];
            a[1] += b[1];
        }
        return acc;
    }
    Ctxt rotate_once(const Ctxt& x, const KeyMap& km, uint32_t which) const {  // microbenchmark helper
        auto rot = cc->GetScheme()->EvalAutomorphism(x.ct, sum_idx.at(which), *km.keys);
        return Ctxt{rot, x.scale};
    }

    // ------------------------------------------------------------------ release (O) and reconstruction (R)
    DCRTPoly restrict_to(const DCRTPoly& p, uint32_t k) const {
        DCRTPoly out = p;
        uint32_t have = out.GetNumOfElements();
        if (have < k) throw std::runtime_error("restrict_to: not enough towers");
        if (have > k) out.DropLastElements(have - k);
        return out;
    }
    // eta given as residues modulo the towers of the ciphertext (COEFFICIENT form)
    py::tuple rerand_partial_decrypt(const Ctxt& x, const PubKey& pk, const SecretShare& sO,
                                     py::array_t<uint64_t, py::array::c_style | py::array::forcecast> eta_res) const {
        const auto& cv = x.ct->GetElements();
        uint32_t k     = cv[0].GetNumOfElements();
        auto params    = params_k(k);
        DCRTPoly p0 = restrict_to(pk.p0, k), a = restrict_to(pk.a, k), s = restrict_to(sO.s, k);
        DCRTPoly::TugType tug;
        DCRTPoly u(tug, params, Format::EVALUATION);
        DCRTPoly e0 = error_poly(params), e1 = error_poly(params);
        DCRTPoly c0p = cv[0] + p0 * u + e0;
        DCRTPoly c1p = cv[1] + a * u + e1;
        DCRTPoly eta = from_np(eta_res, params, Format::COEFFICIENT);
        eta.SetFormat(Format::EVALUATION);
        DCRTPoly h = c0p + c1p * s + eta;
        // c0p, u, e0, e1, eta go out of scope here and are never returned
        return py::make_tuple(to_np(c1p), to_np(h));
    }
    // v = h + c1' s_R (mod q0), CRT to centred int128; returned as (hi:int64, lo:uint64) arrays
    std::vector<i128> crt_centered(const DCRTPoly& vEval) const {
        DCRTPoly v = vEval;
        v.SetFormat(Format::COEFFICIENT);
        uint32_t k = v.GetNumOfElements();
        std::vector<uint64_t> mods(k);
        for (uint32_t i = 0; i < k; ++i) mods[i] = v.GetElementAtIndex(i).GetModulus().ConvertToInt<uint64_t>();
        // Garner: x = r0 + m0*(t1 + m1*(t2 + ...)); requires prod(mods) < 2^126
        double logq = 0;
        for (auto m : mods) logq += std::log2((double)m);
        if (logq > 125.5) throw std::runtime_error("CRT: modulus too large for int128");
        std::vector<i128> out(N);
        u128 Q = 1;
        for (auto m : mods) Q *= m;
        for (uint32_t n = 0; n < N; ++n) {
            u128 x = v.GetElementAtIndex(0)[n].ConvertToInt<uint64_t>();
            u128 prod = mods[0];
            for (uint32_t i = 1; i < k; ++i) {
                uint64_t qi = mods[i];
                uint64_t ri = v.GetElementAtIndex(i)[n].ConvertToInt<uint64_t>();
                uint64_t xm = (uint64_t)(x % qi), pm = (uint64_t)(prod % qi);
                uint64_t t  = mulmod((ri + qi - xm) % qi, invmod(pm, qi), qi);
                x += prod * (u128)t;
                prod *= qi;
            }
            i128 xs = (x > Q / 2) ? (i128)x - (i128)Q : (i128)x;
            out[n]  = xs;
        }
        return out;
    }
    static py::tuple pack_i128(const std::vector<i128>& v) {
        py::array_t<int64_t> hi(v.size());
        py::array_t<uint64_t> lo(v.size());
        auto h = hi.mutable_unchecked<1>();
        auto l = lo.mutable_unchecked<1>();
        for (size_t i = 0; i < v.size(); ++i) {
            h(i) = (int64_t)(v[i] >> 64);
            l(i) = (uint64_t)((u128)v[i]);
        }
        return py::make_tuple(hi, lo);
    }
    static std::vector<i128> unpack_i128(py::array_t<int64_t> hi, py::array_t<uint64_t> lo) {
        auto h = hi.unchecked<1>();
        auto l = lo.unchecked<1>();
        std::vector<i128> v(h.shape(0));
        for (ssize_t i = 0; i < h.shape(0); ++i) v[i] = (i128)(((u128)(uint64_t)h(i) << 64) | (u128)l(i));
        return v;
    }
    py::tuple combine(py::array_t<uint64_t, py::array::c_style | py::array::forcecast> c1p,
                      py::array_t<uint64_t, py::array::c_style | py::array::forcecast> h, const SecretShare& sR) const {
        uint32_t k  = (uint32_t)c1p.shape(0);
        auto params = params_k(k);
        DCRTPoly C1 = from_np(c1p, params, Format::EVALUATION);
        DCRTPoly H  = from_np(h, params, Format::EVALUATION);
        DCRTPoly v  = H + C1 * restrict_to(sR.s, k);
        return pack_i128(crt_centered(v));
    }
    py::array_t<std::complex<double>> decode_i128(py::array_t<int64_t> hi, py::array_t<uint64_t> lo, py::tuple scale) const {
        dd sc{scale[0].cast<double>(), scale[1].cast<double>()};
        auto v = unpack_i128(hi, lo);
        if (v.size() != N) throw std::runtime_error("decode: need N coefficients");
        auto z = enc->decode(v, sc);
        py::array_t<std::complex<double>> out(N / 2);
        auto w = out.mutable_unchecked<1>();
        for (uint32_t k2 = 0; k2 < N / 2; ++k2) w(k2) = z[k2];
        return out;
    }

    // ------------------------------------------------------------------ test-only helpers (never used by the parties)
    // same as rerand_partial_decrypt but also returns c0' so that a test can check it never reaches R
    py::tuple _test_rerand_partial_decrypt_c0p(const Ctxt& x, const PubKey& pk, const SecretShare& sO,
                                               py::array_t<uint64_t, py::array::c_style | py::array::forcecast> eta_res) const {
        const auto& cv = x.ct->GetElements();
        uint32_t k     = cv[0].GetNumOfElements();
        auto params    = params_k(k);
        DCRTPoly p0 = restrict_to(pk.p0, k), a = restrict_to(pk.a, k), s = restrict_to(sO.s, k);
        DCRTPoly::TugType tug;
        DCRTPoly u(tug, params, Format::EVALUATION);
        DCRTPoly e0 = error_poly(params), e1 = error_poly(params);
        DCRTPoly c0p = cv[0] + p0 * u + e0;
        DCRTPoly c1p = cv[1] + a * u + e1;
        DCRTPoly eta = from_np(eta_res, params, Format::COEFFICIENT);
        eta.SetFormat(Format::EVALUATION);
        DCRTPoly h = c0p + c1p * s + eta;
        return py::make_tuple(to_np(c1p), to_np(h), to_np(c0p));
    }
    // R's would-be decryption share c1' s_R (never sent); for the transcript test only
    py::array_t<uint64_t> _test_partial_share(py::array_t<uint64_t, py::array::c_style | py::array::forcecast> c1p,
                                              const SecretShare& sR) const {
        uint32_t k  = (uint32_t)c1p.shape(0);
        DCRTPoly C1 = from_np(c1p, params_k(k), Format::EVALUATION);
        return to_np(C1 * restrict_to(sR.s, k));
    }
    // decrypts with s = s_R + s_O modulo the first `keep` towers only (valid while |value| < q_0 q_1 / 2)
    py::tuple _test_decrypt_both(const Ctxt& x, const SecretShare& sR, const SecretShare& sO, uint32_t keep) const {
        const auto& cv = x.ct->GetElements();
        uint32_t k     = std::min<uint32_t>(keep, cv[0].GetNumOfElements());
        DCRTPoly s     = restrict_to(sR.s, k) + restrict_to(sO.s, k);
        DCRTPoly v     = restrict_to(cv[0], k) + restrict_to(cv[1], k) * s;
        return pack_i128(crt_centered(v));
    }
    py::tuple ctxt_np(const Ctxt& x) const {
        const auto& cv = x.ct->GetElements();
        return py::make_tuple(to_np(cv[0]), to_np(cv[1]));
    }
    py::array_t<int64_t> _test_secret_coeffs(const SecretShare& s) const {
        auto t = s.s.GetElementAtIndex(0);
        t.SetFormat(Format::COEFFICIENT);
        uint64_t q = t.GetModulus().ConvertToInt<uint64_t>();
        py::array_t<int64_t> out(N);
        auto w = out.mutable_unchecked<1>();
        for (uint32_t n = 0; n < N; ++n) {
            uint64_t x = t[n].ConvertToInt<uint64_t>();
            w(n)       = (x > q / 2) ? (int64_t)x - (int64_t)q : (int64_t)x;
        }
        return out;
    }
    uint32_t ctxt_towers(const Ctxt& x) const { return x.ct->GetElements()[0].GetNumOfElements(); }
    py::tuple ctxt_scale(const Ctxt& x) const { return py::make_tuple(x.scale.hi, x.scale.lo); }
    py::tuple ptxt_scale(const Ptxt& x) const { return py::make_tuple(x.scale.hi, x.scale.lo); }
    py::list automorphism_indices() const {
        py::list l;
        for (auto i : sum_idx) l.append(i);
        return l;
    }
};

// =====================================================================================
// BFV core for labeled PSI
// =====================================================================================
struct BFVKeys {
    KeyPair<DCRTPoly> kp;
};
struct BFVCt {
    Ciphertext<DCRTPoly> ct;
};
struct BFVPowers {
    std::map<uint32_t, Ciphertext<DCRTPoly>> pw;
};

template <typename T>
static std::string ser(const T& obj) {
    std::stringstream ss;
    Serial::Serialize(obj, ss, SerType::BINARY);
    return ss.str();
}
template <typename T>
static void deser(T& obj, const std::string& s) {
    std::stringstream ss(s);
    Serial::Deserialize(obj, ss, SerType::BINARY);
}

class BFVCore {
public:
    CryptoContext<DCRTPoly> cc;
    uint64_t t;
    uint32_t N;
    PublicKey<DCRTPoly> pk;  // R's BFV public key (O side)

    BFVCore(uint32_t logN, uint64_t t_, uint32_t depth, bool sec128) : t(t_) {
        CCParams<CryptoContextBFVRNS> p;
        p.SetPlaintextModulus(t_);
        p.SetMultiplicativeDepth(depth);
        p.SetRingDim(1u << logN);
        p.SetSecurityLevel(sec128 ? HEStd_128_classic : HEStd_NotSet);
        p.SetMultiplicationTechnique(HPSPOVERQLEVELED);
        cc = GenCryptoContext(p);
        cc->Enable(PKE);
        cc->Enable(KEYSWITCH);
        cc->Enable(LEVELEDSHE);
        N = cc->GetRingDimension();
    }
    explicit BFVCore(const std::string& ccser) {
        deser(cc, ccser);
        cc->Enable(PKE);
        cc->Enable(KEYSWITCH);
        cc->Enable(LEVELEDSHE);
        t = cc->GetCryptoParameters()->GetPlaintextModulus();
        N = cc->GetRingDimension();
    }
    py::bytes context_bytes() const { return py::bytes(ser(cc)); }
    py::list moduli() const {
        py::list l;
        for (auto& tw : cc->GetCryptoParameters()->GetElementParams()->GetParams())
            l.append(tw->GetModulus().ConvertToInt<uint64_t>());
        return l;
    }
    std::shared_ptr<BFVKeys> keygen() {
        auto k = std::make_shared<BFVKeys>();
        k->kp  = cc->KeyGen();
        cc->EvalMultKeyGen(k->kp.secretKey);
        return k;
    }
    py::tuple public_bytes(const BFVKeys& k) const {
        auto rk = cc->GetEvalMultKeyVector(k.kp.secretKey->GetKeyTag());
        return py::make_tuple(py::bytes(ser(k.kp.publicKey)), py::bytes(ser(rk[0])));
    }
    void load_public(const std::string& pkb, const std::string& rkb) {
        deser(pk, pkb);
        EvalKey<DCRTPoly> rk;
        deser(rk, rkb);
        // in a single-process test both parties share OpenFHE's static key map and the key is already there
        auto have = CryptoContextImpl<DCRTPoly>::GetAllEvalMultKeys();
        if (have.find(rk->GetKeyTag()) == have.end()) cc->InsertEvalMultKey({rk});
    }
    Plaintext pt(py::array_t<uint64_t, py::array::c_style | py::array::forcecast> v) const {
        auto r = v.unchecked<1>();
        std::vector<int64_t> x(N, 0);
        if ((uint32_t)r.shape(0) > N) throw std::runtime_error("BFV plaintext too long");
        for (ssize_t i = 0; i < r.shape(0); ++i) {
            if (r(i) >= t) throw std::runtime_error("BFV plaintext value >= t");
            x[i] = (int64_t)r(i);
        }
        return cc->MakePackedPlaintext(x);
    }
    py::bytes encrypt_sym(const BFVKeys& k, py::array_t<uint64_t, py::array::c_style | py::array::forcecast> v) const {
        return py::bytes(ser(cc->Encrypt(k.kp.secretKey, pt(v))));
    }
    py::array_t<uint64_t> decrypt(const BFVKeys& k, const std::string& ctb) const {
        Ciphertext<DCRTPoly> ct;
        deser(ct, ctb);
        Plaintext out;
        cc->Decrypt(k.kp.secretKey, ct, &out);
        out->SetLength(N);
        const auto& vals = out->GetPackedValue();
        py::array_t<uint64_t> res(N);
        auto w = res.mutable_unchecked<1>();
        for (uint32_t i = 0; i < N; ++i) {
            int64_t x = vals[i];
            if (x < 0) x += (int64_t)t;
            w(i) = (uint64_t)x;
        }
        return res;
    }
    // ---- O: powers w^d for d = 1..D from windows w^{2^i}, each new power costs one product,
    // depth ceil(log2(popcount(d)))
    std::shared_ptr<BFVPowers> compute_powers(std::vector<std::string> windows, uint32_t D) const {
        auto P = std::make_shared<BFVPowers>();
        for (size_t i = 0; i < windows.size(); ++i) {
            Ciphertext<DCRTPoly> c;
            deser(c, windows[i]);
            P->pw[1u << i] = c;
        }
        std::function<Ciphertext<DCRTPoly>(uint32_t)> get = [&](uint32_t d) -> Ciphertext<DCRTPoly> {
            auto it = P->pw.find(d);
            if (it != P->pw.end()) return it->second;
            // split the set bits of d into two halves of (almost) equal popcount
            int pc = __builtin_popcount(d), take = pc / 2, cnt = 0;
            uint32_t a = 0;
            for (int b = 0; b < 32 && cnt < take; ++b)
                if (d & (1u << b)) {
                    a |= 1u << b;
                    ++cnt;
                }
            uint32_t bpart = d ^ a;
            auto r = cc->EvalMult(get(a), get(bpart));
            P->pw[d] = r;
            return r;
        };
        for (uint32_t d = 1; d <= D; ++d) {
            if ((d & (d - 1)) == 0 && P->pw.find(d) == P->pw.end())
                throw std::runtime_error("missing window power " + std::to_string(d));
            get(d);
        }
        return P;
    }
    // Evaluate the two masked polynomials of one partition at the encrypted query:
    //   S(w) = sum_d Sc[d] w^d with Sc = r1 * Pc        (= r1 * P(w): 0 iff the bin's item is a root of P)
    //   Q(w) = sum_d Qc[d] w^d with Qc = r2 * Pc + Lc   (= r2 * P(w) + L(w): the encrypted label at a match)
    // The masks r1, r2 (fresh, uniform in Z_t^*) are folded into the plaintext coefficients by the caller,
    // which saves one plaintext-multiplication level.
    py::tuple eval_partition(const BFVPowers& P, py::array_t<uint64_t, py::array::c_style | py::array::forcecast> Sc,
                             py::array_t<uint64_t, py::array::c_style | py::array::forcecast> Qc) const {
        auto sr = Sc.unchecked<2>();
        auto qr = Qc.unchecked<2>();
        uint32_t D = (uint32_t)sr.shape(0) - 1;
        if ((uint32_t)qr.shape(0) != D + 1) throw std::runtime_error("eval_partition: shape mismatch");
        auto row = [&](auto& rr, uint32_t d) {
            py::array_t<uint64_t> v(N);
            auto w = v.mutable_unchecked<1>();
            for (uint32_t i = 0; i < N; ++i) w(i) = rr(d, i);
            return v;
        };
        Ciphertext<DCRTPoly> accS, accQ;
        for (uint32_t d = 1; d <= D; ++d) {
            auto ts = cc->EvalMult(P.pw.at(d), pt(row(sr, d)));
            auto tq = cc->EvalMult(P.pw.at(d), pt(row(qr, d)));
            accS    = accS ? cc->EvalAdd(accS, ts) : ts;
            accQ    = accQ ? cc->EvalAdd(accQ, tq) : tq;
        }
        { auto s0 = pt(row(sr, 0)); accS = cc->EvalAdd(accS, s0); }
        { auto q0 = pt(row(qr, 0)); accQ = cc->EvalAdd(accQ, q0); }
        auto s = std::make_shared<BFVCt>();
        auto q = std::make_shared<BFVCt>();
        s->ct  = accS;
        q->ct  = accQ;
        return py::make_tuple(s, q);
    }
    // SanitizeBFV: modulus switch to `towers` limbs, add a compressed fresh Enc_pk(0), add flooding noise
    // e (given as residues modulo the remaining limbs) to the c0 component.
    py::bytes sanitize(const BFVCt& x, uint32_t towers,
                       py::array_t<uint64_t, py::array::c_style | py::array::forcecast> flood_res) const {
        if (!pk) throw std::runtime_error("sanitize: public key not loaded");
        auto c  = cc->Compress(x.ct, towers);
        std::vector<int64_t> zeros(N, 0);
        auto z  = cc->Compress(cc->Encrypt(pk, cc->MakePackedPlaintext(zeros)), towers);
        auto r  = cc->EvalAdd(c, z);
        auto& cv = r->GetElements();
        auto fr  = flood_res.unchecked<2>();
        if ((uint32_t)fr.shape(0) != cv[0].GetNumOfElements() || (uint32_t)fr.shape(1) != N)
            throw std::runtime_error("sanitize: flooding noise has the wrong shape");
        DCRTPoly e(cv[0].GetParams(), Format::COEFFICIENT, true);
        for (uint32_t i = 0; i < e.GetNumOfElements(); ++i) {
            auto ti = e.GetElementAtIndex(i);
            for (uint32_t n = 0; n < N; ++n) ti[n] = NativeInteger(fr(i, n));
            e.SetElementAtIndex(i, std::move(ti));
        }
        e.SetFormat(cv[0].GetFormat());
        cv[0] += e;
        return py::bytes(ser(r));
    }
    py::bytes ct_bytes(const BFVCt& x) const { return py::bytes(ser(x.ct)); }
    py::bytes compress_bytes(const BFVCt& x, uint32_t towers) const { return py::bytes(ser(cc->Compress(x.ct, towers))); }
    py::list ct_moduli(const BFVCt& x) const {
        py::list l;
        for (auto& tw : x.ct->GetElements()[0].GetParams()->GetParams())
            l.append(tw->GetModulus().ConvertToInt<uint64_t>());
        return l;
    }
    // test helper: log2 of the centred infinity norm of e = (c0 + c1 s) - round(Q m / t) mod Q, where
    // m = round(t (c0 + c1 s) / Q) mod t is the decrypted coefficient vector (Q = modulus of the ciphertext)
    double _test_noise_log2(const BFVKeys& k, const std::string& ctb) const {
        Ciphertext<DCRTPoly> ct;
        deser(ct, ctb);
        const auto& cv = ct->GetElements();
        uint32_t kk    = cv[0].GetNumOfElements();
        DCRTPoly s     = k.kp.secretKey->GetPrivateElement();
        if (s.GetNumOfElements() > kk) s.DropLastElements(s.GetNumOfElements() - kk);
        DCRTPoly v = cv[0] + cv[1] * s;
        v.SetFormat(Format::COEFFICIENT);
        auto big     = v.CRTInterpolate();
        BigInteger Q = big.GetModulus();
        BigInteger T(t), half = Q >> 1;
        double mx = 0;
        for (uint32_t n = 0; n < N; ++n) {
            BigInteger x = big[n];
            // m = round(t x / Q) mod t
            BigInteger m = ((x * T) + half) / Q;
            m = m.Mod(T);
            BigInteger qm = ((Q * m) + (T >> 1)) / T;
            BigInteger diff = x.ModSub(qm.Mod(Q), Q);
            double d = (diff > half) ? (Q - diff).ConvertToDouble() : diff.ConvertToDouble();
            if (d > mx) mx = d;
        }
        return mx > 0 ? std::log2(mx) : 0.0;
    }
};

// =====================================================================================
// labeled-PSI polynomial preparation over Z_t (t < 2^32): membership polynomials with roots
// {items} padded by the root t-1 to degree D, and label polynomials through (item, label)
// =====================================================================================
static py::tuple psi_build_polys(py::array_t<uint64_t, py::array::c_style | py::array::forcecast> xs,
                                 py::array_t<uint64_t, py::array::c_style | py::array::forcecast> ys,
                                 py::array_t<int32_t, py::array::c_style | py::array::forcecast> counts, uint64_t t) {
    // xs, ys: (alpha, nbins, D); counts: (alpha, nbins)
    auto X = xs.unchecked<3>();
    auto Y = ys.unchecked<3>();
    auto C = counts.unchecked<2>();
    ssize_t alpha = X.shape(0), nb = X.shape(1), D = X.shape(2);
    if (t >= (1ULL << 32)) throw std::runtime_error("psi_build_polys: t must be < 2^32");
    py::array_t<uint64_t> P({(size_t)alpha, (size_t)(D + 1), (size_t)nb});
    py::array_t<uint64_t> L({(size_t)alpha, (size_t)D, (size_t)nb});
    auto Pw = P.mutable_unchecked<3>();
    auto Lw = L.mutable_unchecked<3>();
    const uint64_t dummy = t - 1;
#pragma omp parallel for collapse(2) schedule(static)
    for (ssize_t a = 0; a < alpha; ++a) {
        for (ssize_t b = 0; b < nb; ++b) {
            int n = C(a, b);
            std::vector<uint64_t> lab(D, 0);
            // membership: product over real roots and dummy roots
            std::vector<uint64_t> cur(1, 1);
            for (int i = 0; i < (int)D; ++i) {
                uint64_t r = (i < n) ? X(a, b, i) % t : dummy;
                std::vector<uint64_t> nxt(cur.size() + 1, 0);
                for (size_t j = 0; j < cur.size(); ++j) {
                    nxt[j + 1] = (nxt[j + 1] + cur[j]) % t;
                    nxt[j]     = (nxt[j] + (t - r) % t * cur[j]) % t;
                }
                cur.swap(nxt);
            }
            for (ssize_t d = 0; d <= D; ++d) Pw(a, d, b) = cur[d];
            // label: Lagrange through the n real points
            if (n > 0) {
                std::vector<uint64_t> full(1, 1);
                for (int i = 0; i < n; ++i) {
                    uint64_t r = X(a, b, i) % t;
                    std::vector<uint64_t> nxt(full.size() + 1, 0);
                    for (size_t j = 0; j < full.size(); ++j) {
                        nxt[j + 1] = (nxt[j + 1] + full[j]) % t;
                        nxt[j]     = (nxt[j] + (t - r) % t * full[j]) % t;
                    }
                    full.swap(nxt);
                }
                for (int i = 0; i < n; ++i) {
                    uint64_t xi = X(a, b, i) % t;
                    // q = full / (x - xi) by synthetic division
                    std::vector<uint64_t> q(n, 0);
                    q[n - 1] = full[n];
                    for (int j = n - 1; j >= 1; --j) q[j - 1] = (full[j] + xi * q[j]) % t;
                    uint64_t den = 0;
                    for (int j = n - 1; j >= 0; --j) den = (den * xi + q[j]) % t;
                    if (den == 0) throw std::runtime_error("psi_build_polys: duplicate item in a partition");
                    uint64_t sc = (Y(a, b, i) % t) * powmod(den, t - 2, t) % t;
                    for (int j = 0; j < n; ++j) lab[j] = (lab[j] + sc * q[j]) % t;
                }
            }
            for (ssize_t d = 0; d < D; ++d) Lw(a, d, b) = lab[d];
        }
    }
    return py::make_tuple(P, L);
}

static void set_threads(int n) { omp_set_num_threads(n); }

// =====================================================================================
PYBIND11_MODULE(vfr_native, m) {
    m.doc() = "Polynomial-level CKKS/BFV operations for protocol v3.1 (OpenFHE 1.5.1 backend)";
    m.def("set_threads", &set_threads);
    m.def("openfhe_version", []() { return GetOPENFHEVersion(); });
    py::class_<SecretShare, std::shared_ptr<SecretShare>>(m, "SecretShare");
    py::class_<PubKey, std::shared_ptr<PubKey>>(m, "PubKey");
    py::class_<KeyMap, std::shared_ptr<KeyMap>>(m, "KeyMap");
    py::class_<Ctxt, std::shared_ptr<Ctxt>>(m, "Ctxt");
    py::class_<Ptxt, std::shared_ptr<Ptxt>>(m, "Ptxt");
    py::class_<CKKSCore, std::shared_ptr<CKKSCore>>(m, "CKKSCore")
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, bool>(), py::arg("logN"), py::arg("depth"),
             py::arg("scaling_mod_size"), py::arg("first_mod_size"), py::arg("dnum"), py::arg("sec128") = true)
        .def_readonly("N", &CKKSCore::N)
        .def_readonly("nQ", &CKKSCore::nQ)
        .def_readonly("nP", &CKKSCore::nP)
        .def_readonly("dnum", &CKKSCore::dnum)
        .def_readonly("alpha", &CKKSCore::alpha)
        .def_readonly("qs", &CKKSCore::qs)
        .def_readonly("ps", &CKKSCore::ps)
        .def("automorphism_indices", &CKKSCore::automorphism_indices)
        .def("sample_secret", &CKKSCore::sample_secret)
        .def("pk_share", &CKKSCore::pk_share)
        .def("make_pk", &CKKSCore::make_pk)
        .def("rotkey_share", &CKKSCore::rotkey_share)
        .def("make_rotkeys", &CKKSCore::make_rotkeys)
        .def("encode_plain", &CKKSCore::encode_plain)
        .def("encode_plain_v", &CKKSCore::encode_plain_v, py::call_guard<py::gil_scoped_release>())
        .def("encode_coeffs", &CKKSCore::encode_coeffs)
        .def("decode_coeffs", &CKKSCore::decode_coeffs)
        .def("encrypt_seeded", &CKKSCore::encrypt_seeded)
        .def("convert", &CKKSCore::convert)
        .def("mul_plain", &CKKSCore::mul_plain, py::call_guard<py::gil_scoped_release>())
        .def("add", &CKKSCore::add, py::call_guard<py::gil_scoped_release>())
        .def("add_inplace", &CKKSCore::add_inplace, py::call_guard<py::gil_scoped_release>())
        .def("rescale", &CKKSCore::rescale, py::call_guard<py::gil_scoped_release>())
        .def("eval_sum", &CKKSCore::eval_sum, py::call_guard<py::gil_scoped_release>())
        .def("rotate_once", &CKKSCore::rotate_once, py::call_guard<py::gil_scoped_release>())
        .def("rerand_partial_decrypt", &CKKSCore::rerand_partial_decrypt)
        .def("combine", &CKKSCore::combine)
        .def("decode_i128", &CKKSCore::decode_i128)
        .def("ctxt_towers", &CKKSCore::ctxt_towers)
        .def("ctxt_scale", &CKKSCore::ctxt_scale)
        .def("ptxt_scale", &CKKSCore::ptxt_scale)
        .def("ctxt_np", &CKKSCore::ctxt_np)
        .def("_test_decrypt_both", &CKKSCore::_test_decrypt_both)
        .def("_test_secret_coeffs", &CKKSCore::_test_secret_coeffs)
        .def("_test_rerand_partial_decrypt_c0p", &CKKSCore::_test_rerand_partial_decrypt_c0p)
        .def("_test_partial_share", &CKKSCore::_test_partial_share);
    py::class_<BFVKeys, std::shared_ptr<BFVKeys>>(m, "BFVKeys");
    py::class_<BFVCt, std::shared_ptr<BFVCt>>(m, "BFVCt");
    py::class_<BFVPowers, std::shared_ptr<BFVPowers>>(m, "BFVPowers");
    py::class_<BFVCore, std::shared_ptr<BFVCore>>(m, "BFVCore")
        .def(py::init<uint32_t, uint64_t, uint32_t, bool>(), py::arg("logN"), py::arg("t"), py::arg("depth"),
             py::arg("sec128") = true)
        .def(py::init<const std::string&>())
        .def_readonly("t", &BFVCore::t)
        .def_readonly("N", &BFVCore::N)
        .def("context_bytes", &BFVCore::context_bytes)
        .def("moduli", &BFVCore::moduli)
        .def("keygen", &BFVCore::keygen)
        .def("public_bytes", &BFVCore::public_bytes)
        .def("load_public", &BFVCore::load_public)
        .def("encrypt_sym", &BFVCore::encrypt_sym)
        .def("decrypt", &BFVCore::decrypt)
        .def("compute_powers", &BFVCore::compute_powers)
        .def("eval_partition", &BFVCore::eval_partition)
        .def("sanitize", &BFVCore::sanitize)
        .def("ct_moduli", &BFVCore::ct_moduli)
        .def("ct_bytes", &BFVCore::ct_bytes)
        .def("compress_bytes", &BFVCore::compress_bytes)
        .def("_test_noise_log2", &BFVCore::_test_noise_log2);
    m.def("psi_build_polys", &psi_build_polys);
}
