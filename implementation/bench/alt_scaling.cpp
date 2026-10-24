// alt_scaling.cpp -- OpenFHE-only microbenchmark of the CKKS scale options for circuit A (Section "scale").
// Measures, for a given parameter set, the moduli chain and the cost of the operations the aggregation uses:
// ct-pt product at the top level, rescale, one rotation and EvalSum (log2(N/2) rotations) after one rescale.
//   ./alt_scaling FIXEDMANUAL 59 60 3 2 0      (the implementation's chain, 64-bit build)
//   ./alt_scaling COMPOSITE   62 62 3 2 32     (composite scaling: each scaling modulus = 2 primes of ~31 bits)
//   ./alt_scaling FIXEDMANUAL 64 100 2 2 0     (128-bit native build: 64-bit scaling primes, 100-bit q0)
#include "openfhe.h"
#include <chrono>
#include <iostream>
using namespace lbcrypto;

template <class F>
double timeit(F f, int reps) {
    std::vector<double> ts;
    for (int i = 0; i < reps; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        f();
        ts.push_back(std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count());
    }
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
}

int main(int argc, char** argv) {
    std::string tech = argv[1];
    uint32_t sms = atoi(argv[2]), fms = atoi(argv[3]), depth = atoi(argv[4]), dnum = atoi(argv[5]), rws = atoi(argv[6]);
    CCParams<CryptoContextCKKSRNS> p;
    p.SetMultiplicativeDepth(depth);
    p.SetScalingModSize(sms);
    p.SetFirstModSize(fms);
    p.SetRingDim(1 << 14);
    p.SetBatchSize(1 << 13);
    p.SetKeySwitchTechnique(HYBRID);
    p.SetNumLargeDigits(dnum);
    p.SetSecurityLevel(HEStd_128_classic);
    if (tech == "COMPOSITE") {
        p.SetScalingTechnique(COMPOSITESCALINGAUTO);
        p.SetRegisterWordSize(rws);
    }
    else {
        p.SetScalingTechnique(FIXEDMANUAL);
    }
    CryptoContext<DCRTPoly> cc;
    try {
        cc = GenCryptoContext(p);
    }
    catch (std::exception& e) {
        std::cout << "{\"error\": \"" << e.what() << "\"}" << std::endl;
        return 1;
    }
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    auto cp = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    auto kp = cc->KeyGen();
    cc->EvalSumKeyGen(kp.secretKey);
    std::vector<double> x(1 << 13);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (i % 7) * 0.1;
    auto pt  = cc->MakeCKKSPackedPlaintext(x);
    auto ct  = cc->Encrypt(kp.publicKey, pt);
    int reps = 5;
    double t_enc = timeit([&] { cc->MakeCKKSPackedPlaintext(x); }, reps);
    Ciphertext<DCRTPoly> prod;
    double t_mul = timeit([&] { prod = cc->EvalMult(ct, pt); }, reps);
    Ciphertext<DCRTPoly> r;
    double t_rs  = timeit([&] { r = cc->Rescale(prod); }, reps);
    double t_rot = timeit([&] { cc->EvalRotate(r, 1); }, reps);
    double t_sum = timeit([&] { cc->EvalSum(r, 1 << 13); }, 3);
    std::cout << "{\"tech\": \"" << tech << "\", \"sms\": " << sms << ", \"fms\": " << fms << ", \"depth\": " << depth
              << ", \"native_int\": " << NATIVEINT << ", \"towers_Q\": " << cp->GetElementParams()->GetParams().size()
              << ", \"towers_P\": " << cp->GetParamsP()->GetParams().size()
              << ", \"logQP\": " << cp->GetParamsQP()->GetModulus().GetMSB() << ", \"q_bits\": [";
    bool first = true;
    for (auto& t : cp->GetElementParams()->GetParams()) {
        std::cout << (first ? "" : ", ") << t->GetModulus().GetMSB();
        first = false;
    }
    std::cout << "], \"scale_log2\": " << std::log2(cc->GetEncodingParams()->GetPlaintextModulus() > 0 ? pt->GetScalingFactor() : 0)
              << ", \"t_encode\": " << t_enc << ", \"t_mul_plain\": " << t_mul << ", \"t_rescale\": " << t_rs
              << ", \"t_rotate\": " << t_rot << ", \"t_evalsum\": " << t_sum << "}" << std::endl;
    return 0;
}
