#pragma once

#include "euclidian_rabitq_point.h"

namespace parlayANN {

struct QueryFactorsData {
        float c1 = 0;
        float c2 = 0;
        float c34 = 0;

        float qr_to_c_L2sqr = 0;
        float qr_to_c_L2 = 0;
};

#define faiss

struct L2RabitQ{

    // ||or - qr||^2 = ||or - c||^2 + ||qr - c||^2 - 2 * ||or - c|| * ||qr - c|| * <q, o>
    // 只要实时计算出<q, o>，其他可以提前计算。
    // <q, o¯>=⟨o¯, o⟩ * <q, o> + ⟨o¯, e1⟩ * √︃ (1 - <q, o>^2)，其中⟨o¯, e1⟩ ≈ 0
    // 所以，<q, o> = <q, o¯> / ⟨o¯, o⟩



    // 预存一些中间计算结果------------------------------------------------------
    struct FactorsData {
        // ||or - c||^2 - ((metric==IP) ? ||or||^2 : 0)
        float or_minus_c_l2sqr = 0;    
        // <o,q> = <o¯,q> / ⟨o¯, o⟩，其中o¯是压缩编码，||o¯|| = 1，o¯:={+1/√︃d,-1/√︃d}
        // 由于在后续计算<o¯,q>时将o¯在每个轴上的投影当作1，所以得到的结果需要乘以1/√︃d进行修正
        // 1 / (⟨o¯, o⟩ * √︃d)
        float dp_multiplier = 0;
    };



    L2RabitQ(uint d) 
        : d(d),
        code_size(get_code_size(d)),
        inv_d_sqrt((d == 0) ? 1.0f : (1.0f / std::sqrt((float)d)))
    {
    }

    static size_t get_code_size(const uint d) {
        return (d + 7) / 8 + sizeof(FactorsData);
    }


    // 压缩底库向量【注意，传入的底库向量必须减去质心】
    void encoder_o(const float* rotated_x, uint8_t* code){
        // ||or - c||^2
        float norm_L2sqr = 0;
        // ||or||^2, which is equal to ||P(or)||^2 and ||P^(-1)(or)||^2
        float or_L2sqr = 0;
        // dot product
        float dp_oO = 0;
        
        FactorsData* fac = reinterpret_cast<FactorsData*>(code + (d + 7) / 8);
        memset(code, 0, code_size);

        for (size_t j = 0; j < d; j++) {
            const float or_minus_c = rotated_x[j];
            norm_L2sqr += or_minus_c * or_minus_c;
            or_L2sqr += rotated_x[j] * rotated_x[j];

            const bool xb = (or_minus_c > 0);

            dp_oO += xb ? or_minus_c : (-or_minus_c);

            // store the output data
            if (xb) {
                // enable a particular bit
                code[j / 8] |= (1 << (j % 8));
            }
            
        }

        // 此时，dp_oO / √︃d = ⟨o¯, o⟩ , dp_oO = ⟨o¯, o⟩ * √︃d

        // compute factors

        // compute the inverse norm
        const float inv_norm_L2 =
                (std::abs(norm_L2sqr) < std::numeric_limits<float>::epsilon())
                ? 1.0f
                : (1.0f / std::sqrt(norm_L2sqr));
        dp_oO *= inv_norm_L2; // 此时，dp_oO = ⟨o¯, o⟩ * √︃d / ||or - c||
        dp_oO *= inv_d_sqrt; // 此时，dp_oO = ⟨o¯, o⟩ / ||or - c||

        const float inv_dp_oO =
                (std::abs(dp_oO) < std::numeric_limits<float>::epsilon())
                ? 1.0f
                : (1.0f / dp_oO); // inv_dp_oO = ||or - c|| / ⟨o¯, o⟩

        fac->or_minus_c_l2sqr = norm_L2sqr;

        fac->dp_multiplier = inv_dp_oO;
        #ifdef faiss
            fac->dp_multiplier = inv_dp_oO * std::sqrt(norm_L2sqr);
        #endif
    }

    float fvec_norm_L2sqr(const float* x, uint d) {
        // the double in the _ref is suspected to be a typo. Some of the manual
        // implementations this replaces used float.
        float res = 0;
        for (uint i = 0; i != d; ++i) {
            res += x[i] * x[i];
        }

        return res;
    }    

    QueryFactorsData set_query_fac(const float* rotated_q) {

        QueryFactorsData query_fac;
        query_fac.qr_to_c_L2sqr = fvec_norm_L2sqr(rotated_q, d);
        query_fac.qr_to_c_L2 = std::sqrt(query_fac.qr_to_c_L2sqr);

        // do not quantize the query
        float sum_q = 0;
        for (size_t i = 0; i < d; i++) {
            sum_q += rotated_q[i];
        }

        query_fac.c1 = 2 * inv_d_sqrt;
        query_fac.c2 = 0;
        query_fac.c34 = sum_q * inv_d_sqrt;
        return query_fac;
    }

    // 计算查询向量与被压缩的底库向量的距离【注意，传入的查询向量必须减去质心】
    float euclidian_distance(const uint8_t* code, const float* rotated_q, QueryFactorsData& query_fac){
        // split the code into parts
        const uint8_t* binary_data = code;
        const FactorsData* fac =
                reinterpret_cast<const FactorsData*>(code + (d + 7) / 8);

        // this is the baseline code
        //
        // compute <q,o> using floats
        float dot_qo = 0;
        // It was a willful decision (after the discussion) to not to pre-cache
        //   the sum of all bits, just in order to reduce the overhead per vector.
        uint64_t sum_q = 0;
        for (size_t i = 0; i < d; i++) {
            // extract i-th bit
            const uint8_t masker = (1 << (i % 8));
            const bool b_bit = ((binary_data[i / 8] & masker) == masker);

            // accumulate dp
            dot_qo += (b_bit) ? rotated_q[i] : 0;
            // accumulate sum-of-bits
            sum_q += (b_bit) ? 1 : 0;
        }

        // 此时，dot_qo = <o¯, q> * √︃d，where o¯[i] > 0，即只累加o¯中大于0的维度

        float final_dot = 0;
        // dot-product itself
        final_dot += query_fac.c1 * dot_qo; // 此时，final_dot = 2 * <o¯, q>，where o¯[i] > 0，即只累加o¯中大于0的维度
    
        // query_fac.c34 = <o¯, q>，where o¯[i] > 0，即只累加o¯中小于等于0的维度
        // normalizer coefficients
        final_dot -= query_fac.c34; // 此时，final_dot = <o¯, q>

        // this is ||or - c||^2 - (IP ? ||or||^2 : 0)
        const float or_c_l2sqr = fac->or_minus_c_l2sqr;

        // pre_dist = ||or - c||^2 + ||qr - c||^2 -
        //     2 * ||or - c|| * ||qr - c|| * <q,o> - (IP ? ||or||^2 : 0)
        const float pre_dist = or_c_l2sqr + query_fac.qr_to_c_L2sqr -
                2 *query_fac.qr_to_c_L2 * fac->dp_multiplier * final_dot;
        #ifdef faiss
            return or_c_l2sqr + query_fac.qr_to_c_L2sqr -
                2 * fac->dp_multiplier * final_dot;
        #endif


        // 其中，final_dot = <o¯, q>，fac->dp_multiplier = ||or - c|| / ⟨o¯, o⟩
        return pre_dist;
    }

    uint d;
    size_t code_size;
    float inv_d_sqrt;

};


float euclidian_rabitq_distance_(const uint8_t *r_encoder, const float *rotated_q, QueryFactorsData& query_fac, L2RabitQ& rabitQ) {
    return rabitQ.euclidian_distance(r_encoder, rotated_q, query_fac);
}

struct RabitQ_Euclidian_Point{
  struct parameters {
    L2RabitQ* rabitQ;
    int num_bytes() const {return rabitQ == nullptr ? 0 : (rabitQ->d + 7) / 8 + sizeof(L2RabitQ::FactorsData);}
    parameters() : rabitQ(nullptr) {}
    parameters(L2RabitQ* rabitQ) : rabitQ(rabitQ) {}
  };

  long id() const {return id_;}

  RabitQ_Euclidian_Point() : values(nullptr), id_(-1), params() {}

  RabitQ_Euclidian_Point(uint8_t* values, long id, parameters params)
    : values((uint8_t*) values), id_(id), params(params) {}

  void prefetch() const {
    int l = (params.num_bytes() - 1)/64 + 1;
    for (int i=0; i < l; i++)
      __builtin_prefetch((char*) values + i* 64);
  } 

  float distance(const float* rotated_q, QueryFactorsData& query_fac){
    return params.rabitQ->euclidian_distance(values, rotated_q, query_fac);
  }

  uint8_t* data(){return values;}

  parameters params;

private:
  uint8_t* values;
  long id_;
};

}