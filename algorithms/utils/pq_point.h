
#pragma once

#include <algorithm>
#include <iostream>
#include <bitset>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/internal/file_map.h"

#include "types.h"
// #include "common/time_loop.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "../utils/euclidian_point.h"

namespace parlayANN {

struct Euclidian_Point_PQ{
  using distanceType = float;
  using byte = uint8_t;
  struct parameters {
    int dims;
    int num_bytes() const {return dims * sizeof(float);}
    parameters() : dims(0) {}
    parameters(int dims) : dims(dims) {}

  };

  void prefetch() const {
    int l = (params.dims * sizeof(float) - 1)/64 + 1;
    for (int i=0; i < l; i++)
      __builtin_prefetch((char*) values + i* 64);
  }

  long id() const {return id_;}

  Euclidian_Point_PQ() : values(nullptr), id_(-1), params(0) {}

  Euclidian_Point_PQ(byte* values, long id, parameters params)
    : values((float*)values), id_(id), params(params) {}

  float* data() const {return values;}




  parameters params;

private:
  float* values;
  long id_;
};


template<typename T_>
struct PQ_Point {
  using distanceType = float;
  using T = T_;
  using byte = uint8_t;

  struct parameters {
    int dims;
    int num_bytes() const {return dims * sizeof(T);}
    parameters() : dims(0), sub_chunk(1), combine_dis(false), sub_bit(0), dis_cache(nullptr), pivots_num_per_chunk(0) {}
    parameters(int dims) : dims(dims), sub_chunk(1), combine_dis(false), sub_bit(0), dis_cache(nullptr), pivots_num_per_chunk(0) {}
    parameters(const parameters& p) : dims(p.dims), sub_chunk(p.sub_chunk), combine_dis(p.combine_dis), sub_bit(p.sub_bit), dis_cache(p.dis_cache), pivots_num_per_chunk(p.pivots_num_per_chunk) {}
    
    void set_dis_cache(uint pivots_num_per_chunk, float* dis_cache){
      this->pivots_num_per_chunk = pivots_num_per_chunk;
      this->dis_cache = dis_cache;
    }

    void set_meta(uint sub_chunk, uint sub_bit, bool combine_dis=false){
      this->sub_chunk = sub_chunk;
      this->sub_bit = sub_bit;
      this->combine_dis = combine_dis;
    }

    uint sub_chunk, sub_bit;
    bool combine_dis;

    float* dis_cache;
    uint pivots_num_per_chunk;

  };

  static distanceType d_min() {return 0;}
  static bool is_metric() {return true;}
  T operator[](long i) const {return *(values + i);}
  T* data() const {return values;}

  bool same_as(const Euclidian_Point_PQ& q) const {
    return false;
  }

  float distance(const float* q2p_dis) const {
    if(params.sub_bit == 0){
      std::cout<<"PQ_Point ERROR: Need to set_meta first!!!"<<std::endl;
      abort();
    }
    float dis = 0;
    uint mask = (params.sub_chunk == 1) ? -1 : (1 << params.sub_bit) - 1;
    for(uint i = 0; i < params.dims; ++i){
        auto compress_pid = values[i];
        for(uint j = 0; j < params.sub_chunk; ++j){
          auto pid = (compress_pid & mask);
          dis += q2p_dis[pid * params.dims * params.sub_chunk + i * params.sub_chunk + j];
          compress_pid = (compress_pid >> params.sub_bit);
        }
    }
    return dis;
  }

  float combine_distance(const float* q2cp_dis) const {
    if(params.sub_bit == 0){
      std::cout<<"PQ_Point ERROR: Need to set_meta first!!!"<<std::endl;
      abort();
    }
    float dis = 0;
    for(uint i = 0; i < params.dims; ++i){
        auto pid = values[i];
        dis += q2cp_dis[pid * params.dims + i];
    }
    return dis;
  }

  float distance(const Euclidian_Point_PQ& point) const {
    if(params.combine_dis){
      return this->combine_distance(point.data());
    }
    return this->distance(point.data());
  }

  float distance(const PQ_Point<T_>& point) const {
    assert(params.dis_cache != nullptr);
    if(params.sub_bit == 0){
      std::cout<<"PQ_Point ERROR: Need to set_meta first!!!"<<std::endl;
      abort();
    }
    float dis = 0;
    const auto* pv = point.data();
    auto p = params.pivots_num_per_chunk;
    auto pp = params.pivots_num_per_chunk * params.pivots_num_per_chunk;
    uint mask = (1 << params.sub_bit) - 1;
    for(uint cid = 0; cid < params.dims; cid++){
      auto compress_pi = values[cid];
      auto compress_pj = pv[cid];
      for(uint sid = 0; sid < params.sub_chunk; ++sid){
        auto pi = (compress_pi & mask);
        auto pj = (compress_pj & mask);
        dis += params.dis_cache[(cid * params.sub_chunk + sid) * pp + pi * p + pj];
        compress_pi = (compress_pi >> params.sub_bit);
        compress_pj = (compress_pj >> params.sub_bit);
      }
      
    }
    return dis;
  }

  void prefetch() const {
    int l = (params.dims * sizeof(T) - 1)/64 + 1;
    for (int i=0; i < l; i++)
      __builtin_prefetch((char*) values + i* 64);
  }

  long id() const {return id_;}

  PQ_Point() : values(nullptr), id_(-1), params(0) {}

  PQ_Point(byte* values, long id, const parameters& params)
    : values((T*) values), id_(id), params(params) {}

  bool same_as(const PQ_Point<T_>& q) const {
    return id_ == q.id();
  }

  parameters params;

private:
  T* values;
  long id_;
};




} // end namespace
