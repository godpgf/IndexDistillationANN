// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <sys/mman.h>
#include <algorithm>
#include <iostream>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/internal/file_map.h"
#include "types.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace parlayANN
{

  template <class Point_, typename indexType_ = unsigned int>
  struct PointRange
  {
    // using T = T_;
    using Point = Point_;
    using parameters = typename Point::parameters;
    using byte = uint8_t;
    using indexType = indexType_;

    indexType dimension() const { return params.dims; }

    PointRange() : values(std::shared_ptr<byte[]>(nullptr, std::free)), n(0), max_cache(0), index_cache(nullptr), sid(0), eid(0) {}

    template <typename PR>
    PointRange(const PR &pr, const parameters &p) : params(p)
    {
      n = pr.size();
      sid = pr.sid;
      eid = pr.eid;
      max_cache = pr.getMaxCache();
      index_cache = pr.getIndexCache();
      size_t num_bytes = p.num_bytes();
      aligned_bytes = (num_bytes <= 32) ? 32 : 64 * ((num_bytes - 1) / 64 + 1);
      size_t total_bytes = n * aligned_bytes;
      byte *ptr = (byte *)aligned_alloc(1l << 21, total_bytes);
      madvise(ptr, total_bytes, MADV_HUGEPAGE);
      values = std::shared_ptr<byte[]>(ptr, std::free);
      byte *vptr = values.get();
      parlay::parallel_for(0, n, [&](size_t i)
                           { Point::translate_point(vptr + i * aligned_bytes, pr[i], params); });
    }

    template <typename PR>
    PointRange(PR &pr) : PointRange(pr, Point::generate_parameters(pr)) {}

    template <typename PR>
    PointRange(PR &pr, int dims) : PointRange(pr, Point::generate_parameters(dims)) {}

    PointRange(indexType num_points, indexType d, const parameters p) : n(num_points), max_cache(n), index_cache(nullptr), sid(0), eid(num_points), params(p)
    {
      size_t num_bytes = params.num_bytes();
      aligned_bytes = 64 * ((num_bytes - 1) / 64 + 1);
      if (aligned_bytes != num_bytes)
        std::cout << "Aligning bytes to " << aligned_bytes << std::endl;
      size_t total_bytes = n * aligned_bytes;
      byte *ptr = (byte *)aligned_alloc(1l << 21, total_bytes);
      madvise(ptr, total_bytes, MADV_HUGEPAGE);
      values = std::shared_ptr<byte[]>(ptr, std::free);
    }

    PointRange(indexType num_points, indexType d) : PointRange(num_points, d, parameters(d))
    {
    }

    PointRange(const char *filename, size_t max_cache_size = 0, byte **index_cache = nullptr) : max_cache(max_cache_size), index_cache(index_cache), values(std::shared_ptr<byte[]>(nullptr, std::free))
    {
      sid = 0;
      eid = 0;
      size_t new_eid = eid;
      if (filename == NULL)
      {
        n = 0;
        return;
      }
      reader.open(filename, std::ios::in | std::ios::binary);
      if (!reader.is_open())
      {
        std::cout << "Data file " << filename << " not found" << std::endl;
        std::abort();
      }

      // read num points and max degree
      indexType num_points;
      indexType d;
      reader.read((char *)(&num_points), sizeof(indexType));
      n = num_points;

      if (max_cache == 0 || max_cache > n)
      {
        new_eid = n;
      }
      else
      {
        new_eid = max_cache;
      }
      max_cache = new_eid - sid;
      reader.read((char *)(&d), sizeof(indexType));
      params = parameters(d);
      std::cout << "Data: detected " << num_points << " points with dimension " << d << std::endl;

      size_t num_bytes = params.num_bytes();
      aligned_bytes = 64 * ((num_bytes - 1) / 64 + 1);
      if (aligned_bytes != num_bytes)
        std::cout << "Aligning bytes to " << aligned_bytes << std::endl;
      size_t total_bytes = max_cache * aligned_bytes;
      byte *ptr = (byte *)aligned_alloc(1l << 21, total_bytes);
      madvise(ptr, total_bytes, MADV_HUGEPAGE);
      values = std::shared_ptr<byte[]>(ptr, std::free);

      if (max_cache == n)
      {
        load2cache(sid, new_eid);
      }
    }

    void setIndexCache(byte **index_cache)
    {
      this->index_cache = index_cache;
    }

    byte **getIndexCache() const
    {
      return this->index_cache;
    }

    indexType getMaxCache() const
    {
      return max_cache;
    }

    void save(const char *filename)
    {
      if (filename == NULL)
      {
        std::cout << "ERROR: save filename is nullptr!" << std::endl;
        abort();
      }
      std::ofstream writer(filename);
      indexType num_points = n;
      indexType d = params.dims;
      writer.write((char *)(&num_points), sizeof(indexType));
      writer.write((char *)(&d), sizeof(indexType));
      size_t num_bytes = params.num_bytes();

      size_t BLOCK_SIZE = 1000000;
      size_t index = 0;
      while (index < n)
      {
        size_t floor = index;
        size_t ceiling = index + BLOCK_SIZE <= n ? index + BLOCK_SIZE : n;
        size_t m = ceiling - floor;
        byte *data_start = new byte[m * num_bytes];
        parlay::parallel_for(floor, ceiling, [&](size_t i)
                             { std::memmove(data_start + (i - floor) * num_bytes,
                                            values.get() + i * aligned_bytes,
                                            num_bytes); });
        writer.write((char *)(data_start), m * num_bytes);

        delete[] data_start;
        index = ceiling;
      }
      writer.close();
    }

    size_t get_max_cache() { return max_cache; }

    template<typename idType>
    void load2cache(idType *all_ids, size_t ids_num)
    {
      if (max_cache == n){
        this->sid = 0;
        this->eid = n;
        return;
      }
      if(ids_num > max_cache){
        std::cout<<"load2cache ERROR:max cache is "<<max_cache<<" but load num is "<<ids_num<<std::endl;
        abort();
      }
      if (index_cache == nullptr)
      {
        std::cout << "load2cache ERROR: index_cache is empty!" << std::endl;
        abort();
      }
      size_t num_bytes = params.num_bytes();
      for (size_t i = 0; i < ids_num; ++i)
      {
        reader.seekg(sizeof(indexType) * 2 + all_ids[i] * num_bytes);
        byte *to_data = values.get() + i * aligned_bytes;
        reader.read((char *)(to_data), num_bytes);
        index_cache[all_ids[i]] = to_data;
      }
    }

    void load2cache(indexType sid, indexType eid)
    {        
      if (this->sid <= sid && this->eid >= eid)
        return;
      if (eid - sid > max_cache)
      {
        std::cout << "load2cache ERROR! eis=" << eid << " sid=" << sid << " max_cache=" << max_cache << std::endl;
        abort();
      }
      this->sid = sid;
      this->eid = eid;
      size_t num_bytes = params.num_bytes();
      size_t BLOCK_SIZE = 1000000;
      size_t index = 0;
      reader.seekg(sizeof(indexType) * 2 + sid * num_bytes);
      auto max_cache = eid - sid;
      while (index < max_cache)
      {
        size_t floor = index;
        size_t ceiling = index + BLOCK_SIZE <= max_cache ? index + BLOCK_SIZE : max_cache;
        size_t m = ceiling - floor;
        byte *data_start = new byte[m * num_bytes];
        reader.read((char *)(data_start), m * num_bytes);
        parlay::parallel_for(floor, ceiling, [&](size_t i)
                             { std::memmove(values.get() + i * aligned_bytes,
                                            data_start + (i - floor) * num_bytes,
                                            num_bytes); });
        delete[] data_start;
        index = ceiling;
      }

      if (index_cache != nullptr)
      {
        parlay::parallel_for(sid, eid, [&](size_t i)
                             { index_cache[i] = values.get() + (i - sid) * aligned_bytes; });
      }
    }

    ~PointRange()
    {
      if (reader.is_open())
      {
        reader.close();
      }
    }

    void clearIndexCache(){
      if(index_cache == nullptr || max_cache == n)
        return;
      sid = 0;
      eid = 0;
      // parlay::parallel_for(0, n, [&](size_t i){
      //   index_cache[i] = nullptr;
      // });
    }

    void fill2IndexCache(indexType *all_ids, indexType ids_num){
      if(max_cache == n)
        return;
      if (index_cache == nullptr)
      {
        std::cout << "fill2IndexCache ERROR: index_cache is empty!" << std::endl;
        abort();
      }
      size_t num_bytes = params.num_bytes();
      for (size_t i = 0; i < ids_num; ++i)
      {
        if(index_cache[all_ids[i]] != nullptr)
          continue;
        reader.seekg(sizeof(indexType) * 2 + all_ids[i] * num_bytes);
        byte *to_data = values.get() + eid * aligned_bytes;
        reader.read((char *)(to_data), num_bytes);
        index_cache[all_ids[i]] = to_data;
        ++eid;
        if(eid > max_cache){
          std::cout<<"fill2IndexCache ERROR: index cache is full!"<<std::endl;
          abort();
        }
      }      
    }

    void fill2IndexCache(indexType index, byte *from_data=nullptr){
      if(max_cache == n)
        return;
      if (index_cache == nullptr)
      {
        std::cout << "fill2IndexCache ERROR: index_cache is empty!" << std::endl;
        abort();
      }
      if(index_cache[index] != nullptr)
        return;
      byte *to_data = values.get() + eid * aligned_bytes;
      size_t num_bytes = params.num_bytes();
      if(from_data == nullptr){
        reader.seekg(sizeof(indexType) * 2 + index * num_bytes);
        reader.read((char *)(to_data), num_bytes);
      } else {
        memcpy(to_data, from_data, num_bytes);
      }

      index_cache[index] = to_data;
      ++eid;
      if(eid > max_cache){
        std::cout<<"fill2IndexCache ERROR: index cache is full!"<<std::endl;
        abort();
      }    
    }

    size_t size() const { return n; }

    indexType get_dims() const { return params.dims; }

    Point operator[](size_t i) const
    {
      if (index_cache == nullptr || max_cache == n)
      {
        if (i >= eid || i < sid)
        {
          std::cout << "ERROR: point index out of range: " << i << " from range [" << sid << ", " << eid << ")" << std::endl;
          abort();
        }
        return Point(values.get() + (i - sid) * aligned_bytes, i, params);
      }
      else
      {
        return Point(index_cache[i], i, params);
      }
    }

    parameters params;

  protected:
    std::ifstream reader;
    std::shared_ptr<byte[]> values;
    size_t aligned_bytes;
    indexType n;
    indexType max_cache;
    // 引用values，以支持读取非连续的向量，如Points[100],Points[67984]
    byte **index_cache;

  public:
    // 用来缓存所有向量中的一段
    indexType sid;
    indexType eid;
  };

} // end namespace
