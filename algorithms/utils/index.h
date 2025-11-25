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

#include <math.h>

#include <algorithm>
#include <random>
#include <set>

#include "../utils/point_range.h"
#include "../utils/graph.h"
#include "../utils/types.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/delayed.h"
#include "parlay/random.h"
#include "../utils/beamSearch.h"
#include "../utils/prune.h"
#include "../utils/dfs.h"
#include "../utils/pivot.h"
#include "../utils/split.h"

namespace parlayANN {

template<typename PR_, typename QPR_, typename indexType, typename ivfIndexType>
struct knn_index {
  using Point = typename PR_::Point;
  using QPoint = typename QPR_::Point;
  using distanceType = typename Point::distanceType;
  using pid = std::pair<indexType, distanceType>;
  using PR = PR_;
  using QPR = QPR_;
  using GraphI = Graph<indexType>;
  using IVFGraphI = Graph<ivfIndexType>;

  Prune<QPR, indexType> prune;
  Split<PR, QPR, indexType, ivfIndexType> split;
  
  indexType start_point;

  knn_index(uint PR, uint R) : prune(PR, R) {}

  indexType get_start() { return start_point; }



  // add ngh to candidates without adding any repeats
  template<typename rangeType1, typename rangeType2>
  void add_neighbors_without_repeats(const rangeType1 &ngh, rangeType2& candidates) {
    std::unordered_set<indexType> a;
    for (auto c : candidates) a.insert(c);
    for (int i=0; i < ngh.size(); i++)
      if (a.count(ngh[i]) == 0) candidates.push_back(ngh[i]);
  }

  void set_start(){start_point = 0;}

  /*size_t build_index(GraphI &G, PR &Points, QPR &QPoints,
                   stats<indexType> &BuildStats, bool sort_neighbors = true){
    std::cout << "Building graph..." << std::endl;
    set_start();
    parlay::sequence<indexType> inserts = parlay::tabulate(Points.size(), [&] (size_t i){
      return static_cast<indexType>(i);});
    // auto dijkstra = Dijkstra<PointRange, indexType>(G.size());
    if (BP.single_batch != 0) {
      int degree = BP.single_batch;
      std::cout << "Using single batch per round with " << degree << " random start edges" << std::endl;
      parlay::random_generator gen;
      std::uniform_int_distribution<long> dis(0, G.size());
      parlay::parallel_for(0, G.size(), [&] (long i) {
        std::vector<indexType> outEdges(degree);
        for (int j = 0; j < degree; j++) {
          auto r = gen[i*degree + j];
          outEdges[j] = dis(r);
        }
        G[i].update_neighbors(outEdges);
      });
    }

    // last pass uses alpha
    std::cout << "number of passes = " << BP.num_passes << std::endl;
    for (int i=0; i < BP.num_passes; i++) {
      if (i == BP.num_passes - 1)
        batch_insert(inserts, G, Points, QPoints, BuildStats, BP.alpha, BP.cos_angle, true, 2, .02);
      else
        batch_insert(inserts, G, Points, QPoints, BuildStats, BP.init_alpha, BP.init_cos_angle, true, 2, .02);
    }

    size_t isolate_node_num = DFS::repair_isolate_node(G, start_point);

    if (sort_neighbors) {
      parlay::parallel_for (0, G.size(), [&] (long i) {
        auto less = [&] (indexType j, indexType k) {
          return Points[i].distance(Points[j]) < Points[i].distance(Points[k]);};
        G[i].sort(less);});
    }
    return isolate_node_num;
  }*/

  void batch_insert(parlay::sequence<indexType> &inserts,
                    GraphI &G, QPR &QPoints,
                    stats<indexType> &BuildStats, uint L, double alpha, double cos_angle,
                    bool random_order = true, double base = 2,
                    double max_fraction = .02, bool print=true) {
    for(int p : inserts){
      if(p < 0 || p > (int) G.size()){
        std::cout << "ERROR: invalid point "
                  << p << " given to batch_insert" << std::endl;
        abort();
      }
    }
    size_t n = G.size();
    size_t m = inserts.size();
    size_t inc = 0;
    size_t count = 0;
    float frac = 0.0;
    float progress_inc = .1;
    size_t max_batch_size = std::min(static_cast<size_t>(max_fraction * static_cast<float>(n)),
                                     1000000ul);
    //fix bug where max batch size could be set to zero
    if(max_batch_size == 0) max_batch_size = n;
    parlay::sequence<int> rperm;
    if (random_order) 
      rperm = parlay::random_permutation<int>(static_cast<int>(m));
    else
      rperm = parlay::tabulate(m, [&](int i) { return i; });
    auto shuffled_inserts =
      parlay::tabulate(m, [&](size_t i) { return inserts[rperm[i]]; });
    parlay::internal::timer t_beam("beam search time");
    parlay::internal::timer t_bidirect("bidirect time");
    parlay::internal::timer t_prune("prune time");
    t_beam.stop();
    t_bidirect.stop();
    t_prune.stop();
    while (count < m) {
      size_t floor;
      size_t ceiling;
      if (pow(base, inc) <= max_batch_size) {
        floor = static_cast<size_t>(pow(base, inc)) - 1;
        ceiling = std::min(static_cast<size_t>(pow(base, inc + 1)) - 1, m);
        count = std::min(static_cast<size_t>(pow(base, inc + 1)) - 1, m);
      } else {
        floor = count;
        ceiling = std::min(count + static_cast<size_t>(max_batch_size), m);
        count += static_cast<size_t>(max_batch_size);
      }


      parlay::sequence<parlay::sequence<indexType>> new_out_(ceiling-floor);
      // search for each node starting from the start_point, then call
      // robustPrune with the visited list as its candidate set
      t_beam.start();

      // 1. 创建随机数引擎（推荐使用 mt19937）
      std::random_device rd;  // 用于生成随机种子
      std::mt19937 gen(rd()); // Mersenne Twister 引擎

      // 2. 定义分布范围 [a, b]
      std::uniform_int_distribution<size_t> dist(0, G.size()-1);

      parlay::parallel_for(floor, ceiling, [&](size_t i) {
        size_t index = shuffled_inserts[i];

        int sp = start_point;

        QueryParams QP((long) 0, L, (double) 0.0, (long) QPoints.size(), (long) G.max_degree());
        auto [visited, bs_distance_comps] =
          //beam_search<Point, PointRange, indexType>(Points[index], G, Points, sp, QP);
          beam_search_rerank__<QPoint, QPoint, QPR, QPR, indexType>(QPoints[index],
                                                                 QPoints[index],
                                                                 G,
                                                                 QPoints,
                                                                 QPoints,
                                                                 sp,
                                                                 QP);
        BuildStats.increment_dist(index, bs_distance_comps);
        BuildStats.increment_visited(index, visited.size());

        long rp_distance_comps;
        std::tie(new_out_[i-floor], rp_distance_comps) = prune.robustPrune(index, visited, G, QPoints, alpha, cos_angle);
        BuildStats.increment_dist(index, rp_distance_comps);
      });

      parlay::parallel_for(floor, ceiling, [&](size_t i) {
        G[shuffled_inserts[i]].update_neighbors(new_out_[i-floor]);
      });

      t_beam.stop();

      // make each edge bidirectional by first adding each new edge
      //(i,j) to a sequence, then semisorting the sequence by key values
      t_bidirect.start();

      auto flattened = parlay::delayed::flatten(
        parlay::tabulate(ceiling - floor, [&](size_t i) {
          indexType index = shuffled_inserts[i + floor];
          return parlay::delayed::map(new_out_[i], [=] (indexType ngh) {return std::pair(ngh, index);});
        }));
      auto grouped_by = parlay::group_by_key(parlay::delayed::to_sequence(flattened));

      t_bidirect.stop();
      t_prune.start();
      // finally, add the bidirectional edges; if they do not make
      // the vertex exceed the degree bound, just add them to out_nbhs;
      // otherwise, use robustPrune on the vertex with user-specified alpha
      parlay::parallel_for(0, grouped_by.size(), [&](size_t j) {
        auto &[index, candidates] = grouped_by[j];
	size_t newsize = candidates.size() + G[index].size();
        if (newsize <= prune.R) {
	  add_neighbors_without_repeats(G[index], candidates);
	  G[index].update_neighbors(candidates);
        } else {
          auto [new_out_2_, distance_comps] = prune.robustPrune(index, std::move(candidates), G, QPoints, alpha, cos_angle);
	  G[index].update_neighbors(new_out_2_);
          BuildStats.increment_dist(index, distance_comps);
        }
      });
      t_prune.stop();

      if (print) {
        auto ind = frac * n;
        if (floor <= ind && ceiling > ind) {
          frac += progress_inc;
          std::cout << "Pass " << 100 * frac << "% complete"
                    << std::endl;
        }
      }
      inc += 1;
    }
    t_beam.total();
    t_bidirect.total();
    t_prune.total();
  }

  /*uint batch_insert_and_split(const PR& Points, QPR &QPoints, Pivot& pivot, GraphI& G, IVFGraphI& ivf, uint cNum, 
                    stats<indexType> &BuildStats, uint L, double alpha, double cos_angle,
                    bool random_order = true, double base = 2,
                    double max_fraction = .02, bool print=true){
    size_t m = Points.size();
    size_t inc = 0;
    size_t count = 0;
    float frac = 0.0;
    float progress_inc = .1;
    size_t max_batch_size = std::min(static_cast<size_t>(max_fraction * static_cast<float>(m)),
                                     1000000ul);

    // parlay::worker_id()
    using PQPR = PointRange<Euclidian_Point_PQ>;
    PQPR cachePoint = PQPR((unsigned int)parlay::num_workers(), (unsigned int)(pivot.pivots_num * pivot.times * pivot.chunk * pivot.sub_chunk));
    indexType sp = start_point;

    while (count < m) {
      size_t floor = count;
      size_t ceiling = std::min(count + static_cast<size_t>(max_batch_size), m);
      count += static_cast<size_t>(max_batch_size);

      // 搜索最近的质心
      parlay::sequence<std::pair<indexType, ivfIndexType>> new_out_(ceiling-floor);

      parlay::parallel_for(floor, ceiling, [&](size_t index) {
        // 预先计算当前插入的点到各个轴的距离
        Euclidian_Point_PQ q = cachePoint[parlay::worker_id()];
        pivot.preCalculateDis(Points[index], q.data());
        QueryParams QP((long) 0, L, (double) 0.0, (long) QPoints.size(), (long) G.max_degree());
        auto [visited, bs_distance_comps] =
          //beam_search<Point, PointRange, indexType>(Points[index], G, Points, sp, QP);
          beam_search_rerank__<Euclidian_Point_PQ, Euclidian_Point_PQ, QPR, QPR, indexType>(q,
                                                                 q,
                                                                 G,
                                                                 QPoints,
                                                                 QPoints,
                                                                 sp,
                                                                 QP);   
               
          new_out_[index - floor] = std::pair<indexType, ivfIndexType>(visited[0].first, index);
      });

      std::cout<<floor<<","<<ceiling<<std::endl;
      // 分裂出新节点
      uint newCNum = split.split(new_out_, Points, QPoints, cNum, pivot, G, ivf);

      std::cout<<"new "<<newCNum<<" nodes!"<<std::endl;

      if(newCNum > cNum){
        // 将新节点插入到图中
        parlay::sequence<indexType> inserts = parlay::tabulate(cNum - newCNum, [&](size_t i){return static_cast<indexType>(cNum + i);});
        batch_insert(inserts, G, QPoints, BuildStats, L, alpha, cos_angle, random_order, base, max_fraction, print);
      }


      cNum = newCNum;
    }
    return cNum;
  }*/

};

} // end namespace
