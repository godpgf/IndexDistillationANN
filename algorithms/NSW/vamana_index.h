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

namespace parlayANN {

template<typename PointRange, typename QPointRange, typename indexType>
struct knn_index {
  using Point = typename PointRange::Point;
  using QPoint = typename QPointRange::Point;
  using distanceType = typename Point::distanceType;
  using pid = std::pair<indexType, distanceType>;
  using PR = PointRange;
  using QPR = QPointRange;
  using GraphI = Graph<indexType>;

  BuildParams BP;
  Prune<PointRange, indexType> prune;
  
  indexType start_point;

  knn_index(BuildParams &BP) : BP(BP), prune(BP.PR, BP.R) {}

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

  size_t build_index(GraphI &G, PR &Points, QPR &QPoints,
                    stats<indexType> &BuildStats, bool sort_neighbors = true){
    parlay::sequence<indexType> inserts = parlay::tabulate(Points.size(), [&] (size_t i){
      return static_cast<indexType>(i);});
    build_index(inserts, G, Points, QPoints, BuildStats, sort_neighbors);

    size_t isolate_node_num = DFS::repair_isolate_node(G, start_point);

    if (sort_neighbors) {
      parlay::parallel_for (0, G.size(), [&] (long i) {
        auto less = [&] (indexType j, indexType k) {
          return Points[i].distance(Points[j]) < Points[i].distance(Points[k]);};
        G[i].sort(less);});
    }
    return isolate_node_num;    
  }

  void build_index(parlay::sequence<indexType>& inserts, GraphI &G, PR &Points, QPR &QPoints,
                   stats<indexType> &BuildStats, bool sort_neighbors = true){
    
    std::cout << "Building graph..." << std::endl;
    set_start();

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
      if (i == BP.num_passes - 1){
        uint32_t limit_bin_size = 0;
        batch_insert(inserts, G, Points, QPoints, BuildStats, BP.alpha, BP.cos_angle, true, limit_bin_size, 2, .02);
      }
      else{
        uint32_t limit_bin_size = 2;
        batch_insert(inserts, G, Points, QPoints, BuildStats, BP.init_alpha, BP.init_cos_angle, true, limit_bin_size, 2, .02);
      }
    }

  }


  void batch_insert(parlay::sequence<indexType> &inserts,
                    GraphI &G, PR &Points, QPR &QPoints,
                    stats<indexType> &BuildStats, double alpha, double cos_angle,
                    bool random_order = false, uint32_t limit_bin_size = 0, double base = 2,
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

      if (BP.single_batch != 0) {
        floor = 0;
        ceiling = m;
        count = m;
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
        if(BP.sp_type == 1){
          sp = dist(gen);
        } else if(BP.sp_type == 2) {
          sp = index;
        }

        QueryParams QP((long) 0, BP.L, (double) 0.0, (long) Points.size(), (long) G.max_degree());

        if (limit_bin_size > 0)
        {
            QP.limit_bin_size = limit_bin_size;
            QP.cur_bin = index % QP.limit_bin_size;
        }

        auto [visited, bs_distance_comps] =
          //beam_search<Point, PointRange, indexType>(Points[index], G, Points, sp, QP);
          beam_search_rerank__<Point, QPoint, PR, QPR, indexType>(Points[index],
                                                                 QPoints[index],
                                                                 G,
                                                                 Points,
                                                                 QPoints,
                                                                 sp,
                                                                 QP);
        BuildStats.increment_dist(index, bs_distance_comps);
        BuildStats.increment_visited(index, visited.size());

        long rp_distance_comps;
        std::tie(new_out_[i-floor], rp_distance_comps) = prune.robustPrune(index, visited, G, Points, alpha, cos_angle);
        BuildStats.increment_dist(index, rp_distance_comps);
      });

      parlay::parallel_for(floor, ceiling, [&](size_t i) {
        G[shuffled_inserts[i]].update_neighbors(new_out_[i-floor]);
      });

      t_beam.stop();

      // make each edge bidirectional by first adding each new edge
      //(i,j) to a sequence, then semisorting the sequence by key values
      t_bidirect.start();

      auto flattened = parlay::delayed::flatten(parlay::tabulate(ceiling - floor, [&](size_t i) {
        indexType index = shuffled_inserts[i + floor];
        return parlay::delayed::map(new_out_[i], [=] (indexType ngh) {
          return std::pair(ngh, index);});}));
      auto grouped_by = parlay::group_by_key(parlay::delayed::to_sequence(flattened));

      t_bidirect.stop();
      t_prune.start();
      // finally, add the bidirectional edges; if they do not make
      // the vertex exceed the degree bound, just add them to out_nbhs;
      // otherwise, use robustPrune on the vertex with user-specified alpha
      parlay::parallel_for(0, grouped_by.size(), [&](size_t j) {
        auto &[index, candidates] = grouped_by[j];
	size_t newsize = candidates.size() + G[index].size();
        if (newsize <= BP.R) {
	  add_neighbors_without_repeats(G[index], candidates);
	  G[index].update_neighbors(candidates);
        } else {
          auto [new_out_2_, distance_comps] = prune.robustPrune(index, std::move(candidates), G, Points, alpha, cos_angle);
	  G[index].update_neighbors(new_out_2_);
          BuildStats.increment_dist(index, distance_comps);
        }
      });
      t_prune.stop();

      if (print && BP.single_batch == 0) {
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

};

} // end namespace
