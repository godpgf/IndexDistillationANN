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

namespace parlayANN
{

  template <typename PointRange, typename QPointRange, typename indexType>
  struct knn_index
  {
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

    knn_index(BuildParams &BP) : BP(BP), prune(BP.PR,BP.R) {}

    indexType get_start() { return start_point; }

    // add ngh to candidates without adding any repeats
    template <typename rangeType1, typename rangeType2>
    void add_neighbors_without_repeats(const rangeType1 &ngh, rangeType2 &candidates)
    {
      std::unordered_set<indexType> a;
      for (auto c : candidates)
        a.insert(c);
      for (int i = 0; i < ngh.size(); i++)
        if (a.count(ngh[i]) == 0)
          candidates.push_back(ngh[i]);
    }

    void set_start() { start_point = 0; }

    size_t build_index(GraphI &G, HIdMapping &hid_mapping, PR &Points, QPR &QPoints,
                       stats<indexType> &BuildStats, bool sort_neighbors = true)
    {
      std::cout << "Building graph..." << std::endl;
      set_start();

      batch_insert(G, hid_mapping, Points, QPoints, BuildStats, BP.alpha, BP.m_l, true, 2, .02);
      start_point = hid_mapping.get_sp();
      size_t isolate_node_num = DFS::repair_isolate_node(G, start_point, 1000, &hid_mapping);

      if (sort_neighbors)
      {
        parlay::parallel_for(0, G.size(), [&](long i)
                             {
        auto less = [&] (indexType j, indexType k) {
          return Points[hid_mapping.get_root_id(i)].distance(Points[hid_mapping.get_root_id(j)]) < Points[hid_mapping.get_root_id(i)].distance(Points[hid_mapping.get_root_id(k)]);};
        G[i].sort(less); });
      }
      return isolate_node_num;
    }

    void batch_insert(GraphI &G, HIdMapping &hid_mapping, PR &Points, QPR &QPoints,
                      stats<indexType> &BuildStats, double alpha, double m_l,
                      bool random_order = false, double base = 2,
                      double max_fraction = .02, bool print = true)
    {
      size_t n = G.size();
      size_t m = Points.size();
      size_t inc = 0;
      size_t count = 0;
      float frac = 0.0;
      float progress_inc = .1;
      size_t max_batch_size = std::min(static_cast<size_t>(max_fraction * static_cast<float>(n)),
                                       1000000ul);
      // fix bug where max batch size could be set to zero
      if (max_batch_size == 0)
        max_batch_size = n;

      // HNSW params
      auto all_enter_points = std::vector<size_t>(max_batch_size);
      // auto all_enter_point_size = std::vector<size_t>(max_batch_size, 1);
      uint L = 0;
      uint64_t ep = 0;
      auto inserts = std::vector<indexType>(max_batch_size);
      auto levels = std::vector<uint>(max_batch_size);

      parlay::internal::timer t_beam("beam search time");
      parlay::internal::timer t_bidirect("bidirect time");
      parlay::internal::timer t_prune("prune time");
      t_beam.stop();
      t_bidirect.stop();
      t_prune.stop();
      while (count < m)
      {
        size_t floor;
        size_t ceiling;
        if (pow(base, inc) <= max_batch_size)
        {
          floor = static_cast<size_t>(pow(base, inc)) - 1;
          ceiling = std::min(static_cast<size_t>(pow(base, inc + 1)) - 1, m);
          count = std::min(static_cast<size_t>(pow(base, inc + 1)) - 1, m);
        }
        else
        {
          floor = count;
          ceiling = std::min(count + static_cast<size_t>(max_batch_size), m);
          count += static_cast<size_t>(max_batch_size);
        }

        // set seed
        srand(ceiling);

        // 初始化每个待插入节点的level
        size_t cur_batch_size = ceiling - floor;
       
        /*parlay::parallel_for(0, cur_batch_size, [&](size_t i)
                             {
          inserts[i] = floor + i;
          all_enter_points[i] = ep;
          double r = static_cast<double>(rand()) / RAND_MAX;
          levels[i] = std::min((uint)(-log(r) * m_l), L + 1); });*/
        for(size_t i = 0; i < cur_batch_size; ++i){
          all_enter_points[i] = ep;
          double r = static_cast<double>(rand()) / RAND_MAX;
          levels[i] = std::min((uint)(-log(r) * m_l), L + 1);         
        }

        // test
        if (L > 0)
        {
          assert(hid_mapping.get_level(ep) == L);
        }

        // 申请图上的节点，并将所插入的向量id转化为节点id
        for (size_t i = 0; i < cur_batch_size; ++i)
        {
          size_t &cur_node_num = hid_mapping.cur_node_num;
          int l = levels[i];
          int64_t node_id = cur_node_num;
          inserts[i] = node_id;
          int64_t pre_id;
          while (l >= 0)
          {
            cur_node_num++;
            assert(cur_node_num <= hid_mapping.max_node_num);
            if (l == 0)
            {
              pre_id = ((int64_t)-1) - (floor + i);
              assert(-pre_id - 1 < m);
            }
            else
            {
              pre_id = cur_node_num;
            }
            hid_mapping.set_id_in_pre_level(node_id, pre_id);
            node_id = pre_id;
            l--;
          }
        }

        // 顶层之上如果又加了一层，在这一层随意插入当前的节点，并保证连通性，但是暂时先不对这一层进行搜索
        std::vector<size_t> new_level_nodes;
        for (size_t i = 0; i < cur_batch_size; ++i)
        {
          if (levels[i] > L)
          {
            new_level_nodes.push_back(inserts[i]);
            inserts[i] = hid_mapping.get_id_in_pre_level(inserts[i]);
            --levels[i];
          }
        }
        for (size_t i = 0; i < new_level_nodes.size(); ++i)
        {
          for (size_t j = 1; j < std::min(new_level_nodes.size(), (size_t)G.max_degree()); ++j)
          {
            size_t nb_id = new_level_nodes[(i + j) % new_level_nodes.size()];
            size_t node_id = new_level_nodes[i];
            G[node_id].append_neighbor(nb_id);
            // assert(hid_mapping.get_level(node_id)==hid_mapping.get_level(nb_id));
          }
        }

        // hnsw核心代码
        for (int lc = L; lc >= 0; lc--)
        {
          // test
          /*for (size_t i = 0; i < cur_batch_size; ++i)
          {
            size_t s_id = all_enter_points[i];
            auto s_level = hid_mapping.get_level(s_id);
            assert(s_level == lc);
          }*/

          // 从起点开始搜索一层结果
          parlay::sequence<parlay::sequence<indexType>> new_out_(cur_batch_size);
          t_beam.start();
          parlay::parallel_for(0, cur_batch_size, [&](size_t i)
                               {
                                 size_t node_id = inserts[i];
                                 size_t sp = all_enter_points[i];
                                 QueryParams QP((long)0, lc > levels[i] ? 1 : BP.L, (double)0.0, (long)Points.size(), (long)G.max_degree());
                                 auto [visited, bs_distance_comps] =
                                     // beam_search<Point, PointRange, indexType>(Points[index], G, Points, sp, QP);
                                     beam_search_rerank__<Point, QPoint, PR, QPR, indexType>(Points[hid_mapping.get_root_id(node_id)],
                                                                                             QPoints[hid_mapping.get_root_id(node_id)],
                                                                                             G,
                                                                                             Points,
                                                                                             QPoints,
                                                                                             sp,
                                                                                             QP,
                                                                                             &hid_mapping);
                                 
                                 BuildStats.increment_dist(node_id, bs_distance_comps);
                                 BuildStats.increment_visited(node_id, visited.size());
                                 if (lc <= levels[i])
                                 {
                                   // 给当前插入的点加入正向边
                                   long rp_distance_comps;
                                   std::tie(new_out_[i], rp_distance_comps) = prune.robustPrune(node_id, visited, G, Points, alpha, 0, true, &hid_mapping);
                                   BuildStats.increment_dist(node_id, rp_distance_comps);
                                   G[inserts[i]].update_neighbors(new_out_[i]);

                                   //test
                                   auto ilvl = hid_mapping.get_level(inserts[i]);
                                   auto slvl = hid_mapping.get_level(sp);
                                   assert(slvl == ilvl);
                                   for(auto a : new_out_[i]){
                                    auto nlvl = hid_mapping.get_level(a);
                                    if(nlvl != ilvl){
                                      std::cout<<L<<std::endl;
                                    }
                                    assert(nlvl == ilvl);
                                   }
                                  //  std::cout<<"add neighbors:"<<new_out_[i].size()<<std::endl;
                                 }
                                 // 设置下一个层级的起点
                                 auto lvl_sp = hid_mapping.get_level(sp);
                                 auto near_node_id = visited[0].first;
                                 auto lvl = hid_mapping.get_level(near_node_id);
                                 assert(lvl == lc && lvl == lvl_sp);
                                 if(lc > 0){
                                  auto next_ep = hid_mapping.get_id_in_pre_level(near_node_id);
                                  assert(next_ep < G.size() && next_ep >= 0);
                                  all_enter_points[i] = next_ep;
                                 }

                                 
                               });
          t_beam.stop();

          // make each edge bidirectional by first adding each new edge
          //(i,j) to a sequence, then semisorting the sequence by key values
          t_bidirect.start();

          auto flattened = parlay::delayed::flatten(parlay::tabulate(cur_batch_size, [&](size_t i)
                                                                     {
            indexType node_id = inserts[i];
            return parlay::delayed::map(new_out_[i], [=] (indexType ngh) {return std::pair(ngh, node_id);}); }));
          auto grouped_by = parlay::group_by_key(parlay::delayed::to_sequence(flattened));
          t_bidirect.stop();
          t_prune.start();
          // finally, add the bidirectional edges; if they do not make
          // the vertex exceed the degree bound, just add them to out_nbhs;
          // otherwise, use robustPrune on the vertex with user-specified alpha
          parlay::parallel_for(0, grouped_by.size(), [&](size_t j)
                               {
            auto &[node_id, candidates] = grouped_by[j];
	          size_t newsize = candidates.size() + G[node_id].size();
            if (newsize <= BP.R) {
	            add_neighbors_without_repeats(G[node_id], candidates);
	            G[node_id].update_neighbors(candidates);
            } else {
              auto [new_out_2_, distance_comps] = prune.robustPrune(node_id, std::move(candidates), G, Points, alpha, 0, true, &hid_mapping);
	            G[node_id].update_neighbors(new_out_2_);
             
              BuildStats.increment_dist(node_id, distance_comps);
            } });
          t_prune.stop();

          // 将节点id转到下一个层级
          for (size_t i = 0; i < cur_batch_size; ++i)
          {
            if (levels[i] >= lc)
            {
              auto node_id = inserts[i];
              assert(node_id < hid_mapping.cur_node_num);
              inserts[i] = hid_mapping.get_id_in_pre_level(node_id);
              if (lc > 0)
              {
                node_id = inserts[i];
                assert(node_id < hid_mapping.cur_node_num);
              }
            }
          }
        }

        if (new_level_nodes.size() > 0)
        {
          // new level
          ep = new_level_nodes[0];
          auto ep_level = hid_mapping.get_level(ep);
          assert(ep_level == L + 1);
          ++L;
        }
          

        if (print && BP.single_batch == 0)
        {
          auto ind = frac * n;
          if (floor <= ind && ceiling > ind)
          {
            frac += progress_inc;
            std::cout << "Pass " << 100 * frac << "% complete"
                      << std::endl;
          }
        }
        inc += 1;
      }
      // 记录hnsw的起点
      hid_mapping.set_sp(ep);

      t_beam.total();
      t_bidirect.total();
      t_prune.total();
    }
  };

} // end namespace
