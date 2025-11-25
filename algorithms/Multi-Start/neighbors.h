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

#include <algorithm>

#include "../utils/beamSearch.h"
#include "../utils/check_nn_recall.h"
#include "../utils/parse_results.h"
#include "../utils/mips_point.h"
#include "../utils/euclidian_point.h"
#include "../utils/jl_point.h"
#include "../utils/stats.h"
#include "../utils/types.h"
#include "../utils/graph.h"
#include "index.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"



namespace parlayANN {

template<typename PointRange, typename QPointRange, typename QQPointRange, typename indexType>
void ANN_Quantized(Graph<indexType> &G, long k, BuildParams &BP,
                   PointRange &Query_Points, QPointRange &Q_Query_Points, QQPointRange &QQ_Query_Points,
                   groundTruth<indexType> GT, const char* meta_file, char *res_file,
                   bool graph_built,
                   PointRange &Points, QPointRange &Q_Points, QQPointRange &QQ_Points) {
  parlay::internal::timer t("ANN");
  using pid = std::pair<indexType, typename QPointRange::Point::distanceType>;

  // 读取字典、轴以及量化编码-------------------------------------------------------------------
  std::string meta_str(meta_file);
  auto first = meta_str.find('[', 0);
  auto last = meta_str.find(']', 0);
  if(first == std::string::npos || first >= last){
    std::cout<<"meta_file:"<<meta_file<<" Error!";
  }
  auto context = meta_str.substr(first + 1, last - first - 1);
  auto base_path = meta_str.substr(0, first);
  auto mid = context.find(',', 0);
  auto first_name = context.substr(0, mid);
  auto second_name = context.substr(mid+1);
  
  /*std::vector<std::string> dict_files = {base_path + first_name + ".dict", base_path + second_name + ".dict"};
  std::vector<std::string> quant_files = {base_path + first_name + ".quant", base_path + second_name + ".quant"};
  std::vector<std::string> piv_files = {base_path + first_name + ".piv", base_path + second_name + ".piv"};
  std::vector<std::vector<float>> pivots = std::vector<std::vector<float>>(2);
  std::vector<std::vector<uint>> order_ids = std::vector<std::vector<uint>>(2);
  uint dim;
  std::vector<uint> times = {0, 0};
  std::vector<uint> pivots_num = {0, 0};
  std::vector<uint> chunk = {0, 0};

  for(int i = 0; i < 2; ++i){
    std::ifstream reader(piv_files[i].c_str());
    auto& cur_times = times[i];
    auto& cur_pivots_num = pivots_num[i];
    reader.read((char*)&cur_times, sizeof(uint));
    reader.read((char*)&cur_pivots_num, sizeof(uint));
    reader.read((char*)&dim, sizeof(uint));
    auto& cur_chunk = chunk[i];
    reader.read((char*)&cur_chunk, sizeof(uint));
    pivots[i] = std::vector<float>(cur_times * cur_pivots_num * dim);
    order_ids[i] = std::vector<uint>(cur_times * dim);
    reader.read((char*)order_ids[i].data(), sizeof(float) * cur_times * dim);
    reader.read((char*)pivots[i].data(), sizeof(float) * cur_times * cur_pivots_num * dim);
    reader.close();      
  }
  auto qd1 = DictForest<pid>(dict_files[0].c_str());
  auto qd2 = DictForest<pid>(dict_files[1].c_str());
  std::vector<DictForest<pid>*> quant_dict = {&qd1, &qd2};
  quant_dict.push_back(&qd1);
  quant_dict.push_back(&qd2);*/
  // -----------------------------------------------------------------------------------------------

  bool verbose = BP.verbose;
  using findex = knn_index<QPointRange, QQPointRange, indexType>;
  findex I(BP);
  indexType start_point;
  double idx_time;
  size_t isolate_node_num = 0;
  stats<unsigned int> BuildStats(G.size());
  if(graph_built){
    idx_time = 0;
    start_point = 0;
  } else{
    // isolate_node_num = I.build_index(G, quant_files, quant_dict, chunk, Q_Points, QQ_Points, BuildStats);
    isolate_node_num = I.build_index(G, base_path + first_name, Q_Points, QQ_Points, BuildStats);
    start_point = I.get_start();
    idx_time = t.next_time();
  }
  std::cout << "start index = " << start_point << std::endl;

  std::string name = "scour";
  std::string params =
    "R = " + std::to_string(BP.R) + ", L = " + std::to_string(BP.L);
  auto [avg_deg, max_deg] = graph_stats_(G);
  auto vv = BuildStats.visited_stats();
  std::cout << "Average visited: " << vv[0] << ", Tail visited: " << vv[1]
            << std::endl;

  Graph_ G_(name, params, G.size(), avg_deg, max_deg, idx_time, isolate_node_num);
  G_.print();

  long build_num_distances = parlay::reduce(parlay::map(BuildStats.distances,
                                                        [] (auto x) {return (long) x;}));

  

  // query---------------
  if(Query_Points.size() != 0) {

    // 获取入口点，这是关键代码--------------------------------
    std::cout<<"start calculate start points ....."<<std::endl;
    parlay::sequence<parlay::sequence<indexType>> query_start_points(Query_Points.size());
    uint n, d;
    std::string sp_file = base_path + second_name;
    std::ifstream reader(sp_file.c_str());
    reader.read((char*)&n, sizeof(uint));
    reader.read((char*)&d, sizeof(uint));
    reader.seekg(sizeof(uint) * 2 + n * sizeof(indexType));
    std::vector<indexType> q_start_points_vec = std::vector<indexType>(n * d);
    reader.read((char*)q_start_points_vec.data(), n * d * sizeof(indexType));
    reader.close();

    parlay::parallel_for(0, Query_Points.size(), [&](size_t qid){
      auto* cur_start_point = q_start_points_vec.data() + qid * d;
      uint cur_sp_num = 0;
      while(cur_sp_num < d && cur_start_point[cur_sp_num] != -1){
          cur_sp_num++;
      }
            
      if(cur_sp_num == 0){
        std::cout<<"qid="<<qid<<" start point num = 0!"<<std::endl;
        abort();
      }
      query_start_points[qid] = parlay::tabulate(cur_sp_num, [&](size_t j){return cur_start_point[j];});  
    });
    std::cout<<"finish cal start points!"<<std::endl;

    search_and_parse<PointRange, QPointRange, QQPointRange, indexType>(
                     G_, G,
                     Points, Query_Points,
                     Q_Points, Q_Query_Points,
                     QQ_Points, QQ_Query_Points,
                     GT,
                     res_file, k, false, start_point,
                     verbose, BP.Q, BP.rerank_factor, nullptr, &query_start_points);
  } else if (BP.self) {
    if (BP.range) {
      parlay::internal::timer t_range("range search time");
      double radius = BP.radius;
      double radius_2 = BP.radius_2;
      std::cout << "radius = " << radius << " radius_2 = " << radius_2 << std::endl;
      QueryParams QP;
      long n = Points.size();
      parlay::sequence<long> counts(n);
      parlay::sequence<long> distance_comps(n);
      parlay::parallel_for(0, G.size(), [&] (long i) {
        parlay::sequence<indexType> pts;
        pts.push_back(Points[i].id());
        auto [r, dc] = range_search(Points[i], G, Points, pts, radius, radius_2, QP, true);
        counts[i] = r.size();
        distance_comps[i] = dc;});
      t_range.total();
      long range_num_distances = parlay::reduce(distance_comps);

      std::cout << "edges within range: " << parlay::reduce(counts) << std::endl;
      std::cout << "distance comparisons during build = " << build_num_distances << std::endl;
      std::cout << "distance comparisons during range = " << range_num_distances << std::endl;
    }
  }
}

template<typename Point, typename PointRange_, typename indexType>
void ANN(Graph<indexType> &G, long k, BuildParams &BP,
         PointRange_ &Query_Points,
         groundTruth<indexType> GT, const char* meta_file, char *res_file,
         bool graph_built, PointRange_ &Points) {
  if (BP.quantize != 0) {
    std::cout << "quantizing build and first pass of search to 1 byte" << std::endl;
    if (Point::is_metric()) {
      using QT = uint8_t;
      using QPoint = Euclidian_Point<QT>;
      using QPR = PointRange<QPoint>;
      QPR Q_Points(Points);  // quantized to one byte
      QPR Q_Query_Points(Query_Points, Q_Points.params);
      if (BP.quantize == 1) {
        ANN_Quantized(G, k, BP, Query_Points, Q_Query_Points, Q_Query_Points,
                      GT, meta_file, res_file, graph_built, Points, Q_Points, Q_Points);
      } else if (BP.quantize == 2) {
        using QQPoint = Euclidean_Bit_Point;
        using QQPR = PointRange<QQPoint>;
        QQPR QQ_Points(Points);
        QQPR QQ_Query_Points(Query_Points, QQ_Points.params);
        ANN_Quantized(G, k, BP, Query_Points, Q_Query_Points, QQ_Query_Points,
                      GT, meta_file, res_file, graph_built, Points, Q_Points, QQ_Points);
      } else if (BP.quantize == 3) {
        using QQPoint = Euclidean_JL_Sparse_Point<1024>;
        using QQPR = PointRange<QQPoint>;
        QQPR QQ_Points(Points);
        QQPR QQ_Query_Points(Query_Points, QQ_Points.params);
        ANN_Quantized(G, k, BP, Query_Points, Q_Query_Points, QQ_Query_Points,
                      GT, meta_file, res_file, graph_built, Points, Q_Points, QQ_Points);
      }
    } else {
      using QT = int8_t;
      //using QPoint = Euclidian_Point<uint8_t>;
      using QPoint = Quantized_Mips_Point<8,true,255>;
      using QPR = PointRange<QPoint>;
      QPR Q_Points(Points);
      QPR Q_Query_Points(Query_Points, Q_Points.params);
      if (BP.quantize == 1) {
        ANN_Quantized(G, k, BP, Query_Points, Q_Query_Points, Q_Query_Points,
                      GT, meta_file, res_file, graph_built, Points, Q_Points, Q_Points);
      } else if (BP.quantize == 2) {
        using QQPoint = Mips_Bit_Point;
        using QQPR = PointRange<QQPoint>;
        QQPR QQ_Points(Points);
        QQPR QQ_Query_Points(Query_Points, QQ_Points.params);
        ANN_Quantized(G, k, BP, Query_Points, Q_Query_Points, QQ_Query_Points,
                      GT, meta_file, res_file, graph_built, Points, Q_Points, QQ_Points);
      } else if (BP.quantize == 3) {
        using QQPoint = Mips_2Bit_Point;
        using QQPR = PointRange<QQPoint>;
        QQPR QQ_Points(Points);
        QQPR QQ_Query_Points(Query_Points, QQ_Points.params);
        ANN_Quantized(G, k, BP, Query_Points, Q_Query_Points, QQ_Query_Points,
                      GT, meta_file, res_file, graph_built, Points, Q_Points, QQ_Points);
      } else if (BP.quantize == 4) {
        using QQPoint = Mips_JL_Bit_Point<512>;
        using QQPR = PointRange<QQPoint>;
        QQPR QQ_Points(Points);
        QQPR QQ_Query_Points(Query_Points, QQ_Points.params);
        ANN_Quantized(G, k, BP, Query_Points, Q_Query_Points, QQ_Query_Points,
                      GT, meta_file, res_file, graph_built, Points, Q_Points, QQ_Points);
      } else if (BP.quantize == 5) {
        using QQPoint = Mips_JL_Sparse_Point<1500>;
        using QQPR = PointRange<QQPoint>;
        QQPR QQ_Points(Points);
        QQPR QQ_Query_Points(Query_Points, QQ_Points.params);
        ANN_Quantized(G, k, BP, Query_Points, Q_Query_Points, QQ_Query_Points,
                      GT, meta_file, res_file, graph_built, Points, Q_Points, QQ_Points);
      }
    }
  } else {
    ANN_Quantized(G, k, BP, Query_Points, Query_Points, Query_Points,
                  GT, meta_file, res_file, graph_built, Points, Points, Points);
  }
}

} // end namespace
