#ifndef ALGORITHMS_CHECK_NN_RECALL_H_
#define ALGORITHMS_CHECK_NN_RECALL_H_

#include <algorithm>
#include <set>
#include <functional>
#include "beamSearch.h"
#include "csvfile.h"
#include "parse_results.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "types.h"
#include "stats.h"
#include "ivf.h"

namespace parlayANN {

template<typename PointRange, typename QPointRange, typename PQPointRange, typename indexType>
nn_result pq_checkRecall(const Graph<indexType> &G, Pivot& pivot,
                      const PointRange &Base_Points,
                      const PointRange &Query_Points,
                      const QPointRange &Q_Base_Points,
                      const groundTruth<indexType> &GT,
                      const indexType start_point,
                      const long k,
                      const QueryParams &QP,
                      const bool verbose
                    ) {
  using Point = typename PointRange::Point;

  if (GT.size() > 0 && k > GT.dimension()) {
    std::cout << k << "@" << k << " too large for ground truth data of size "
              << GT.dimension() << std::endl;
    abort();
  }

  parlay::sequence<parlay::sequence<indexType>> all_ngh;

  parlay::internal::timer t;
  float query_time;
  stats<indexType> QueryStats(Query_Points.size());
  QueryStats.clear();
  // to help clear the cache between runs
  auto volatile xx = parlay::random_permutation<long>(5000000);
  t.next_time();

  all_ngh = pq_qsearchAll<PointRange, QPointRange, PQPointRange, indexType>(Query_Points, G, pivot, 
                                                                              Base_Points, Q_Base_Points, 
                                                                              QueryStats, start_point, QP);

  query_time = t.next_time();
  return calRecall(all_ngh, Base_Points, Query_Points, GT,k, QP, verbose, QueryStats, query_time);
}

template<typename PointRange, typename QPointRange, typename PQPointRange, typename indexType, typename ivfIndexType>
nn_result ivfpq_checkRecall(
                      Graph<indexType> &centroidGraph, 
                      IVFReader<ivfIndexType>& ivfReader, 
                      Pivot& centroidPivot, Pivot& QPivot,
                      PointRange &Base_Points, QPointRange &Q_Points,
                      const PointRange &Query_Points,
                      const QPointRange &Q_Centroid_Points,
                      const groundTruth<indexType> &GT,
                      const indexType start_point,
                      const long k, const float pq_rerank_factor,
                      const QueryParams &QP,
                      const bool verbose
                    ) {
  using Point = typename PointRange::Point;

  if (GT.size() > 0 && k > GT.dimension()) {
    std::cout << k << "@" << k << " too large for ground truth data of size "
              << GT.dimension() << std::endl;
    abort();
  }

  parlay::sequence<parlay::sequence<indexType>> all_ngh;

  parlay::internal::timer t;
  float query_time;
  stats<indexType> QueryStats(Query_Points.size());
  QueryStats.clear();
  // to help clear the cache between runs
  auto volatile xx = parlay::random_permutation<long>(5000000);
  t.next_time();

  parlay::sequence<parlay::sequence<indexType>> all_centroids = pq_qsearchAll<PointRange, QPointRange, PQPointRange, indexType>(Query_Points, centroidGraph, centroidPivot, 
                                                                              Q_Centroid_Points, QueryStats, start_point, QP);

  // 加载ivf TODO 以后放到一个函数中
  {
    size_t pq_recall_size = k * pq_rerank_factor;
    all_ngh = parlay::sequence<parlay::sequence<indexType>>(Query_Points.size());

    // 一、先加载ivf key
    parlay::parallel_for(0, Query_Points.size(), [&](size_t i){
      parlay::sequence<indexType> cur_centroids = all_centroids[i];
      size_t recall_size = std::min((size_t)k * QP.rerank_factor, cur_centroids.size());
      std::vector<ivfIndexType> res;
      for(size_t j = 0; j < recall_size; ++j){
        auto cid = cur_centroids[j];
        res.push_back(cid);
      }
      all_ngh[i] = parlay::to_sequence(res);
    });

    if(ivfReader.is_use_cache()){
      auto ivfIds = parlay::unique(parlay::flatten(all_ngh));
      ivfReader.load2cache(ivfIds.data(), ivfIds.size());
    }
    
    // 二、计算ivf value
    parlay::parallel_for(0, Query_Points.size(), [&](size_t i){
      size_t recall_size = all_ngh[i].size();
      std::vector<ivfIndexType> cur_ivf;
      for(size_t j = 0; j < recall_size; ++j){
        size_t cid = all_ngh[i][j];
        auto [ivf_data, ivf_size] = ivfReader[cid];

        for(size_t jj = 0; jj < ivf_size; ++jj){
          ivfIndexType cur_index = ivf_data[jj];
          cur_ivf.push_back(cur_index);
          // res.push_back(std::pair<indexType, float>(cur_index, Query_Points[i].distance(Base_Points[cur_index])));
        }
      }

      // 1. 排序
      std::sort(cur_ivf.begin(), cur_ivf.end());
      // 2. 去重
      cur_ivf.erase(std::unique(cur_ivf.begin(), cur_ivf.end()),
                      cur_ivf.end());
      all_ngh[i] = parlay::to_sequence(cur_ivf);
    });
    
    // 三、开始初排序和精排序
    uint max_query_batch_size = parlay::num_workers();

    auto pstc = QPivot.pivots_num * QPivot.times * QPivot.chunk * QPivot.sub_chunk;
    PQPointRange cachePoint(max_query_batch_size, (unsigned int)pstc);
    auto cptc = QPivot.get_combine_pivots_num() * QPivot.times * QPivot.chunk;
    PQPointRange compressCachePoint(max_query_batch_size, (unsigned int)cptc);

    // 1、粗排序---------------------------------------------
    auto rough_sort = [&](size_t sid, size_t eid){
      parlay::parallel_for(sid, eid, [&](size_t i){
        auto q = cachePoint[parlay::worker_id()];
        auto cq = compressCachePoint[parlay::worker_id()];
        auto* tq = &q;
        QPivot.preCalculateDis(Query_Points[i], q.data());
        if(Q_Points.params.combine_dis){
          QPivot.combineDis(q.data(), cq.data());
          tq = &cq;
        }

        std::vector<std::pair<indexType, float>> res;
        res.reserve(all_ngh[i].size());
        for(auto cur_index : all_ngh[i])
          res.push_back(std::pair<indexType, float>(cur_index, Q_Points[cur_index].distance(*tq)));
        std::sort(res.begin(), res.end(), [](std::pair<indexType, float> a, std::pair<indexType, float> b){return a.second < b.second;});
        res.resize(pq_recall_size);
        all_ngh[i] = parlay::tabulate(res.size(), [&](size_t j){return res[j].first;});
      });      
    };
    auto rough_load_cache = [&](size_t sid, size_t eid){
      auto all_ivf = parlay::sequence<parlay::sequence<indexType>>(eid - sid);
      parlay::parallel_for(sid, eid, [&](size_t i){
        all_ivf[i-sid] = all_ngh[i];
      });
      auto first_ivf = parlay::unique(parlay::flatten(all_ivf)).to_vector();
      Q_Points.load2cache(first_ivf.data(), (ivfIndexType)first_ivf.size());
    };

    if(Q_Points.getMaxCache() == Q_Points.size()){
      rough_sort(0, Query_Points.size());
    } else {
      size_t pre_id = 0;
      size_t pre_cnt = 0;
      for(size_t ngh_id = 0; ngh_id < all_ngh.size(); ngh_id++){
        if(all_ngh[ngh_id].size() > Q_Points.getMaxCache()){
          std::cout<<"Q_Points.getMaxCache() is less then "<<all_ngh[ngh_id].size()<<"!"<<std::endl;
          abort();
        }
        if(pre_cnt + all_ngh[ngh_id].size() > Q_Points.getMaxCache()){
          rough_load_cache(pre_id, ngh_id);
          rough_sort(pre_id, ngh_id);
          pre_cnt = all_ngh[ngh_id].size();
          pre_id = ngh_id;
        } else {
          pre_cnt += all_ngh[ngh_id].size();
        }
      }
      rough_load_cache(pre_id, all_ngh.size());
      rough_sort(pre_id, all_ngh.size());
    }

    // 2、精排序-------------------------------------
    auto refined_sort = [&](size_t sid, size_t eid){
      parlay::parallel_for(sid, eid, [&](size_t i){
        std::vector<std::pair<indexType, float>> res;
        res.reserve(all_ngh[i].size());
        for(auto cur_index : all_ngh[i])
          res.push_back(std::pair<indexType, float>(cur_index, Base_Points[cur_index].distance(Query_Points[i])));
        std::sort(res.begin(), res.end(), [](std::pair<indexType, float> a, std::pair<indexType, float> b){return a.second < b.second;});
        all_ngh[i] = parlay::tabulate(res.size(), [&](size_t j){return res[j].first;});
      });      
    };
    auto refined_load_cache = [&](size_t sid, size_t eid){
      auto all_ivf = parlay::sequence<parlay::sequence<indexType>>(eid - sid);
      parlay::parallel_for(sid, eid, [&](size_t i){
        all_ivf[i-sid] = all_ngh[i];
      });
        auto second_ivf = parlay::unique(parlay::flatten(all_ivf)).to_vector();
        Base_Points.load2cache(second_ivf.data(), (ivfIndexType)second_ivf.size());
    };

    if(Base_Points.getMaxCache() == Base_Points.size()){
      refined_sort(0, Query_Points.size());
    } else {
      size_t pre_id = 0;
      size_t pre_cnt = 0;
      for(size_t ngh_id = 0; ngh_id < all_ngh.size(); ngh_id++){
        if(all_ngh[ngh_id].size() > Q_Points.getMaxCache()){
          std::cout<<"Q_Points.getMaxCache() is less then "<<all_ngh[ngh_id].size()<<"!"<<std::endl;
          abort();
        }
        if(pre_cnt + all_ngh[ngh_id].size() > Q_Points.getMaxCache()){
          refined_load_cache(pre_id, ngh_id);
          refined_sort(pre_id, ngh_id);
          pre_cnt = all_ngh[ngh_id].size();
          pre_id = ngh_id;
        } else {
          pre_cnt += all_ngh[ngh_id].size();
        }
      }
      refined_load_cache(pre_id, all_ngh.size());
      refined_sort(pre_id, all_ngh.size());
    }
  }

  query_time = t.next_time();
  return calRecall(all_ngh, Base_Points, Query_Points, GT,k, QP, verbose, QueryStats, query_time);
}

template<typename PointRange, typename QPointRange, typename QQPointRange, typename indexType>
nn_result checkRecall(const Graph<indexType> &G,
                      const PointRange &Base_Points,
                      const PointRange &Query_Points,
                      const QPointRange &Q_Base_Points,
                      const QPointRange &Q_Query_Points,
                      const QQPointRange &QQ_Base_Points,
                      const QQPointRange &QQ_Query_Points,
                      const groundTruth<indexType> &GT,
                      const bool random,
                      const long start_point,
                      const long k,
                      const QueryParams &QP,
                      const bool verbose,
                      HIdMapping* hid_mapping = nullptr,
                      parlay::sequence<parlay::sequence<indexType>>* query_start_points_ptr = nullptr
                      ) {
  

  if (GT.size() > 0 && k > GT.dimension()) {
    std::cout << k << "@" << k << " too large for ground truth data of size "
              << GT.dimension() << std::endl;
    abort();
  }

  parlay::sequence<parlay::sequence<indexType>> all_ngh;

  parlay::internal::timer t;
  float query_time;
  stats<indexType> QueryStats(Query_Points.size());
  QueryStats.clear();
  // to help clear the cache between runs
  auto volatile xx = parlay::random_permutation<long>(5000000);
  t.next_time();
  if(hid_mapping == nullptr){
    if (random) {
      all_ngh = beamSearchRandom(Query_Points, G, Base_Points, QueryStats, QP);
    } else {
      if(query_start_points_ptr == nullptr){
        all_ngh = qsearchAll<PointRange, QPointRange, QQPointRange, indexType>(Query_Points, Q_Query_Points, QQ_Query_Points,
                                                                              G,
                                                                              Base_Points, Q_Base_Points, QQ_Base_Points,
                                                                              QueryStats, start_point, QP);
      } else {
        all_ngh = qsearchAll<PointRange, QPointRange, QQPointRange, indexType>(Query_Points, Q_Query_Points, QQ_Query_Points,
                                                                              G,
                                                                              Base_Points, Q_Base_Points, QQ_Base_Points,
                                                                              QueryStats, *query_start_points_ptr, QP);
      }

    }
  } else {
    // 分层查询出结果
    all_ngh = hid_qsearchAll<PointRange, QPointRange, QQPointRange, indexType>(Query_Points, Q_Query_Points, QQ_Query_Points,
                                                                                G,
                                                                                Base_Points, Q_Base_Points, QQ_Base_Points,
                                                                                QueryStats, QP, hid_mapping);   
  }

  query_time = t.next_time();
  
  return calRecall(all_ngh, Base_Points, Query_Points, GT,k, QP, verbose, QueryStats, query_time);
}

template<typename PointRange, typename indexType>
nn_result calRecall(parlay::sequence<parlay::sequence<indexType>>& all_ngh,
                    const PointRange &Base_Points,
                    const PointRange &Query_Points,
                    const groundTruth<indexType> &GT,
                    const long k,
                    const QueryParams &QP,
                    const bool verbose, stats<indexType>& QueryStats, float query_time){
  using Point = typename PointRange::Point;
  float recall = 0.0;
  //TODO deprecate this after further testing
  bool dists_present = true;
  if (GT.size() > 0 && !dists_present) {
    size_t n = Query_Points.size();
    int numCorrect = 0;
    for (indexType i = 0; i < n; i++) {
      std::set<indexType> reported_nbhs;
      if (all_ngh[i].size() != k) {
        std::cout << "bad number of neighbors reported: " << all_ngh[i].size() << std::endl;
        abort();
      }
      for (indexType l = 0; l < k; l++) reported_nbhs.insert((all_ngh[i])[l]);
      if (reported_nbhs.size() != k) {
        std::cout << "duplicate entries in reported neighbors" << std::endl;
        abort();
      }
      for (indexType l = 0; l < k; l++) {
        if (reported_nbhs.find((GT.coordinates(i,l))) !=
            reported_nbhs.end()) {
          numCorrect += 1;
        }
      }
    }
    recall = static_cast<float>(numCorrect) / static_cast<float>(k * n);
  } else if (GT.size() > 0 && dists_present) {
    size_t n = Query_Points.size();

    int numCorrect = 0;
    for (indexType i = 0; i < n; i++) {
      parlay::sequence<int> results_with_ties;
      for (indexType l = 0; l < k; l++)
        results_with_ties.push_back(GT.coordinates(i,l));
      Point qp = Query_Points[i];
      float last_dist = qp.distance(Base_Points[GT.coordinates(i, k-1)]);
      //float last_dist = GT.distances(i, k-1);
      for (indexType l = k; l < GT.dimension(); l++) {
        //if (GT.distances(i,l) == last_dist) {
        if (qp.distance(Base_Points[GT.coordinates(i, l)]) == last_dist) {
          results_with_ties.push_back(GT.coordinates(i,l));
        }
      }
      std::set<int> reported_nbhs;
      for (indexType l = 0; l < k; l++) reported_nbhs.insert((all_ngh[i])[l]);
      for (indexType l = 0; l < results_with_ties.size(); l++) {
        if (reported_nbhs.find(results_with_ties[l]) != reported_nbhs.end()) {
          numCorrect += 1;
        }
      }
    }
    recall = static_cast<float>(numCorrect) / static_cast<float>(k * n);
  }
  float QPS = Query_Points.size() / query_time;
  if (verbose)
    std::cout << "search: Q=" << QP.beamSize << ", k=" << QP.k
              << ", limit=" << QP.limit
      //<< ", dlimit=" << QP.degree_limit
              << ", recall=" << recall
              << ", visited=" << QueryStats.visited_stats()[0]
              << ", comparisons=" << QueryStats.dist_stats()[0]
              << ", QPS=" << QPS
              << ", ctime=" << 1/(QPS*QueryStats.dist_stats()[0]) * 1e9 << std::endl;

  auto stats_ = {QueryStats.dist_stats(), QueryStats.visited_stats()};
  parlay::sequence<indexType> stats = parlay::flatten(stats_);
  nn_result N(recall, stats, QPS, k, QP.beamSize, QP.cut, Query_Points.size(), QP.limit, QP.degree_limit, k);
  return N; 
}

void write_to_csv(std::string csv_filename, parlay::sequence<float> buckets,
                  parlay::sequence<nn_result> results, Graph_ G) {
  csvfile csv(csv_filename);
  csv << "GRAPH"
      << "Parameters"
      << "Size"
      << "Build time"
      << "Avg degree"
      << "Max degree"
      << "Isolated"
      << endrow;
  csv << G.name << G.params << G.size << G.time << G.avg_deg << G.max_deg << G.isolated_cnt
      << endrow;
  csv << endrow;
  csv << "Num queries"
      << "Target recall"
      << "Actual recall"
      << "QPS"
      << "Average Cmps"
      << "Tail Cmps"
      << "Average Visited"
      << "Tail Visited"
      << "k"
      << "Q"
      << "cut" << endrow;
  for (int i = 0; i < results.size(); i++) {
    nn_result N = results[i];
    csv << N.num_queries << buckets[i] << N.recall << N.QPS << N.avg_cmps
        << N.tail_cmps << N.avg_visited << N.tail_visited << N.k << N.beamQ
        << N.cut << endrow;
  }
  csv << endrow;
  csv << endrow;
}

parlay::sequence<long> calculate_limits(size_t upper_bound) {
  parlay::sequence<long> L(6);
  for (float i = 0; i < 6; i++) {
    L[i] = (long)((4 + i) * ((float) upper_bound) * .1);
    //std::cout << L[i - 1] << std::endl;
  }
  //auto limits = parlay::remove_duplicates(L);
  return L; //limits;
}

template<typename PointRange, typename indexType>
void search_and_parse(Graph_ G_,
                      Graph<indexType> &G,
                      PointRange &Base_Points,
                      PointRange &Query_Points,
                      groundTruth<indexType> GT, const char* res_file, long k,
                      bool verbose = false,
                      long fixed_beam_width = 0) {
  search_and_parse(G_, G, Base_Points, Query_Points, Base_Points, Query_Points, Base_Points, Query_Points, GT, res_file, k, false, 0u, verbose, fixed_beam_width);
}

template<typename indexType>
void _search_and_parse(Graph_ G_,
                      Graph<indexType> &G,
                      const char* res_file, long k,
                      long fixed_beam_width,
                      int rerank_factor,
                      std::function<nn_result(const long, const QueryParams)> check) {
  parlay::sequence<nn_result> results;
  std::vector<long> beams;
  std::vector<long> allr;
  std::vector<double> cuts;

  QueryParams QP;
  QP.limit = (long) G.size();
  QP.rerank_factor = rerank_factor;
  QP.degree_limit = (long) G.max_degree();
  beams = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 32,
    34, 36, 38, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100, 120, 140, 160,
    180, 200, 225, 250, 275, 300, 375, 500, 750, 1000};
  if(k==0) allr = {10};
  else allr = {k};
  cuts = {1.35};

  if (fixed_beam_width != 0) {
    QP.k = allr[0];
    QP.cut = cuts[0];
    QP.beamSize = fixed_beam_width;
    for (int i = 0; i < 5; i++)
      check(QP.k, QP);
  } else {
    for (long r : allr) {
      results.clear();
      QP.k = r;
      for (float cut : cuts){
        QP.cut = cut;
        for (float Q : beams){
          QP.beamSize = Q;
          if (Q >= r){
            results.push_back(check(r, QP));
          }
        }
      }

      // check "limited accuracy"
      // {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 30, 35}; //
      parlay::sequence<long> limits = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 30, 35};

      QP = QueryParams(r, r, 1.35, (long) G.size(), (long) G.max_degree());
      for(long l : limits){
        QP.limit = l;
        QP.beamSize = std::max<long>(l, r);
        //for(long dl : degree_limits){
        QP.degree_limit = std::min<int>(G.max_degree(), 5 * l);
        results.push_back(check(r, QP));
      }
      // check "best accuracy"
      QP = QueryParams((long) 100, (long) 1000, (double) 10.0, (long) G.size(), (long) G.max_degree());
      results.push_back(check(r, QP));

      parlay::sequence<float> buckets =  {.1, .2, .3,  .4,  .5,  .6, .7, .75,  .8, .85,
        .9, .93, .95, .97, .98, .99, .995, .999, .9995,
        .9999, .99995, .99999};
      auto [res, ret_buckets] = parse_result(results, buckets);
      std::cout << std::endl;
      if (res_file != NULL)
        write_to_csv(std::string(res_file), ret_buckets, res, G_);
    }
  }
}

template<typename PointRange, typename QPointRange, typename QQPointRange, typename indexType>
void search_and_parse(Graph_ G_,
                      Graph<indexType> &G,
                      PointRange &Base_Points,
                      PointRange &Query_Points,
                      QPointRange &Q_Base_Points,
                      QPointRange &Q_Query_Points,
                      QQPointRange &QQ_Base_Points,
                      QQPointRange &QQ_Query_Points,
                      groundTruth<indexType> GT, const char* res_file, long k,
                      bool random = true,
                      indexType start_point = 0,
                      bool verbose = false,
                      long fixed_beam_width = 0,
                      int rerank_factor = 100,
                      HIdMapping* hid_mapping = nullptr,
                      parlay::sequence<parlay::sequence<indexType>>* query_start_points_ptr = nullptr) {
  auto check = [&] (const long k, const QueryParams QP) {
    return checkRecall(G,
                       Base_Points, Query_Points,
                       Q_Base_Points, Q_Query_Points,
                       QQ_Base_Points, QQ_Query_Points,
                       GT,
                       random,
                       start_point, k, QP, verbose, hid_mapping, query_start_points_ptr);
                       
  };
  _search_and_parse(G_, G, res_file, k, fixed_beam_width, rerank_factor, check);

}

template<typename PointRange, typename QPointRange, typename PQPointRange, typename indexType>
void pq_search_and_parse(Graph_ G_, 
                      Graph<indexType> &G, Pivot& pivot,
                      PointRange &Base_Points,
                      PointRange &Query_Points,
                      const QPointRange &Q_Base_Points,
                      groundTruth<indexType> GT, const char* res_file, long k,
                      indexType start_point = 0,
                      bool verbose = false,
                      long fixed_beam_width = 0,
                      int rerank_factor = 100
){

  auto check = [&] (const long k, const QueryParams QP) {
    return pq_checkRecall<PointRange, QPointRange, PQPointRange, indexType>(
                       G, pivot,
                       Base_Points, Query_Points,
                       Q_Base_Points, 
                       GT,
                       start_point, k, QP, verbose);
                       
  };
  _search_and_parse(G_, G, res_file, k, fixed_beam_width, rerank_factor, check);
}

template<typename PointRange, typename QPointRange, typename PQPointRange, typename indexType, typename ivfIndexType>
void ivfpq_search_and_parse(Graph_ G_, 
                      Graph<indexType> &centroidGraph, IVFReader<ivfIndexType>& ivfReader, 
                      Pivot& centroidPivot, Pivot& QPivot, PointRange &Base_Points, QPointRange &Q_Points,
                      PointRange &Query_Points, const QPointRange &Q_Centroid_Points,
                      groundTruth<indexType> GT, const char* res_file, long k, float pq_rerank_factor,
                      indexType start_point = 0,
                      bool verbose = false,
                      long fixed_beam_width = 0,
                      int rerank_factor = 100
){

  auto check = [&] (const long k, const QueryParams QP) {
    return ivfpq_checkRecall<PointRange, QPointRange, PQPointRange, indexType, ivfIndexType>(
                       centroidGraph, ivfReader, centroidPivot, QPivot,
                       Base_Points, Q_Points, Query_Points,
                       Q_Centroid_Points, GT,
                       start_point, k, pq_rerank_factor, QP, verbose);
                       
  };
  _search_and_parse(G_, centroidGraph, res_file, k, fixed_beam_width, rerank_factor, check);
}

} // end namespace

#endif // ALGORITHMS_CHECK_NN_RECALL_H_
