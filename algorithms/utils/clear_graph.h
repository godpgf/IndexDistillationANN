#ifndef CLEAR_GRAPH_H_
#define CLEAR_GRAPH_H_

#include <algorithm>
#include <functional>
#include <random>
#include <set>
#include <unordered_set>
#include <queue>
#include <map>

#include "parlay/io.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "types.h"
#include "graph.h"
#include "stats.h"
#include "dijkstra.h"

namespace parlayANN {

template<typename indexType>
struct PathElement{
    PathElement(indexType nodeId, indexType preNodeId, indexType preEdgeId) : nodeId(nodeId), preNodeId(preNodeId), preEdgeId(preEdgeId){}
    PathElement() : nodeId(0), preNodeId(0), preEdgeId(0){}
    indexType nodeId;
    indexType preNodeId;
    indexType preEdgeId;
};

template<typename indexType, class GT>
void init_path_graph(const GT &G, GT &pathG){
  parlay::parallel_for(0, G.size(), [&](long i){
    parlay::sequence<indexType> data = parlay::tabulate(G[i].size(),[](long j){return static_cast<indexType>(0);});
    pathG[i].update_neighbors(data);
  });
}

template<typename indexType, typename Point, class GT>
void fill_path_graph(parlay::sequence<PathElement<indexType>>& frontier, parlay::sequence<PathElement<indexType>>& visited, GT &pathG, int topk=-1){
  std::map<indexType, indexType> nodeId2visitedId;
  for(indexType i = 0; i < visited.size(); ++i)
    nodeId2visitedId.insert(std::pair<indexType, indexType>(visited[i].nodeId, i));
  if(topk < 0 || topk > frontier.size())
    topk = frontier.size();
  if(frontier.size() < topk)
    topk = frontier.size();
  for(indexType i = 0; i < topk; ++i){
    auto d = frontier[i];
    while(d.preNodeId != d.nodeId){
      pathG[d.preNodeId].fill(d.preEdgeId, pathG[d.preNodeId][d.preEdgeId]+1);
      auto it = nodeId2visitedId.find(d.preNodeId);
      if(it == nodeId2visitedId.end()){
        break;
      } else {
        d = visited[(*it).second];
      }
    }
  }
}


// main beam search
template<typename indexType, typename Point, typename PointRange,
         typename QPoint, typename QPointRange, class GT>
std::pair<std::pair<parlay::sequence<std::pair<PathElement<indexType>, typename Point::distanceType>>,
                    parlay::sequence<std::pair<PathElement<indexType>, typename Point::distanceType>>>,
          size_t>
filtered_beam_search_path(const GT &G,
                     const Point p,  const PointRange &Points,
                     const QPoint qp, const QPointRange &Q_Points,
                     const parlay::sequence<indexType> starting_points,
                     const QueryParams &QP,
                     bool use_filtering = false
                     ) {
  using dtype = typename Point::distanceType;
  using id_dist = std::pair<PathElement<indexType>, dtype>;
  int beamSize = QP.beamSize;

  if (starting_points.size() == 0) {
    std::cout << "beam search expects at least one start point" << std::endl;
    abort();
  }

  // compare two (node_id,distance) pairs, first by distance and then id if
  // equal
  using distanceType = typename Point::distanceType;
  auto less = [&](id_dist a, id_dist b) {
    return a.second < b.second || (a.second == b.second && a.first.nodeId < b.first.nodeId);
  };

  // used as a hash filter (can give false negative -- i.e. can say
  // not in table when it is)
  int bits = std::max<int>(10, std::ceil(std::log2(beamSize * beamSize)) - 2);
  std::vector<indexType> hash_filter(1 << bits, -1);
  auto has_been_seen = [&](indexType a) -> bool {
    int loc = parlay::hash64_2(a) & ((1 << bits) - 1);
    if (hash_filter[loc] == a) return true;
    hash_filter[loc] = a;
    return false;
  };

  // Frontier maintains the closest points found so far and its size
  // is always at most beamSize.  Each entry is a (id,distance) pair.
  // Initialized with starting points and kept sorted by distance.
  std::vector<id_dist> frontier;
  frontier.reserve(beamSize);
  for (auto q : starting_points) {
    auto qe = PathElement(static_cast<indexType>(q), static_cast<indexType>(q), static_cast<indexType>(0));
    frontier.push_back(id_dist(qe, Points[q].distance(p)));
    has_been_seen(q);
  }
  std::sort(frontier.begin(), frontier.end(), less);

  // The subset of the frontier that has not been visited
  // Use the first of these to pick next vertex to visit.
  std::vector<id_dist> unvisited_frontier(beamSize);
  for (int i=0; i < frontier.size(); i++)
    unvisited_frontier[i] = frontier[i];

  // maintains sorted set of visited vertices (id-distance pairs)
  std::vector<id_dist> visited;
  visited.reserve(2 * beamSize);

  // counters
  size_t dist_cmps = starting_points.size();
  size_t full_dist_cmps = starting_points.size();
  int remain = frontier.size();
  int num_visited = 0;

  // used as temporaries in the loop
  std::vector<id_dist> new_frontier(2 * std::max<size_t>(beamSize,starting_points.size()) +
                                    G.max_degree());
  std::vector<id_dist> candidates;
  candidates.reserve(G.max_degree() + beamSize);
  std::vector<PathElement<indexType>> filtered;
  filtered.reserve(G.max_degree());
  std::vector<PathElement<indexType>> pruned;
  pruned.reserve(G.max_degree());

  dtype filter_threshold_sum = 0.0;
  int filter_threshold_count = 0;
  dtype filter_threshold;

  // offset into the unvisited_frontier vector (unvisited_frontier[offset] is the next to visit)
  int offset = 0;

  // The main loop.  Terminate beam search when the entire frontier
  // has been visited or have reached max_visit.
  while (remain > offset && num_visited < QP.limit) {
    // the next node to visit is the unvisited frontier node that is closest to p
    id_dist current = unvisited_frontier[offset];
    G[current.first.nodeId].prefetch();
    // add to visited set
    auto position = std::upper_bound(visited.begin(), visited.end(), current, less);
    visited.insert(position, current);
    num_visited++;
    bool frontier_full = frontier.size() == beamSize;

    // if using filtering based on lower quality distances measure, then maintain the average
    // of low quality distance to the last point in the frontier (if frontier is full)
    if (use_filtering && frontier_full) {
      filter_threshold_sum += Q_Points[frontier.back().first.nodeId].distance(qp);
      filter_threshold_count++;
      filter_threshold = filter_threshold_sum / filter_threshold_count;
    }

    // keep neighbors that have not been visited (using approximate
    // hash). Note that if a visited node is accidentally kept due to
    // approximate hash it will be removed below by the union.
    pruned.clear();
    filtered.clear();
    long num_elts = std::min<long>(G[current.first.nodeId].size(), QP.degree_limit);
    for (indexType i=0; i<num_elts; i++) {
      auto a = G[current.first.nodeId][i];
      if (has_been_seen(a) || Points[a].same_as(p)) continue;  // skip if already seen
      Q_Points[a].prefetch();
      pruned.push_back(PathElement<indexType>(static_cast<indexType>(a), static_cast<indexType>(current.first.nodeId), static_cast<indexType>(i)));
    }
    dist_cmps += pruned.size();

    // filter using low-quality distance
    if (use_filtering && frontier_full) {
      for (auto a : pruned) {
        if (frontier_full && Q_Points[a.nodeId].distance(qp) >= filter_threshold) continue;
        filtered.push_back(a);
        Points[a.nodeId].prefetch();
      }
    } else std::swap(filtered, pruned);

    // Further remove if distance is greater than current
    // furthest distance in current frontier (if full).
    distanceType cutoff = (frontier_full
                           ? frontier[frontier.size() - 1].second
                           : (distanceType)std::numeric_limits<int>::max());
    for (auto a : filtered) {
      distanceType dist = Points[a.nodeId].distance(p);
      full_dist_cmps++;
      // skip if frontier not full and distance too large
      if (dist >= cutoff) continue;
      candidates.push_back(std::pair{a, dist});
    }
    // If candidates insufficently full then skip rest of step until sufficiently full.
    // This iproves performance for higher accuracies (e.g. beam sizes of 100+)
    if (candidates.size() == 0 ||
        (QP.limit >= 2 * beamSize &&
         candidates.size() < beamSize/8 &&
         offset + 1 < remain)) {
      offset++;
      continue;
    }
    offset = 0;

    // sort the candidates by distance from p,
    // and remove any duplicates (to be robust for neighbor lists with duplicates)
    std::sort(candidates.begin(), candidates.end(), less);
    auto candidates_end = std::unique(candidates.begin(), candidates.end(),
                                      [] (auto a, auto b) {return a.first.nodeId == b.first.nodeId;});

    // union the frontier and candidates into new_frontier, both are sorted
    auto new_frontier_size =
      std::set_union(frontier.begin(), frontier.end(), candidates.begin(),
                     candidates_end, new_frontier.begin(), less) -
      new_frontier.begin();
    candidates.clear();

    // trim to at most beam size
    new_frontier_size = std::min<size_t>(beamSize, new_frontier_size);

    // if a k is given (i.e. k != 0) then trim off entries that have a
    // distance greater than cut * current-kth-smallest-distance.
    // Only used during query and not during build.
    if (QP.k > 0 && new_frontier_size > QP.k && Points[0].is_metric())
      new_frontier_size = std::max<indexType>(
        (std::upper_bound(new_frontier.begin(),
                          new_frontier.begin() + new_frontier_size,
                          id_dist{PathElement<indexType>(0, 0, 0), QP.cut * new_frontier[QP.k].second}, less) -
         new_frontier.begin()), frontier.size());

    // copy new_frontier back to the frontier
    frontier.clear();
    for (indexType i = 0; i < new_frontier_size; i++)
      frontier.push_back(new_frontier[i]);

    // get the unvisited frontier
    remain = (std::set_difference(frontier.begin(),
                                  frontier.begin() + std::min<long>(frontier.size(), QP.beamSize),
                                  visited.begin(),
                                  visited.end(),
                                  unvisited_frontier.begin(), less) -
              unvisited_frontier.begin());
  }


  return std::make_pair(std::make_pair(parlay::to_sequence(frontier),
                                       parlay::to_sequence(visited)),
                        full_dist_cmps);
}

// version without filtering
template<typename Point, typename PointRange, typename indexType>
std::pair<std::pair<parlay::sequence<std::pair<PathElement<indexType>, typename Point::distanceType>>,
                    parlay::sequence<std::pair<PathElement<indexType>, typename Point::distanceType>>>, size_t>
beam_search_path(const Point p, const Graph<indexType> &G, const PointRange &Points,
            const parlay::sequence<indexType> starting_points, const QueryParams &QP) {
  return filtered_beam_search_path(G, p, Points, p, Points, starting_points, QP, false);
}

// backward compatibility (for hnsw)
template<typename indexType, typename Point, typename PointRange, class GT>
std::pair<std::pair<parlay::sequence<std::pair<PathElement<indexType>, typename Point::distanceType>>, parlay::sequence<std::pair<PathElement<indexType>, typename Point::distanceType>>>, size_t>
beam_search_impl_path(Point p, GT &G, PointRange &Points,
                 parlay::sequence<indexType> starting_points, QueryParams &QP) {
  return filtered_beam_search_path(G, p, Points, p, Points, starting_points, QP, false);
}

// pass single start point
template<typename Point, typename PointRange, typename indexType>
std::pair<std::pair<parlay::sequence<std::pair<PathElement<indexType>, typename Point::distanceType>>,
                    parlay::sequence<std::pair<PathElement<indexType>, typename Point::distanceType>>>, indexType>
beam_search_path(const Point p, const Graph<indexType> &G, const PointRange &Points,
            const indexType starting_point, const QueryParams &QP) {
  std::vector<indexType> start_points = {starting_point};
  return beam_search_path(p, G, Points, start_points, QP);
}



template<typename PointRange, typename indexType>
parlay::sequence<parlay::sequence<PathElement<indexType>>>
searchAll_path(PointRange& Query_Points,
          Graph<indexType> &G, PointRange &Base_Points, stats<indexType> &QueryStats,
          indexType starting_point, QueryParams &QP) {
  std::vector<indexType> start_points = {starting_point};
  return searchAll_path<PointRange, indexType>(Query_Points, G, Base_Points, QueryStats, start_points, QP);
}

template< typename PointRange, typename indexType>
parlay::sequence<parlay::sequence<PathElement<indexType>>>
searchAll_path(PointRange &Query_Points,
          Graph<indexType> &G, PointRange &Base_Points, stats<indexType> &QueryStats,
          parlay::sequence<indexType> starting_points,
          QueryParams &QP) {
  if (QP.k > QP.beamSize) {
    std::cout << "Error: beam search parameter Q = " << QP.beamSize
              << " same size or smaller than k = " << QP.k << std::endl;
    abort();
  }
  parlay::sequence<parlay::sequence<PathElement<PathElement<indexType>>>> all_neighbors(Query_Points.size());
  parlay::parallel_for(0, Query_Points.size(), [&](size_t i) {
    parlay::sequence<PathElement<indexType>> neighbors = parlay::sequence<PathElement<indexType>>(QP.k);
    auto [pairElts, dist_cmps] = beam_search_path(Query_Points[i], G, Base_Points, starting_points, QP);
    auto [beamElts, visitedElts] = pairElts;
    for (indexType j = 0; j < QP.k; j++) {
      neighbors[j] = beamElts[j].first;
    }
    all_neighbors[i] = neighbors;
    QueryStats.increment_visited(i, visitedElts.size());
    QueryStats.increment_dist(i, dist_cmps);
  });

  return all_neighbors;
}

// Returns a sequence of nearest neighbors each with their distance
template<typename Point, typename QPoint, typename QQPoint,
         typename PointRange, typename QPointRange, typename QQPointRange,
         typename indexType>
parlay::sequence<std::pair<PathElement<indexType>, typename Point::distanceType>>
beam_search_rerank_path(const Point &p,
                   const QPoint &qp,
                   const QQPoint &qqp,
                   const Graph<indexType> &G,
                   const PointRange &Base_Points,
                   const QPointRange &Q_Base_Points,
                   const QQPointRange &QQ_Base_Points,
                   stats<indexType> &QueryStats,
                   const parlay::sequence<indexType> starting_points,
                   const QueryParams &QP,
                   bool stats = true) {
  using dtype = typename Point::distanceType;
  using id_dist = std::pair<PathElement<indexType>, dtype>;
  auto QPP = QP;

  bool use_rerank = (Base_Points.params.num_bytes() != Q_Base_Points.params.num_bytes());
  bool use_filtering = (Q_Base_Points.params.num_bytes() != QQ_Base_Points.params.num_bytes());
  auto [pairElts, dist_cmps] = filtered_beam_search_path(G,
                                                    qp, Q_Base_Points,
                                                    qqp, QQ_Base_Points,
                                                    starting_points, QPP, use_filtering);
  auto [beamElts, visitedElts] = pairElts;
  if (beamElts.size() < QP.k) {
    std::cout << "Error: for point id " << p.id() << " beam search returned " << beamElts.size() << " elements, which is less than k = " << QP.k << std::endl;
    abort();
  }

  if (stats) {
    QueryStats.increment_visited(p.id(), visitedElts.size());
    QueryStats.increment_dist(p.id(), dist_cmps);
  }

  if (use_rerank) {
    // recalculate distances with non-quantized points and sort
    int num_check = std::min<int>(QP.k * QP.rerank_factor, beamElts.size());
    std::vector<id_dist> pts;
    for (int i=0; i < num_check; i++) {
      auto j = beamElts[i].first;
      pts.push_back(id_dist(j, p.distance(Base_Points[j.nodeId])));
    }
    auto less = [&] (id_dist a, id_dist b) {
      return a.second < b.second || (a.second == b.second && a.first.nodeId < b.first.nodeId);
    };
    std::sort(pts.begin(), pts.end(), less);

    // keep first k
    parlay::sequence<id_dist> results;
    for (int i= 0; i < QP.k; i++)
      results.push_back(pts[i]);

    return results;
  } else {
    //return beamElts;
    parlay::sequence<id_dist> results;
    for (int i= 0; i < QP.k; i++) {
      auto j = beamElts[i].first;
      results.push_back(id_dist(j, p.distance(Base_Points[j.nodeId])));
    }
    return results;
  }
}


template<typename indexType, typename Point, typename PR, class GT>
void clear_graph(GT &G, PR &Points, int start_point, long Q, indexType topk=10, double max_fraction = .02, bool print=true) {
    GT pathG(G.max_degree(), Points.size());
    parlay::parallel_for(0, G.size(), [&](long i){
      parlay::sequence<indexType> data = parlay::tabulate(G[i].size(),[](long j){return static_cast<indexType>(0);});
      pathG[i].update_neighbors(data);
    });
    std::cout<<"finish init path G"<<std::endl;

    size_t m = G.size();
    size_t count = 0;
    size_t max_batch_size = std::min(static_cast<size_t>(max_fraction * static_cast<float>(m)),
                                     1000000ul);
    //fix bug where max batch size could be set to zero
    if(max_batch_size == 0) max_batch_size = m;

    while (count < m) {
      size_t floor;
      size_t ceiling;

      floor = count;
      ceiling = std::min(count + static_cast<size_t>(max_batch_size), m);
      count += static_cast<size_t>(max_batch_size);


      parlay::sequence<parlay::sequence<PathElement<indexType>>> frontier_out(ceiling-floor);
      parlay::sequence<parlay::sequence<PathElement<indexType>>> visited_out(ceiling-floor);

      parlay::parallel_for(floor, ceiling, [&](size_t i) {
        indexType index = static_cast<indexType>(i);
        int sp = start_point;
        QueryParams QP((long) 0, Q, (double) 0.0, (long) Points.size(), (long) G.max_degree());


        auto [pairElts, dist_cmps] = beam_search_path<Point, PR, indexType>(Points[index], G, Points, sp, QP);
        auto [frontier, visited] = pairElts;


        visited_out[i-floor] = parlay::tabulate(visited.size(), [&](long j){return visited[j].first;});
        frontier_out[i-floor] = parlay::tabulate(frontier.size(), [&](long j){return frontier[j].first;});

      });

      for(size_t i = 0; i < frontier_out.size(); ++i){
        fill_path_graph<indexType, Point, GT>(frontier_out[i], visited_out[i], pathG);
      }

      if (print) {
        printProgressBar(count / (float)m);
      }
    }
    std::cout<<std::endl;

    parlay::parallel_for(0, G.size(), [&](long i){
      std::vector<indexType> cands;
      for(indexType j = 0; j < pathG[i].size(); ++j){
        if(pathG[i][j] > 0)
          cands.push_back(G[i][j]);
      }
      G[i].update_neighbors(cands);
    });
}


} // end namespace

#endif // ALGORITHMS_ANN_BEAM_SEARCH_H_
