// This code is part of the Local Index Distillation
// Copyright (c) 2025 Yu Yan and Hunan University
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
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "parlay/internal/get_time.h"
#include <random>
#include <set>
#include <queue>
#include <math.h>
#include "clusterNN.h"
#include "../utils/beamSearch.h"
#include "../utils/prune.h"
#include "../utils/log.h"

namespace parlayANN
{

    template <typename Point, typename PointRange, typename indexType>
    struct NN_index
    {
        using distanceType = typename Point::distanceType;
        using GraphI = Graph<indexType>;
        using PR = PointRange;
        using edge = std::pair<indexType, indexType>;
        using pid = std::pair<indexType, distanceType>;
        using ppid = std::pair<indexType, pid>;
        using labelled_edge = std::pair<indexType, pid>;

        BuildParams BP;
        Prune<PointRange, indexType> prune;

        static constexpr auto less = [](edge a, edge b)
        { return a.second < b.second; };

        NN_index(BuildParams &BP) : BP(BP), prune(BP.PR, BP.R) {}

        parlay::sequence<parlay::sequence<pid>> old_neighbors;

        void push_into_queue(std::priority_queue<edge, std::vector<edge>, decltype(less)> &Q, edge p)
        {
            if (Q.size() < 2 * BP.R)
            {
                Q.push(p);
            }
            else
            {
                indexType highest_p = Q.top().second;
                if (p.second < highest_p)
                {
                    Q.pop();
                    Q.push(p);
                }
            }
        }

        parlay::sequence<int> nn_descent(PR &Points, parlay::sequence<int> &changed)
        {
            auto new_changed = parlay::sequence<int>(Points.size(), 0);
            auto rev = reverse_graph();
            parlay::random_generator gen;
            size_t n = Points.size();
            std::uniform_int_distribution<indexType> dis(0, n - 1);
            int batch_size = 100000;
            std::pair<indexType, parlay::sequence<indexType>> *begin;
            std::pair<indexType, parlay::sequence<indexType>> *end = rev.begin();
            int counter = 0;
            while (end != rev.end())
            {
                counter++;
                begin = end;
                int remaining = rev.end() - end;
                end += std::min(remaining, batch_size);
                nn_descent_chunk(Points, changed, new_changed, begin, end);
            }
            return new_changed;
        }

        void nn_descent_chunk(PR &Points, parlay::sequence<int> &changed,
                              parlay::sequence<int> &new_changed, std::pair<indexType, parlay::sequence<indexType>> *begin,
                              std::pair<indexType, parlay::sequence<indexType>> *end)
        {
            size_t stride = end - begin;
            auto less = [&](pid a, pid b)
            { return a.second < b.second; };
            auto grouped_labelled = parlay::tabulate(stride, [&](size_t i)
                                                     {
            indexType index = (begin+i)->first;
            std::set<indexType> to_filter;
            to_filter.insert(index);
            for(indexType j=0; j<old_neighbors[index].size(); j++){
                to_filter.insert(old_neighbors[index][j].first);
            }
            auto f = [&] (indexType a) {return (to_filter.find(a) == to_filter.end());};
            auto filtered_candidates = parlay::filter((begin+i)->second, f);
	    parlay::sequence<labelled_edge> edges;
	    edges.reserve(BP.R*2);
	    for(indexType l=0; l<filtered_candidates.size(); l++){
                indexType j=filtered_candidates[l];
		distanceType j_max = old_neighbors[j][old_neighbors[j].size()-1].second;
		for(indexType m=l+1; m<filtered_candidates.size(); m++){
                    indexType k=filtered_candidates[m];
		    if (changed[j] || changed[k]) {
              distanceType dist = Points[j].distance(Points[k]);
		      distanceType k_max = old_neighbors[k][old_neighbors[k].size()-1].second;
		      if(dist < j_max) edges.push_back(std::make_pair(j, std::make_pair(k, dist)));
		      if(dist < k_max) edges.push_back(std::make_pair(k, std::make_pair(j, dist)));
		    }
		}
	    }
            for(indexType l=0; l<old_neighbors[index].size(); l++){
                indexType j = old_neighbors[index][l].first;
                for(const indexType& k : filtered_candidates){
		  if (changed[index] || changed[k]) {
                    distanceType dist = Points[j].distance(Points[k]);
                    distanceType j_max = old_neighbors[j][old_neighbors[j].size()-1].second;
                    distanceType k_max = old_neighbors[k][old_neighbors[k].size()-1].second;
                    if(dist < j_max) edges.push_back(std::make_pair(j, std::make_pair(k, dist)));
                    if(dist < k_max) edges.push_back(std::make_pair(k, std::make_pair(j, dist)));
		  }
                }
            }
			return edges; }, 1);
            auto candidates = parlay::group_by_key(parlay::flatten(grouped_labelled));
            parlay::parallel_for(0, candidates.size(), [&](size_t i)
                                 {
            auto less2 = [&] (pid a, pid b) {
                if(a.second < b.second) return true;
                else if(a.second == b.second){
                    if(a.first < b.first) return true;
                }
                return false;
            };
            parlay::sort_inplace(candidates[i].second, less2);
            indexType cur_index=std::numeric_limits<unsigned int>::max();
            parlay::sequence<pid> filtered_candidates;
            for(const pid& p : candidates[i].second){
                if(p.first!=cur_index){
                    filtered_candidates.push_back(p);
                    cur_index = p.first;
                }
            }
            indexType index = candidates[i].first;
            auto less3 = [&] (pid a, pid b) {return a.second < b.second;};
            auto [new_edges, change] = seq_union_bounded(old_neighbors[index], filtered_candidates, BP.R, less3);
            if(change){
                new_changed[index]=1;
                old_neighbors[index]=new_edges;
            } });
        }

        parlay::sequence<std::pair<indexType, parlay::sequence<indexType>>> reverse_graph()
        {
            parlay::sequence<parlay::sequence<edge>> to_group = parlay::tabulate(old_neighbors.size(), [&](size_t i)
                                                                                 {
            size_t s = old_neighbors[i].size();
            parlay::sequence<edge> e(s);
            for(indexType j=0; j<s; j++){
                e[j] = std::make_pair(old_neighbors[i][j].first, (int) i);
            }
            return e; });
            auto sorted_graph = parlay::group_by_key(parlay::flatten(to_group));
            parlay::parallel_for(0, sorted_graph.size(), [&](size_t i)
                                 {
            auto shuffled = parlay::remove_duplicates(parlay::random_shuffle(sorted_graph[i].second, i));
            indexType upper_bound = std::min((long) shuffled.size(), BP.R);
            auto truncated = parlay::tabulate(upper_bound, [&] (size_t j){
                return shuffled[j];
            });
            sorted_graph[i].second = truncated; });
            return sorted_graph;
        }

        int nn_descent_wrapper(PR &Points)
        {
            size_t n = Points.size();
            parlay::sequence<int> changed = parlay::tabulate(n, [&](size_t i)
                                                             { return 1; });
            int rounds = 0;
            //        int max_rounds = std::max(10, (int) log2(Points.dimension()));
            //        if(Points.dimension()==256) max_rounds=20; //hack for ssnpp
            int max_rounds = 64;
            while (parlay::reduce(changed) >= BP.delta * n && rounds < max_rounds)
            {
                auto new_changed = nn_descent(Points, changed);
                changed = new_changed;
                rounds++;
                std::cout << parlay::reduce(new_changed) << " elements changed" << std::endl;
                std::cout << "Round " << rounds << " of " << max_rounds << " completed" << std::endl;
            }

            std::cout << "descent converged in " << rounds << " rounds";
            if (rounds < max_rounds)
                std::cout << " (Early termination)";
            std::cout << std::endl;
            return rounds;
        }

        void undirect_and_prune(GraphI &G, PR &Points)
        {
            parlay::sequence<parlay::sequence<ppid>> to_group = parlay::tabulate(old_neighbors.size(), [&](size_t i){
                size_t s = old_neighbors[i].size();
                assert(s == BP.R);

                parlay::sequence<ppid> e(s);
                for(indexType j=0; j<s; j++){
                    e[j] = std::make_pair(old_neighbors[i][j].first, std::make_pair(static_cast<indexType>(i), old_neighbors[i][j].second));
                }
                return e; });
            auto undirected_graph = parlay::group_by_key_ordered(parlay::flatten(to_group));

            size_t offset = undirected_graph.size() * BP.delta;
            for(size_t m = 0; m < undirected_graph.size(); m+=offset){
                size_t mm = undirected_graph.size() < (m + offset) ? undirected_graph.size() : (m + offset);
                parlay::parallel_for(m, mm, [&](size_t i){
                    indexType index = undirected_graph[i].first;

                    std::vector<pid> nbs = undirected_graph[i].second.to_vector();
                    std::sort(nbs.begin(), nbs.end(), [](pid a, pid b){return a.first < b.first;});
                    auto nbs_end =std::unique(nbs.begin(), nbs.end(), [&] (pid a, pid b) {return a.first == b.first;});
                    nbs = std::vector(nbs.begin(), nbs_end);
                    auto undirected_pids = parlay::to_sequence(nbs);

                    parlay::sort_inplace(undirected_pids, less);
                    auto less3 = [&] (pid a, pid b) {return a.second < b.second;};
                    auto [merged_pids, changed] = seq_union_bounded(old_neighbors[index], undirected_pids, BP.L, less3);
                    old_neighbors[index] = merged_pids; });
                printProgressBar(mm / (float)undirected_graph.size());
            }
            std::cout<<std::endl;

            // fill new graph ready to beam search
            parlay::parallel_for(0, old_neighbors.size(), [&](size_t index){
                parlay::sequence<indexType> out;
                long rp_distance_comps;
                std::tie(out, rp_distance_comps) = prune.robustPrune(index, old_neighbors[index], G, Points, BP.alpha, BP.cos_angle);
                //parlay::sequence<indexType> out = parlay::tabulate(BP.R, [&](long j){return old_neighbors[index][j].first;});
                G[index].update_neighbors(out);});

        }

        size_t build_index(GraphI &G, PR &Points, long cluster_size, long num_clusters)
        {
            clusterPID<Point, PointRange, indexType> C;
            old_neighbors = parlay::sequence<parlay::sequence<pid>>(G.size());
            C.multiple_clustertrees(Points, cluster_size, num_clusters, BP.R, old_neighbors);
            nn_descent_wrapper(Points);
            undirect_and_prune(G, Points);
            return 0;
        }

    };

} // end namespace
