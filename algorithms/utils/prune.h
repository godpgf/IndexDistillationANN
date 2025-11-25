// This code is part of the pruning strategy
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
#pragma once

#include <math.h>

#include <algorithm>
#include <random>
#include <set>
#include <cstdlib>  // 包含 rand() 和 srand()
#include <ctime>    // 包含 time()

#include "../utils/point_range.h"
#include "../utils/graph.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/delayed.h"
#include "parlay/random.h"
#include "hierarchical_idmapping.h"

namespace parlayANN
{

    template <typename PointRange, typename indexType>
    struct Prune
    {
        using Point = typename PointRange::Point;

        using distanceType = typename Point::distanceType;
        using pid = std::pair<indexType, distanceType>;
        //using PR = PointRange;
        using GraphI = Graph<indexType>;

        uint PR, R;

        Prune(uint PR, uint R) : PR(PR), R(R) {}

        std::pair<parlay::sequence<pid>, long>
        robustPrune(indexType p, std::vector<pid> &candidates,
                    PointRange &Points, double alpha, double cos_angle, HIdMapping* hid_mapping=nullptr){
            auto f_map = [&](size_t node_id){
                return (hid_mapping == nullptr) ? node_id : hid_mapping->get_root_id(node_id);
            };
            long distance_comps = 0;
            

            // Sort the candidate set according to distance from p
            auto less = [&](std::pair<indexType, distanceType> a, std::pair<indexType, distanceType> b)
            {
                return a.second < b.second || (a.second == b.second && a.first < b.first);
            };
            std::sort(candidates.begin(), candidates.end(), less);


            // remove any duplicates
            auto new_end = std::unique(candidates.begin(), candidates.end(),
                                       [&](auto x, auto y)
                                       { return x.first == y.first; });
            candidates = std::vector(candidates.begin(), new_end);

            std::vector<pid> new_nbhs;
            new_nbhs.reserve(PR);

            size_t candidate_idx = 0;

            while (new_nbhs.size() < PR && candidate_idx < candidates.size())
            {
                // Don't need to do modifications.
                int p_star = candidates[candidate_idx].first;
                candidate_idx++;
                if (p_star == p || p_star == -1)
                {
                    continue;
                }

                new_nbhs.push_back(candidates[candidate_idx - 1]);

                for (size_t i = candidate_idx; i < candidates.size(); i++)
                {
                    int p_prime = candidates[i].first;
                    if (p_prime != -1)
                    {
                        distance_comps++;
                        distanceType dist_starprime = Points[f_map(p_star)].distance(Points[f_map(p_prime)]);
                        distanceType dist_pprime = candidates[i].second;
                        if (alpha > 1e-5){    
                            if (alpha * dist_starprime <= dist_pprime)
                            {
                                candidates[i].first = -1;
                            }
                        } else if(cos_angle > 1e-5){
                            float a2 = dist_starprime;
                            float b2 = dist_pprime;
                            float c2 = candidates[candidate_idx - 1].second;
                            float b = std::sqrt(b2);
                            float c = std::sqrt(c2);
                            float cur_cos = (b2 + c2 - a2) / (2 * b * c);
                            if (cur_cos > cos_angle)
                            {
                                // 夹角越小cos越大，需要把夹角小的（cos大的）删掉
                                candidates[i].first = -1;
                            }
                        }
                    }
                }
            }
            
            
            if(PR > R && new_nbhs.size() > R) {    
                candidates.clear();
                candidates.reserve(new_nbhs.size());
                std::vector<distanceType> score(new_nbhs.size(), 0);
                for(auto j : new_nbhs)
                    candidates.push_back(j);
                new_nbhs.clear();

                int min_score_id = 0;
                while(new_nbhs.size() < R && min_score_id >= 0){
                    score[min_score_id] = -1;
                    new_nbhs.push_back(candidates[min_score_id]);
                    min_score_id = -1;
                    int p_star = new_nbhs[new_nbhs.size() - 1].first;
                    distanceType dist_star = new_nbhs[new_nbhs.size() - 1].second;

                    for(int i = 1; i < candidates.size(); ++i){
                        if(score[i] < -1e-5)
                            continue;
                        int p_prime = candidates[i].first;
                        distance_comps++;
                        distanceType dist_starprime = Points[f_map(p_star)].distance(Points[f_map(p_prime)]);
                        distanceType dist_pprime = candidates[i].second;                        
                        float a2 = dist_starprime;
                        float b2 = dist_pprime;
                        float c2 = dist_star;
                        float b = std::sqrt(b2);
                        float c = std::sqrt(c2);
                        float cur_cos = (b2 + c2 - a2) / (2 * b * c);   
                        score[i] = std::max<distanceType>(score[i], (1.0001f + cur_cos) * b);       
                        if(min_score_id < 0 || score[i] < score[min_score_id]){
                            min_score_id = i;
                        }       
                    }
                }

            }

            auto new_neighbors_seq = parlay::tabulate(new_nbhs.size(), [&](long i)
                                                      { return new_nbhs[i]; });
            return std::pair(new_neighbors_seq, distance_comps);
        }

        // robustPrune routine as found in DiskANN paper, with the exception
        // that the new candidate set is added to the field new_nbhs instead
        // of directly replacing the out_nbh of p
        std::pair<parlay::sequence<indexType>, long>
        robustPrune(indexType p, parlay::sequence<pid> &cand,
                    GraphI &G, PointRange &Points, double alpha, double cos_angle, bool add = true, HIdMapping* hid_mapping=nullptr)
        {
            auto f_map = [&](size_t node_id){
                return (hid_mapping == nullptr) ? node_id : hid_mapping->get_root_id(node_id);
            };
            // add out neighbors of p to the candidate set.
            std::vector<pid> candidates;
            long distance_comps = 0;
            for (auto x : cand)
                candidates.push_back(x);

            if (add)
            {
                size_t out_size = G[p].size();
                for (size_t i = 0; i < out_size; i++)
                {
                    distance_comps++;
                    candidates.push_back(std::make_pair(G[p][i], Points[f_map(G[p][i])].distance(Points[f_map(p)])));
                }
            }

            auto [ngh_seq, dc] = robustPrune(p, candidates, Points, alpha, cos_angle, hid_mapping);
            distance_comps += dc;
            auto new_neighbors_seq = parlay::tabulate(ngh_seq.size(), [&](long i)
                                                      { return ngh_seq[i].first; });
            return std::pair(new_neighbors_seq, distance_comps);
        }

        std::pair<parlay::sequence<indexType>, long>
        robustPrune(indexType p,  GraphI &G, PointRange &Points, double alpha, double cos_angle, HIdMapping* hid_mapping=nullptr)
        {
            auto f_map = [&](size_t node_id){
                return (hid_mapping == nullptr) ? node_id : hid_mapping->get_root_id(node_id);
            };
            // add out neighbors of p to the candidate set.
            size_t out_size = G[p].size();
            std::vector<pid> candidates;
            long distance_comps = 0;

            for (size_t i = 0; i < out_size; i++)
            {
                distance_comps++;
                candidates.push_back(std::make_pair(G[p][i], Points[f_map(G[p][i])].distance(Points[f_map(p)])));
            }
            

            auto [ngh_seq, dc] = robustPrune(p, candidates, Points, alpha, cos_angle, hid_mapping);
            distance_comps += dc;
            auto new_neighbors_seq = parlay::tabulate(ngh_seq.size(), [&](long i)
                                                      { return ngh_seq[i].first; });
            return std::pair(new_neighbors_seq, distance_comps);
        }

        // wrapper to allow calling robustPrune on a sequence of candidates
        // that do not come with precomputed distances
        std::pair<parlay::sequence<indexType>, long>
        robustPrune(indexType p, parlay::sequence<indexType> candidates,
                    GraphI &G, PointRange &Points, double alpha, double cos_angle, bool add = true, HIdMapping* hid_mapping=nullptr)
        {
            auto f_map = [&](size_t node_id){
                return (hid_mapping == nullptr) ? node_id : hid_mapping->get_root_id(node_id);
            };
            parlay::sequence<pid> cc;
            long distance_comps = 0;
            cc.reserve(candidates.size()); // + size_of(p->out_nbh));
            for (size_t i = 0; i < candidates.size(); ++i)
            {
                distance_comps++;
                cc.push_back(std::make_pair(candidates[i], Points[f_map(candidates[i])].distance(Points[f_map(p)])));
            }
            auto [ngh_seq, dc] = robustPrune(p, cc, G, Points, alpha, cos_angle, add, hid_mapping);
            return std::pair(ngh_seq, dc + distance_comps);
        }

    };

}
