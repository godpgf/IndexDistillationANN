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
#include "../utils/types.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/delayed.h"
#include "parlay/random.h"

namespace parlayANN
{

    template <typename PointRange, typename indexType>
    struct Prune
    {
        using Point = typename PointRange::Point;

        using distanceType = typename Point::distanceType;
        using pid = std::pair<indexType, distanceType>;
        using PR = PointRange;
        using GraphI = Graph<indexType>;

        BuildParams BP;

        Prune(BuildParams &BP) : BP(BP) {}

        std::pair<parlay::sequence<pid>, long>
        robustPrune(indexType p, std::vector<pid> &candidates,
                    PR &Points, double alpha, double angle){
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
            new_nbhs.reserve(BP.PR);

            size_t candidate_idx = 0;

            if (alpha > 1e-5)
            {
                while (new_nbhs.size() < BP.PR && candidate_idx < candidates.size())
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
                            distanceType dist_starprime = Points[p_star].distance(Points[p_prime]);
                            distanceType dist_pprime = candidates[i].second;
                            if (alpha * dist_starprime <= dist_pprime)
                            {
                                candidates[i].first = -1;
                            }
                        }
                    }
                }
            }
            else if(angle > 1e-5)
            {
                for (const pid &j : candidates)
                {
                    if (new_nbhs.size() == BP.PR)
                        break;
                    else if (new_nbhs.size() == 0)
                        new_nbhs.push_back(j);
                    else
                    {
                        distanceType dist_p = j.second;
                        bool add = true;
                        for (const pid &k : new_nbhs)
                        {
                            distanceType a2 = Points[j.first].distance(Points[k.first]);
                            distanceType b2 = j.second;
                            distanceType c2 = k.second;
                            distanceType b = std::sqrt(b2);
                            distanceType c = std::sqrt(c2);
                            if ((b2 + c2 - a2) / (2 * b * c) > angle)
                            {
                                add = false;
                                break;
                            }
                        }
                        if (add)
                            new_nbhs.push_back(j);
                    }
                }
            } 
            
            
            
            if(BP.PR > BP.R && new_nbhs.size() > BP.R) {
                candidates.clear();
                candidates.reserve(new_nbhs.size());
                for(auto j : new_nbhs)
                    candidates.push_back(j);
                new_nbhs.clear();

                using pidd = std::pair<pid, distanceType>;
                std::vector<pidd> frontier;
                frontier.reserve(candidates.size());
                std::vector<pidd> new_frontier;
                new_frontier.reserve(candidates.size());

                auto less = [&](pidd a, pidd b)
                {
                    if(a.second < b.second )
                        return true;
                    if(a.second > b.second )
                        return false;   
                    if(a.first.second < b.first.second )
                        return true;
                    if(a.first.second > b.first.second )
                        return false;   
                    return a.first.first < b.first.first;
                };

                // initialize
                for (const pid &j : candidates){
                    frontier.push_back(std::make_pair(j, 0.0f));
                }

                // 预留一条边用来挂接孤立节点
                while (new_nbhs.size() < (BP.R-1) && frontier.size() > 0)
                {
                    const pid &j = frontier[0].first;
                    new_nbhs.push_back(j);
                    // update score
                    for(auto i = 1; i < frontier.size(); ++i){
                        const pid &k = frontier[i].first;
                        distance_comps++;
                        distanceType a2 = Points[j.first].distance(Points[k.first]);
                        distanceType b2 = j.second;
                        distanceType c2 = k.second;
                        distanceType b = std::sqrt(b2);
                        distanceType c = std::sqrt(c2);
                        distanceType cos =  (b2 + c2 - a2) / (2 * b * c);
                        distanceType score = std::max<distanceType>((1.0001f + cos) * c, frontier[i].second);
                        new_frontier.push_back(std::make_pair(k, score));
                    }
                    frontier.clear();

                    for(auto cand : new_frontier){
                        frontier.push_back(cand);
                    }
                    new_frontier.clear();
                    std::sort(frontier.begin(), frontier.end(), less);
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
                    GraphI &G, PR &Points, double alpha, double angle, bool add = true)
        {
            // add out neighbors of p to the candidate set.
            size_t out_size = G[p].size();
            std::vector<pid> candidates;
            long distance_comps = 0;
            for (auto x : cand)
                candidates.push_back(x);

            if (add)
            {
                for (size_t i = 0; i < out_size; i++)
                {
                    distance_comps++;
                    candidates.push_back(std::make_pair(G[p][i], Points[G[p][i]].distance(Points[p])));
                }
            }

            auto [ngh_seq, dc] = robustPrune(p, candidates, Points, alpha, angle);
            distance_comps += dc;
            auto new_neighbors_seq = parlay::tabulate(ngh_seq.size(), [&](long i)
                                                      { return ngh_seq[i].first; });
            return std::pair(new_neighbors_seq, distance_comps);
        }

        std::pair<parlay::sequence<indexType>, long>
        robustPrune(indexType p,  GraphI &G, PR &Points, double alpha, double angle)
        {
            // add out neighbors of p to the candidate set.
            size_t out_size = G[p].size();
            std::vector<pid> candidates;
            long distance_comps = 0;

            for (size_t i = 0; i < out_size; i++)
            {
                distance_comps++;
                candidates.push_back(std::make_pair(G[p][i], Points[G[p][i]].distance(Points[p])));
            }
            

            auto [ngh_seq, dc] = robustPrune(p, candidates, Points, alpha, angle);
            distance_comps += dc;
            auto new_neighbors_seq = parlay::tabulate(ngh_seq.size(), [&](long i)
                                                      { return ngh_seq[i].first; });
            return std::pair(new_neighbors_seq, distance_comps);
        }

        // wrapper to allow calling robustPrune on a sequence of candidates
        // that do not come with precomputed distances
        std::pair<parlay::sequence<indexType>, long>
        robustPrune(indexType p, parlay::sequence<indexType> candidates,
                    GraphI &G, PR &Points, double alpha, double angle, bool add = true)
        {

            parlay::sequence<pid> cc;
            long distance_comps = 0;
            cc.reserve(candidates.size()); // + size_of(p->out_nbh));
            for (size_t i = 0; i < candidates.size(); ++i)
            {
                distance_comps++;
                cc.push_back(std::make_pair(candidates[i], Points[candidates[i]].distance(Points[p])));
            }
            auto [ngh_seq, dc] = robustPrune(p, cc, G, Points, alpha, angle, add);
            return std::pair(ngh_seq, dc + distance_comps);
        }

    };

}
