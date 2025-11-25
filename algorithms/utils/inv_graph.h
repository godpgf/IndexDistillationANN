#ifndef INV_GRAPH_H_
#define INV_GRAPH_H_

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

template<typename indexType, class GT>
void inv_graph(GT& G){
    GT GL(G.max_degree() >> 1, G.size());
    GT GR(G.max_degree() >> 1, G.size());
    parlay::parallel_for(0, G.size(), [&](long i){
        for(size_t j = 0; j < G[i].size(); ++j){
            if(j % 2 == 0){
                GL[i].append_neighbor(G[i][j]);
            } else {
                GR[i].append_neighbor(G[i][j]);
            }
            
        }
        G[i].clear_neighbors();
    });

    auto flattened = parlay::delayed::flatten(parlay::tabulate(GR.size(), [&](size_t i) {
        indexType index = static_cast<indexType>(i);
        return parlay::delayed::map(GR[i], [=] (indexType ngh) {return std::pair(ngh, index);});}));
    auto grouped_by = parlay::group_by_key(parlay::delayed::to_sequence(flattened));
    parlay::parallel_for(0, grouped_by.size(), [&](size_t i) {
        auto &[index, candidates] = grouped_by[i];
        G[index].update_neighbors(candidates);
    });
    parlay::parallel_for(0, G.size(), [&](size_t i) {
        for(size_t j = 0; j < GL[i].size(); ++j){
            G[i].append_neighbor(GL[i][j]);
        }
    });
}

}

#endif