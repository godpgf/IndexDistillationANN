#pragma once
#include <limits>
#include <algorithm>
#include <queue>
#include "graph.h"
#include "hierarchical_idmapping.h"

namespace parlayANN
{
    struct DFS{
        template<typename indexType, typename rangeType>
        static size_t repair_isolate_node(Graph<indexType>& G, rangeType& enter_node_ids, size_t node_num, size_t batch_size=1000, HIdMapping* hid_mapping=nullptr){
            auto flag = parlay::tabulate(G.size(), [&](size_t i){return 0;});         
            size_t cnt = 0;
            std::queue<size_t> next_visited;

            for(auto eid : enter_node_ids){
                flag[eid] = 1;
                next_visited.push(eid);
            }


            std::vector<size_t> cur_visited = std::vector<size_t>(batch_size);
            size_t cur_visited_size = 0;

            while(!next_visited.empty()){
                cur_visited_size = 0;
                while(cur_visited_size < batch_size && !next_visited.empty()){
                    auto cur_node_id = next_visited.front();
                    cur_visited[cur_visited_size++] = cur_node_id;
                    next_visited.pop();
                }

                for(size_t i = 0; i < cur_visited_size; ++i){
                    auto cur_node_id = cur_visited[i];
                    if(hid_mapping != nullptr){
                        auto eid = hid_mapping->get_id_in_pre_level(cur_node_id);
                        if(eid >= 0 && flag[eid] == 0){
                            flag[eid] = 1;
                            next_visited.push(eid);                           
                        }
                    }

                    auto edges = G[cur_node_id];
                    for(size_t j = 0; j < edges.size(); ++j){
                        size_t nid = static_cast<size_t>(edges[j]);
                        if(flag[nid] == 0){
                            flag[nid] = 1;
                            next_visited.push(nid);
                        }
                    }
                }
            }

            for(size_t i = 0; i < node_num; ++i){
                assert(i < G.size());
                if(flag[i] == 0){
                    if(hid_mapping != nullptr && hid_mapping->get_level(i) > 0)
                        continue;
                    cnt++;
                    auto edges = G[i];
                    for(size_t j = 0; j < edges.size(); ++j){
                        size_t nid = static_cast<size_t>(edges[j]);
                        auto n_edges = G[nid];
                        if(n_edges.size() < G.max_degree()){
                            n_edges.append_neighbor(i);
                            break;
                        }
                    }                    
                }
            }
            return cnt;
        }

        template<typename indexType>
        static size_t repair_isolate_node(Graph<indexType>& G, size_t sp, size_t batch_size=1000, HIdMapping* hid_mapping=nullptr){
            std::vector<size_t> enter_node_ids = {sp};
            return repair_isolate_node(G, enter_node_ids, hid_mapping == nullptr ? G.size() : hid_mapping->cur_node_num, batch_size, hid_mapping);
        }
    };
}

