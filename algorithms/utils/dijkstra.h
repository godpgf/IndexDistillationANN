// This code is part of the Dijkstra algorithm
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

#ifndef DJK
#define DJK
#include <iomanip>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/delayed.h"
#include "graph.h"
#include "avl.h"

void printProgressBar(double progress, int width = 50) {
    std::cout << "[";
    int pos = width * progress;
    for (int i = 0; i < width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(2) << progress * 100.0 << " %\r";
    std::cout.flush();
}

namespace parlayANN {

template<typename PointRange, typename indexType>
struct Dijkstra{
    Dijkstra(size_t g_size):
        parent(parlay::sequence<indexType>(g_size, static_cast<indexType>(0))),
        final(parlay::sequence<bool>(g_size, false)),
        dist(parlay::sequence<float>(g_size, std::numeric_limits<float>::max())),
        avl_ids(parlay::sequence<indexType>(g_size, static_cast<indexType>(0))),
        avl(g_size){

    }

    void clear(indexType all_size){
        parlay::parallel_for(0, parent.size(), [&](size_t i) {
            // 记录节点的父亲
            parent[i] = static_cast<indexType>(i);
            // 标记某个节点是否已经在树中
            final[i] = false;
            // 标记起点到某个节点的距离
            dist[i] = std::numeric_limits<float>::max();
            // 标记某个节点在平衡二叉树中的id
            avl_ids[i] = all_size;
        });
        avl.clear();
    }

    void createShortestPathTree(Graph<indexType> &G, parlay::sequence<indexType> &inserts, PointRange &Points, indexType start_point)
    {
        // Dijkstra
        indexType all_size = inserts.size();
        clear(all_size);
        // cal avg dis
        double max_dis = Points.dimension();
        /*{
            auto all_dis = parlay::sequence<double>(parlay::num_workers(), 0.0);
            parlay::parallel_for(0, all_size, [&](size_t i) {
                double dis = 0;
                indexType index = inserts[i];
                for(int j = 0; j < G[index].size(); ++j){
                    dis += Points[index].distance(Points[G[index][j]]);
                }
                dis /= G[index].size();

                auto wid = parlay::worker_id();
                if(all_dis[wid] < dis)
                    all_dis[wid] = dis;
            });
            for(int i = 0; i < all_dis.size(); ++i)
                if(all_dis[i] > max_dis)
                    max_dis = all_dis[i];
        }*/

        auto dist_comp = [&](const std::pair<indexType, float>& a, const std::pair<indexType, float>& b)
        {
            if(a.second < b.second){
                return static_cast<int>(-1);
            }
            if(a.second > b.second){
                return static_cast<int>(1);
            }
            if (a.first < b.first)
            {
                return static_cast<int>(-1);
            }
            if (a.first > b.first)
            {
                return static_cast<int>(1);
            }
            return static_cast<int>(0);
        };

        // first insert
        dist[start_point] = 0;
        avl_ids[start_point] = avl.insert(std::make_pair(static_cast<indexType>(start_point), 0.0f), dist_comp);

        for (auto i = 1; i < all_size; ++i){
            if(avl.size() == 0)
                break;
            // get min dist
            auto min_avl_id = avl.get_min();
            auto u = avl.get_value(min_avl_id).first;

            final[u] = true;
            avl.erase(min_avl_id, dist_comp);
            // std::cout<<"finish erase"<<std::endl;

            //float scale = beta / max_dis;
            parlay::sequence<float> cur_dis = parlay::tabulate(G[u].size(), [&](size_t j){
                indexType v = G[u][j];
                if(final[v])
                    return std::numeric_limits<float>::max();
                //if(is_repair)
                //    return (float)(std::sqrt(Points[u].distance(Points[v]) + 1e-5) / max_dis);
                //float dis_uv = Points[u].distance(Points[v]) * scale + 1;
                float dis_uv = Points[u].distance(Points[v]) / max_dis;
                return dis_uv;
            });

            for(size_t j = 0; j < G[u].size(); ++j){
                indexType v = G[u][j];
                if(final[v])
                    continue;
                // std::cout<<v<<std::endl;
                float dis_uv = cur_dis[j];
                if (dist[u] + dis_uv < dist[v])
                {
                    // std::cout<<"pre insert "<<v<<std::endl;
                    if(avl_ids[v] < all_size){
                        // has insert into avl
                        // std::cout<<"del "<<avl_ids[v]<<std::endl;
                        avl.erase(avl_ids[v], dist_comp);
                    }
                    dist[v] = dist[u] + dis_uv;
                    parent[v] = u;
                    // std::cout<<"insert "<<v<<std::endl;
                    if(v >= G.size()){
                        std::cout<<v<<std::endl;
                        throw std::invalid_argument("v error!");
                    }
                    if(dist[v] > 1e10){
                        std::cout<<"error dist["<<v<<"]="<<dist[v]<<std::endl;
                        throw std::invalid_argument("v error!");
                    }
                    avl_ids[v] = avl.insert(std::make_pair(static_cast<indexType>(v), dist[v]), dist_comp);
                    if(avl_ids[v] >= all_size){
                        std::cout<<"insert error!\n";
                    }

                }
            }
            printProgressBar((i+1) / (double)inserts.size());
        }

        std::cout<<std::endl;
    }

    std::pair<size_t, double> repair(Graph<indexType> &G, parlay::sequence<indexType> &inserts, PointRange &Points, indexType start_point){
        createShortestPathTree(G, inserts, Points, start_point);
	std::cout<<"finish createShortestPathTree"<<std::endl;
        indexType all_size = inserts.size();
        double sum_dis = calSumDistance(inserts);
	std::cout<<"sum_dis:"<<sum_dis<<std::endl;

        parlay::sequence<indexType> filter_inserts = parlay::filter(inserts, [&](indexType i){return (parent[i] == i) && (i > 0);});
        std::cout<<"filter_inserts size:"<<filter_inserts.size()<<std::endl;
        parlay::sequence<std::pair<indexType, indexType>> nearst = parlay::tabulate(filter_inserts.size(), [&](long i){
          indexType index = filter_inserts[i];
          indexType n = index;
          for(int j = 0; j < G[index].size(); j++){
            indexType nn = G[index][j];
            if(G[nn].size() < G.max_degree()){
              n = nn;
              break;
            }
          }
          return std::pair(static_cast<indexType>(n), static_cast<indexType>(index));
        });
        std::cout<<"nearst size:"<<nearst.size()<<std::endl;
        auto grouped_by = parlay::group_by_key(nearst);
        std::cout<<"No. of isolated node:"<<filter_inserts.size()<<" / "<< all_size <<" repair cnt:"<<grouped_by.size()<<" avg dis:"<<sum_dis / (all_size - filter_inserts.size())<<std::endl;
        parlay::parallel_for(0, grouped_by.size(), [&](size_t i) {
            auto &[index, candidates] = grouped_by[i];
            for(auto c : candidates){
              if(G[index].size() == G.max_degree())
                break;
              if(c != index){
                G[index].append_neighbor(c);
              }
            }
        });
        return std::make_pair((size_t)filter_inserts.size(), sum_dis / (all_size - filter_inserts.size()));
    }

    std::pair<size_t, double> repair(Graph<indexType> &G, PointRange &Points, indexType start_point){
        parlay::sequence<indexType> inserts = parlay::tabulate(G.size(), [&] (size_t i){
            return static_cast<indexType>(i);});
        return repair(G, inserts, Points, start_point);
    }

     void removeRedundantBranches(Graph<indexType> &G, parlay::sequence<indexType> &inserts, parlay::sequence<size_t>& hiddenSize){
        //using edge = std::pair<indexType, indexType>;

        for (size_t i = 0; i < inserts.size(); ++i){
            indexType idx = inserts[i];
            if(parent[idx] == idx)
                continue;
            bool inEdge = false;
            for(size_t j = 0; j < G[parent[idx]].size(); ++j)
                if(G[parent[idx]][j] == idx)
                    inEdge = true;
            assert(inEdge);
            G[parent[idx]].clear_neighbors();
        }

         for (size_t i = 0; i < inserts.size(); ++i){
            indexType idx = inserts[i];
            if(parent[idx] == idx)
                continue;
            G[parent[idx]].append_neighbor(idx, hiddenSize[parent[idx]]);
        }
        // 占用内存太大，不并行执行
        /*parlay::sequence<indexType> filter_inserts = parlay::filter(inserts, [&](indexType i){
            return parent[i] != i;
        });

        parlay::sequence<edge> to_group = parlay::tabulate(filter_inserts.size(), [&](size_t i){
            indexType idx = filter_inserts[i];
            return std::make_pair(parent[idx], static_cast<indexType>(idx));
        });
        auto undirected_graph = parlay::group_by_key_ordered(to_group);
        parlay::parallel_for(0, undirected_graph.size(), [&](size_t i){
            indexType index = undirected_graph[i].first;
            G[index].update_neighbors(undirected_graph[i].second); });*/
    }

    void removeRedundantBranchesAndLeaf(Graph<indexType> &G, parlay::sequence<indexType> &inserts, parlay::sequence<size_t>& hiddenSize){
        for (size_t i = 0; i < inserts.size(); ++i){
            indexType idx = inserts[i];
            G[idx].clear_neighbors();
        }

         for (size_t i = 0; i < inserts.size(); ++i){
            indexType idx = inserts[i];
            if(parent[idx] == idx)
                continue;
            G[parent[idx]].append_neighbor(idx, hiddenSize[parent[idx]]);
        }
        /*using edge = std::pair<indexType, indexType>;
        parlay::sequence<indexType> filter_inserts = parlay::filter(inserts, [&](indexType i){
            return parent[i] != i;
        });

        parlay::sequence<edge> to_group = parlay::tabulate(filter_inserts.size(), [&](size_t i){
            indexType idx = filter_inserts[i];
            return std::make_pair(parent[idx], static_cast<indexType>(idx));
        });

        auto undirected_graph = parlay::group_by_key_ordered(to_group);

        parlay::parallel_for(0, undirected_graph.size(), [&](size_t i){
            indexType index = undirected_graph[i].first;
            G[index].update_neighbors(undirected_graph[i].second); });*/
    }

    void removeLeaf(Graph<indexType> &G, const parlay::sequence<indexType> &child_num){
        parlay::parallel_for(0, G.size(), [&](size_t i){
            if(child_num[i] == 0)
                G[i].clear_neighbors();
        });
    }

    void removeLeaf(Graph<indexType> &G, const parlay::sequence<indexType> &child_num, const parlay::sequence<indexType> &has_insert){
        parlay::parallel_for(0, G.size(), [&](size_t i){
            if(child_num[i] == 0 && has_insert[i] == 0)
                G[i].clear_neighbors();
        });
    }

    void statMonotonicityOfRetrievalPath(PointRange &Points,indexType start_point){
        int path_num = 0;
        int error_path_num = 0;
        for (int i = 1; i < parent.size(); ++i)
        {
            if(parent[i] == i)
                continue;
            path_num ++;
            if(Points[start_point].distance(Points[parent[i] ]) > Points[start_point].distance(Points[i]))
                error_path_num++;

        }
        std::cout<<"Monotonicity:"<<1.0f - error_path_num / (float)path_num<<std::endl;
    }

    size_t calIsolatedNodeNum(parlay::sequence<indexType> &inserts){
        // Isolated 
        parlay::sequence<indexType> filter_inserts = parlay::filter(inserts, [&](indexType i){return (parent[i] == i) && (i > 0);});
        return filter_inserts.size();
    }

    double calSumDistance(parlay::sequence<indexType> &inserts){
        auto all_size = inserts.size();
        double sum_dis = 0;
        {
            auto all_dis = parlay::sequence<double>(parlay::num_workers(), 0.0);
            parlay::parallel_for(0, all_size, [&](size_t i) {
                double dis = 0;
                if((parent[i] != i) || (i == 0))
                    dis = dist[i];
                if(dis > 1e8f){
                    std::cout<<"parent["<<i<<"]="<<parent[i]<<" error dist["<<i<<"]="<<dis<<std::endl;
                    std::cout<<"max float :"<<std::numeric_limits<float>::max()<<std::endl;
                    abort();
                }
                auto wid = parlay::worker_id();
                all_dis[wid] = all_dis[wid] + dis;
            });
            for(int i = 0; i < all_dis.size(); ++i)
                sum_dis += all_dis[i];
        }
        return sum_dis;
    }

    indexType calChildNumAndDepth(parlay::sequence<indexType> &inserts){
        parlay::sequence<indexType> depth = parlay::tabulate(inserts.size(), [&](size_t j){return static_cast<indexType>(0);});
        parlay::sequence<indexType> child_num = parlay::tabulate(parent.size(), [&](size_t j){return static_cast<indexType>(0);});
        indexType error_cnt = 0;
        for (int i = 0; i < inserts.size(); ++i)
        {
            indexType cur_id = inserts[i];
            if(cur_id == 0)
                continue;
            while(parent[cur_id] != cur_id){
                depth[i] = depth[i] + 1;
                bool update_child_num = (cur_id == inserts[i]);
                cur_id = parent[cur_id];
                if(update_child_num)
                    child_num[cur_id] = child_num[cur_id] + 1;
            }

            if (parent[inserts[i]] == inserts[i])
                error_cnt++;
        }
        std::cout << "error cnt=" << error_cnt << std::endl;

        size_t leaf_sum_depth = 0;
        size_t leaf_cnt = 0;
        size_t branch_child_num = 0;
        size_t branch_cnt = 0;
        size_t all_sum_depth = 0;
        for(int i = 0; i < inserts.size(); ++i){
            all_sum_depth += depth[i];
            if(child_num[inserts[i]] == 0){
                leaf_sum_depth += depth[i];
                leaf_cnt++;
            } else {
                branch_child_num += child_num[inserts[i]];
                branch_cnt++;
            }
        }
        std::cout<<"leaf num:"<<leaf_cnt<<" avg leaf depth:"<<leaf_sum_depth/(float)leaf_cnt<<" avg depth:"<<all_sum_depth / (float)parent.size()<<" branch num:"<<branch_cnt<<" avg child:"<<branch_child_num/(float)branch_cnt<<std::endl;
        return error_cnt;
    }

    void save_tree(char* oFile){

        std::ofstream writer;
        writer.open(oFile, std::ios::binary | std::ios::out);
        indexType all_num = static_cast<indexType>(parent.size());
        writer.write((char*)(&all_num), sizeof(indexType));
        indexType dim = 1;
        writer.write((char*)(&dim), sizeof(indexType));
        writer.write((char*)parent.data(), sizeof(indexType) * all_num);
        writer.close();
    }

    parlay::sequence<indexType> parent;
    parlay::sequence<bool> final;
    parlay::sequence<float> dist;
    parlay::sequence<indexType> avl_ids;
    AVL<indexType, std::pair<indexType, float>> avl;
};


/*
template<typename distanceType>
distanceType sum_sequence(parlay::sequence<distanceType>& all_data){
    parlay::sequence<distanceType> t_dis = parlay::tabulate(parlay::num_workers(), [](long i){return static_cast<distanceType>(0);});
    parlay::parallel_for(0, all_data.size(), [&](long i){
        auto wid = parlay::worker_id();
        t_dis[wid] = t_dis[wid] + all_data[i];
    });
    distanceType sum_dis = 0;
    for(int i = 0; i < t_dis.size(); ++i)
        sum_dis += t_dis[i]; 
    return sum_dis;
}


template<typename distanceType>
distanceType max_sequence(parlay::sequence<distanceType>& all_data){
    parlay::sequence<distanceType> t_dis = parlay::tabulate(parlay::num_workers(), [](long i){return static_cast<distanceType>(0);});
    parlay::parallel_for(0, all_data.size(), [&](long i){
        auto wid = parlay::worker_id();
        if(all_data[i] > t_dis[wid])
            t_dis[wid] = all_data[i];
    });
    distanceType max_dis = 0;
    for(int i = 0; i < t_dis.size(); ++i)
        if(t_dis[i] > max_dis)
            max_dis = t_dis[i]; 
    return max_dis;
}*/


}

#endif
