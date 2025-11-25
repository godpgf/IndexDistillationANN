#include <iostream>
#include <stdlib.h>
#include <vector>
#include <chrono>
#include <map>
#include <algorithm>   // std::sort, std::unique
#include "parlay/parallel.h"
#include "common.h"
#include "utils/euclidian_point.h"
#include "utils/pq_point.h"
#include "utils/point_range.h"
#include "utils/dictForest.h"
#include "utils/pivot.h"
using namespace parlayANN;

using pid = std::pair<uint, float>;

void exportDict(PointRange<PQ_Point<uint8_t>>& quant_Points, PointRange<Euclidian_Point<float>>& quant_loss, Pivot& pivot, DictForest<pid>& dict, const size_t max_cache){
    auto times = pivot.times;
    auto dim = pivot.dim;
    auto chunk = pivot.chunk;
    auto& pivots = pivot.pivots;
    auto& order_ids = pivot.order_ids;
    auto td = times * dim;
    auto tc = times * chunk;
    auto chunk_dim = dim / chunk;
    auto num_workers = parlay::num_workers();
    
    auto all_maps = std::vector<std::map<std::string, pid>>(num_workers);
    auto filter_ids = std::vector<std::vector<uint>>(num_workers);

    for(size_t t = 0; t < times; ++t){
        // 清空上一轮的数据
        for(auto& cur_map : all_maps)
            cur_map.clear();
        for(size_t i = 0; i < quant_Points.size(); i += max_cache){
            size_t cur_batch_size = (i + max_cache > quant_Points.size()) ? quant_Points.size() - i : max_cache;
            quant_Points.load2cache(i, i + cur_batch_size);
            quant_loss.load2cache(i, i + cur_batch_size);

            // 切分数据给不同的线程
            for(auto& cur_ids : filter_ids)
                cur_ids.resize(0);
            for(size_t ii = i; ii < i + cur_batch_size; ++ii){
                uint k = *((uint*)quant_Points[ii].data());
                filter_ids[k % num_workers].push_back(ii);
            }

            // 多线程插入map
            parlay::parallel_for(0, num_workers, [&](size_t wid){
                auto& cur_ids = filter_ids[wid];
                auto& cur_map = all_maps[wid];
                for(auto ii : cur_ids){
                    auto qp = quant_Points[ii];
                    auto loss = quant_loss[ii];
                    // 插入字典
                    float dis = loss[t];
                    std::string k = std::string((const char*)qp.data() + t * chunk, chunk);
                    auto v = pid(ii, dis);
                    auto it = cur_map.find(k);
                    if(it != cur_map.end()){
                        if(it->second.second > dis){
                            it->second = v;
                        }
                    } else {
                        cur_map.insert({k, v});
                    }
                }
            });
            printProgressBar((i + cur_batch_size) / (double)quant_Points.size());
        }

        std::cout<<"start fill dict tree..."<<std::endl;
        // 将数据写入字典树
        for(auto& cur_map : all_maps){
            for(const auto& pr : cur_map){
                dict.insert(t, (const char*)pr.first.c_str(), chunk, pr.second, [&](pid pre_v){return false;});
            }            
        }
        std::cout<<"finish fill dict tree."<<std::endl;
    }

}

int main(int argc, char *argv[])
{
    CLI::App app{"export-dict"};

    std::string pivotFile;
    app.add_option("--pivotFile", pivotFile, "File path for load pivots");

    std::string quantFile;
    app.add_option("--quantFile", quantFile, "File path for load quant");

    std::string quantLossFile = "";
    app.add_option("--quantLossFile", quantLossFile, "File path for load quant loss");

    std::string dictFile = "";
    app.add_option("--dictFile", dictFile, "File path for save quant dict");

    std::string spFile = "";
    app.add_option("--spFile", spFile, "File path for save all start points");

    CLI11_PARSE(app, argc, argv);

    auto pivot = Pivot(pivotFile.c_str());

    /*uint dim, times, pivots_num;
    std::vector<float> pivots;
    std::vector<uint> order_ids;

    std::ifstream reader(pivotFile.c_str());
    reader.read((char*)&times, sizeof(uint));
    reader.read((char*)&pivots_num, sizeof(uint));
    reader.read((char*)&dim, sizeof(uint));
    uint chunk;
    reader.read((char*)&chunk, sizeof(uint));

    pivots = std::vector<float>(times * pivots_num * dim);
    order_ids = std::vector<uint>(times * dim);
    auto td = times * dim;
    auto tc = times * chunk;
    auto ptc = pivots_num * tc;
    auto chunk_dim = dim / chunk;

    reader.read((char*)order_ids.data(), sizeof(float) * times * dim);
    reader.read((char*)pivots.data(), sizeof(float) * times * pivots_num * dim);
    reader.close();  */
    
    // 每次计算多少个底库向量
    const size_t max_cache = 100000;

    DictForest<pid> dict = DictForest<pid>(pivot.times);
    PointRange<PQ_Point<uint8_t>> quant_Points(quantFile.c_str(), max_cache);
    PointRange<Euclidian_Point<float>> quant_loss(quantLossFile.c_str(), max_cache);

    exportDict(quant_Points, quant_loss, pivot, dict, max_cache);
    
    if(dictFile != "")
        dict.save(dictFile.c_str());

    if(spFile != ""){
        std::vector<size_t> all_start_points;
        for(uint i = 0; i < dict.nodes.size(); ++i){
            if(dict.nodes[i].childId == -1){
                all_start_points.push_back(dict.nodes[i].v.first);
            }
        }
        std::sort(all_start_points.begin(),          // 1. 排序
                all_start_points.end());

        auto last = std::unique(all_start_points.begin(),
                                all_start_points.end()); // 2. 去重
        all_start_points.erase(last,                   // 3. 擦掉多余元素
                            all_start_points.end());

        size_t n = all_start_points.size();
        size_t d = 1;
        std::ofstream writer(spFile.c_str());
        writer.write((char*)&n, sizeof(size_t));
        writer.write((char*)&d, sizeof(size_t));
        writer.write((char*)all_start_points.data(), sizeof(size_t) * n);
        writer.close();
        std::cout<< spFile<<" sp num:"<<n<<std::endl;
    }
        
    

    return 0;
}