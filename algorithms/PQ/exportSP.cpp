#include <iostream>
#include <stdlib.h>
#include <vector>
#include <chrono>
#include <fstream>
#include "parlay/parallel.h"
#include "common.h"
#include "utils/euclidian_point.h"
#include "utils/pq_point.h"
#include "utils/point_range.h"
#include "utils/dictForest.h"
#include "utils/pivot.h"
using namespace parlayANN;

template<typename PR, typename indexType>
void exportQuerySP(PR& Query_Points, indexType* start_points_, std::vector<Pivot>& piv, std::vector<DictForest<std::pair<indexType, float>>*>& quant_dict, uint sp_num, uint sp=0){
    auto dim = Query_Points.dimension();

    parlay::parallel_for(0, Query_Points.size(), [&](size_t qid){
        std::vector<indexType> start_points;
        for(uint i = 0; i < piv.size(); ++i){
            // 计算最近轴
            uint tc = piv[i].chunk * piv[i].times;
            std::vector<uint8_t> min_pivot_id = std::vector<uint8_t>(tc);
            auto* cur_min_pivot_id = min_pivot_id.data();
            std::vector<float> min_dis = std::vector<float>(tc);
            std::vector<float> tmp_dis = std::vector<float>(tc);  
            piv[i].encode(Query_Points[qid], cur_min_pivot_id, min_dis.data(), tmp_dis.data());

            // get start point
            for(uint t = 0; t < piv[i].times; ++t){
                auto* cur_quant = cur_min_pivot_id + t * piv[i].chunk;
                auto* data = quant_dict[i]->getValue(t, (const char*)cur_quant, piv[i].chunk);
                if(data != nullptr){
                    indexType cur_sp = data->first;
                    start_points.push_back(cur_sp);
                }  
            }
        }
        // 去重
        std::sort(start_points.begin(), start_points.end()); // 排序
        start_points.erase(std::unique(start_points.begin(), start_points.end()), start_points.end()); // 去除相邻重复元素
        if(start_points.size() == 0)
            start_points.push_back(sp);
        auto* cur_start_points = start_points_ + qid * sp_num;
        for(uint j = 0; j < start_points.size(); ++j)
            cur_start_points[j] = start_points[j];
    });
}

int main(int argc, char *argv[])
{
    CLI::App app{"export-start-point"};
    std::string base_sp_file;
    app.add_option("--base_sp_file", base_sp_file, "save base start point");

    std::string query_sp_file;
    app.add_option("--query_sp_file", query_sp_file, "save query start point");

    std::string df = "Euclidian";
    app.add_option("--dist_func", df, "distance function");

    std::string tp = "uint8";
    app.add_option("--data_type", tp, "data type");

    std::string meta_path;
    app.add_option("--meta_path", meta_path, "Meta file path");

    std::string qFile;
    app.add_option("--query_path", qFile, "query file path");
    // std::string pivotFile;
    // app.add_option("--pivotFile", pivotFile, "File path for load pivots");

    // std::string quantFile;
    // app.add_option("--quantFile", quantFile, "File path for load quant");

    // std::string dictFile;
    // app.add_option("--dictFile", dictFile, "File path for save quant dict");

    CLI11_PARSE(app, argc, argv);

    using indexType = uint;
    using PQPoint = PointRange<PQ_Point<uint8_t>>;

    // 读取字典、轴以及量化编码-------------------------------------------------------------------
    std::string meta_str = meta_path;
    auto first = meta_str.find('[', 0);
    auto last = meta_str.find(']', 0);
    if(first == std::string::npos || first >= last){
        std::cout<<"meta_file:"<<meta_path<<" Error!";
    }
    auto context = meta_str.substr(first + 1, last - first - 1);
    auto base_path = meta_str.substr(0, first);
    auto mid = context.find(',', 0);
    auto first_name = context.substr(0, mid);
    auto second_name = context.substr(mid+1);
    
    std::vector<std::string> dict_files = {base_path + first_name + ".dict", base_path + second_name + ".dict"};
    std::vector<std::string> quant_files = {base_path + first_name + ".quant", base_path + second_name + ".quant"};
    std::vector<Pivot> piv = {Pivot((base_path + first_name + ".piv").c_str()), Pivot((base_path + second_name + ".piv").c_str())};
    // std::vector<std::string> piv_files = {base_path + first_name + ".piv", base_path + second_name + ".piv"};
    
    // std::vector<std::vector<float>> pivots = std::vector<std::vector<float>>(2);
    // std::vector<std::vector<uint>> order_ids = std::vector<std::vector<uint>>(2);
    // uint dim;
    // std::vector<uint> times = {0, 0};
    // std::vector<uint> pivots_num = {0, 0};
    // std::vector<uint> chunk = {0, 0};

    /*uint sp_num = 0;
    for(int i = 0; i < 2; ++i){
        std::ifstream reader(piv_files[i].c_str());
        auto& cur_times = times[i];
        auto& cur_pivots_num = pivots_num[i];
        reader.read((char*)&cur_times, sizeof(uint));
        reader.read((char*)&cur_pivots_num, sizeof(uint));
        reader.read((char*)&dim, sizeof(uint));
        auto& cur_chunk = chunk[i];
        reader.read((char*)&cur_chunk, sizeof(uint));
        sp_num += cur_times;
        pivots[i] = std::vector<float>(cur_times * cur_pivots_num * dim);
        order_ids[i] = std::vector<uint>(cur_times * dim);
        reader.read((char*)order_ids[i].data(), sizeof(float) * cur_times * dim);
        reader.read((char*)pivots[i].data(), sizeof(float) * cur_times * cur_pivots_num * dim);
        reader.close();      
    }*/
    uint sp_num = piv[0].times + piv[1].times;
    using pid = std::pair<indexType, float>;
    auto qd1 = DictForest<pid>(dict_files[0].c_str());
    auto qd2 = DictForest<pid>(dict_files[1].c_str());
    std::vector<DictForest<pid>*> quant_dict = {&qd1, &qd2};
    quant_dict.push_back(&qd1);
    quant_dict.push_back(&qd2);
    // -----------------------------------------------------------------------------------------------


    auto pqp1 = PQPoint((const char *)quant_files[0].c_str(), static_cast<size_t>(0));
    auto pqp2 = PQPoint((const char *)quant_files[1].c_str(), static_cast<size_t>(0));
    std::vector<PQPoint *> quant_points_list = {&pqp1, &pqp2};

    uint n = pqp1.size();
    uint d = sp_num;
    std::ofstream baseWriter(base_sp_file.c_str());
    baseWriter.write((char*)&n, sizeof(uint));
    baseWriter.write((char*)&d, sizeof(uint));
    std::cout<<n<<" "<<d<<std::endl;

    bool random_order = true;
    std::vector<indexType> shuffled_inserts = std::vector<indexType>(n);
    {
        parlay::sequence<int> rperm;
        if (random_order)
          rperm = parlay::random_permutation<int>(static_cast<int>(n));
        else
          rperm = parlay::tabulate(n, [&](int i){ return i; });
        parlay::parallel_for(0, n, [&](size_t i){
          shuffled_inserts[i] = rperm[i];
        });
    }
    baseWriter.write((char*)shuffled_inserts.data(), sizeof(indexType) * n);

    size_t max_batch_size = 100000;
    uint sp = 0;
    std::vector<indexType> start_points_vec = std::vector<indexType>(max_batch_size * d);
    indexType* start_points_ = start_points_vec.data();
    size_t all_sp_num = 0;
    for(size_t i = 0; i < pqp1.size(); i += max_batch_size){
        size_t cur_batch_size = (i + max_batch_size > pqp1.size()) ? pqp1.size() - i : max_batch_size;
        parlay::parallel_for(0, cur_batch_size * d, [&](size_t j){start_points_[j] = -1;});
        // 获取入口点，这是关键代码--------------------------------
        parlay::parallel_for(i, i + cur_batch_size, [&](size_t ii){
            auto index = shuffled_inserts[ii];
            std::vector<indexType> start_points;
            for(uint j = 0; j < piv.size(); ++j){
                auto& qp = (*quant_points_list[j]);
                uint tc = qp.dimension();
                uint cur_time = qp.dimension() / piv[j].chunk;
                auto& cur_dict = *quant_dict[j];
                for(uint t = 0; t < cur_time; ++t){
                    auto* cur_quant = qp[index].data() + t * piv[j].chunk;
                    auto* data = cur_dict.getValue(t, (const char*)cur_quant, piv[j].chunk);
                    if(data != nullptr){
                        indexType cur_sp = data->first;
                        if(cur_sp == index)
                            continue;
                        start_points.push_back(cur_sp);
              
                    }
                }
            }
            // 去重
            std::sort(start_points.begin(), start_points.end()); // 排序
            start_points.erase(std::unique(start_points.begin(), start_points.end()), start_points.end()); // 去除相邻重复元素
            if(start_points.size() == 0){
                start_points.push_back(sp);
            }
            auto* cur_start_points = start_points_ + (ii - i) * d;
            for(int j = 0; j < start_points.size(); ++j)
                cur_start_points[j] = start_points[j];


        });
        for(size_t j = 0; j < cur_batch_size * d; ++j){
            if(start_points_[j] != -1){
                all_sp_num++;
            }
        }
        baseWriter.write((char*)start_points_, sizeof(indexType) * cur_batch_size * d);
        /*for(size_t j = 0; j < cur_batch_size; ++j){

            // test
            if(i + j == 933891336){
                for(int k = 0; k < d; ++k)
                    std::cout<<start_points_[j * d + k]<<" ";
                std::cout<<std::endl;
            }

            assert(start_points_[j * d] != -1);
        }*/
            
        printProgressBar((i + cur_batch_size) / (float)n);
    }
    baseWriter.close();
    std::cout<<" avg sp num:"<<all_sp_num / n<<std::endl;
    
    if(tp == "uint8"){
        if(df == "Euclidian"){
            PointRange<Euclidian_Point<uint8_t>> Query_Points(qFile.c_str());
            n = Query_Points.size();
            start_points_vec = std::vector<indexType>(n * d, -1);
            start_points_ = start_points_vec.data();
            exportQuerySP(Query_Points, start_points_, piv, quant_dict, sp_num);
        }
    } else if(tp == "float"){
        if(df == "Euclidian"){
            PointRange<Euclidian_Point<float>> Query_Points(qFile.c_str());
            n = Query_Points.size();
            start_points_vec = std::vector<indexType>(n * d, -1);
            start_points_ = start_points_vec.data();
            exportQuerySP(Query_Points, start_points_, piv, quant_dict, sp_num);
        }        
    }
    std::ofstream qWriter(query_sp_file.c_str());
    qWriter.write((char*)&n, sizeof(uint));
    qWriter.write((char*)&d, sizeof(uint));   

    shuffled_inserts = std::vector<indexType>(n);
    parlay::parallel_for(0, n, [&](size_t i){
        shuffled_inserts[i] = i;
    });
    qWriter.write((char*)shuffled_inserts.data(), sizeof(indexType) * n);

    qWriter.write((char*)start_points_, sizeof(indexType) * n * d);
    qWriter.close();

    return 0;
}