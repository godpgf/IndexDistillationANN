#include <iostream>
#include <stdlib.h>
#include <chrono>
#include "parlay/parallel.h"
#include "common.h"
#include "utils/euclidian_point.h"
#include "utils/pq_point.h"
#include "utils/point_range.h"
#include "utils/pivot.h"
using namespace parlayANN;

using pid = std::pair<size_t, float>;

template <typename GroundTruthType>
float cmpRecall(const GroundTruthType &GT, std::vector<std::vector<pid>> &res_recall, const long k = 100)
{
    float recall = 0;
    if (GT.size() > 0)
    {
        size_t n = res_recall.size();

        int numCorrect = 0;
        for (auto i = 0; i < n; i++)
        {
            parlay::sequence<size_t> results_with_ties;
            for (auto l = 0; l < k; l++)
                results_with_ties.push_back((size_t)GT.coordinates(i, l));

            std::set<size_t> reported_nbhs;
            for (auto l : res_recall[i])
                reported_nbhs.insert(l.first);
            for (auto l = 0; l < results_with_ties.size(); l++)
            {
                size_t t = results_with_ties[l];
                if (reported_nbhs.find(t) != reported_nbhs.end())
                {
                    numCorrect += 1;
                }
            }
        }
        recall = static_cast<float>(numCorrect) / static_cast<float>(k * n);
    }
    return recall;
}

template <typename dataType>
void test(std::string pivotFile, std::string qFile, std::string gtFile, std::string quantFile, uint recall_num, uint topk, bool combine_dis = false)
{
    Pivot pivot = Pivot(pivotFile.c_str());
    if(pivot.sub_chunk == 1){
        combine_dis = false;
    }

    // 每次计算多少个底库向量
    const size_t max_cache = 1000;
    // PointRange<Euclidian_Point<float>> Points(iFile.c_str(), max_cache);
    PointRange<Euclidian_Point<dataType>> Query(qFile.c_str());
    PointRange<PQ_Point<uint8_t>> quant_Points(quantFile.c_str(), max_cache);
    quant_Points.params.set_meta(pivot.sub_chunk, pivot.get_sub_bit(), combine_dis);
    if(quant_Points.getMaxCache() != max_cache){
        std::cout<<quant_Points.getMaxCache()<<" != "<<max_cache<<std::endl;
        abort();
    }

    // 缓存最靠近查询向量的底库向量
    auto q_num = Query.size();
    std::vector<std::vector<pid>> pq_recall;
    std::pair<int, float> *top_dist_and_pos = new std::pair<int, float>[q_num];
    memset(top_dist_and_pos, 0, sizeof(std::pair<int, float>) * q_num);
    for (size_t i = 0; i < q_num; ++i)
    {
        pq_recall.push_back(std::vector<pid>());
    }

    // 存储查询向量到每个轴的距离
    // std::vector<float> q2p_dis = std::vector<float>(Query.size() * ptc);
    auto pstc = pivot.pivots_num * pivot.times * pivot.chunk * pivot.sub_chunk;
    PointRange<Euclidian_Point_PQ> q2p((unsigned int)Query.size(), (unsigned int)pstc);

    auto cptc = pivot.get_combine_pivots_num() * pivot.times * pivot.chunk;
    PointRange<Euclidian_Point_PQ> q2cp((unsigned int)Query.size(), (unsigned int)cptc);

    parlay::parallel_for(0, Query.size(), [&](size_t i){
        pivot.preCalculateDis(Query[i], q2p[i].data());

        if(combine_dis){
            pivot.combineDis(q2p[i].data(), q2cp[i].data());
        }
    });

    // 计算查询向量与所有当前读出向量的距离
    parlay::internal::timer t_infer("ANN");
    float *res_buffer = new float[q_num * max_cache];
    for (size_t sid = 0; sid < quant_Points.size(); sid += max_cache)
    {
        size_t cur_batch_size = (sid + max_cache) > quant_Points.size() ? quant_Points.size() - sid : max_cache;
        quant_Points.load2cache(sid, sid + cur_batch_size);
        parlay::parallel_for(0, Query.size(), [&](size_t i)
                             {
                    float* cur_res_buffer = res_buffer + i * cur_batch_size;
                    // auto* cur_q2p_dis = q2p_dis.data() + i * ptc;
                    for(size_t j = 0; j < cur_batch_size; ++j){
                        // cur_res_buffer[j] = quant_Points[sid + j].distance(cur_q2p_dis);
                        if(combine_dis){
                            cur_res_buffer[j] = quant_Points[sid + j].distance(q2cp[i]);
                        } else {
                            cur_res_buffer[j] = quant_Points[sid + j].distance(q2p[i]);
                        }
                        
                    } });

        // fill res
        parlay::parallel_for(0, q_num, [&](size_t qi)
                             {
                    float& topdist = top_dist_and_pos[qi].second;
                    int& toppos = top_dist_and_pos[qi].first;
                    auto& topk = pq_recall[qi];
                    for (size_t vi = 0; vi < cur_batch_size; ++vi){
                        float dist = res_buffer[qi * cur_batch_size + vi];
                        size_t index = sid + vi;
                                   
                        if(topk.size() < recall_num){
                            if(dist > topdist){
                                topdist = dist;
                                toppos = topk.size();
                            }
                            topk.push_back(std::make_pair(index, dist));
                        }
                        else if(dist < topdist || (dist == topdist && index < topk[toppos].first)){
                            float new_topdist=0;
                            int new_toppos=0;
                            topk[toppos] = std::make_pair(index, dist);
                            for(size_t l=0; l<topk.size(); l++){
                                if(topk[l].second > new_topdist){
                                    new_topdist = topk[l].second;
                                    new_toppos = (int) l;
                                }
                            }
                            topdist = new_topdist;
                            toppos = new_toppos;
                        }

                    } });

        printProgressBar((sid + cur_batch_size) / (quant_Points.size() + 1e-5));
    }
    std::cout<<"[Timer 1]:infer pq:"<<t_infer.next_time()<<std::endl;
    groundTruth<uint> GT = groundTruth<uint>(gtFile.c_str(), false);
    std::cout << "recall " << topk << "@" << recall_num << " = " << cmpRecall(GT, pq_recall, topk) << std::endl;
}

int main(int argc, char *argv[])
{
    CLI::App app{"test-pq"};

    std::string pivotFile;
    app.add_option("--pivotFile", pivotFile, "File path for save pivots");

    std::string qFile;
    app.add_option("--query_path", qFile, "query vectors path");

    std::string gtFile;
    app.add_option("--gt_path", gtFile, "gt path");

    std::string quantFile;
    app.add_option("--quantFile", quantFile, "File path for save quant");

    uint recall_num = 320;
    app.add_option("--recall_num", recall_num, "recall num");

    uint topk = 100;
    app.add_option("--topk", topk, "topk");

    std::string df = "Euclidian";
    app.add_option("--dist_func", df, "distance function");

    std::string tp = "uint8";
    app.add_option("--data_type", tp, "data type");

    bool combine_dis = false;
    app.add_flag("--combine_dis", combine_dis, "combine dis");

    CLI11_PARSE(app, argc, argv);

    if (tp == "uint8")
    {
        if (df == "Euclidian")
        {
            test<uint8_t>(pivotFile, qFile, gtFile, quantFile, recall_num, topk, combine_dis);
        }
    }
    else if (tp == "float")
    {
        if (df == "Euclidian")
        {
            test<float>(pivotFile, qFile, gtFile, quantFile, recall_num, topk, combine_dis);
        }
    }

    return 0;
}