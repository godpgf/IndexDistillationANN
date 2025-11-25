#include "common.h"

#include <iostream>
#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "utils/euclidian_point.h"
#include "utils/point_range.h"
#include "utils/pivot.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
using namespace parlayANN;


template <typename PR>
void train(Pivot& pivot, PR& points, uint epoch_num, uint init_scale=100){
    using indexType = typename PR::indexType;
    size_t max_cache_size = points.getMaxCache();

    auto& pivots = pivot.pivots;
    auto& order_ids = pivot.order_ids;
    auto chunk = pivot.chunk;
    auto sub_chunk = pivot.sub_chunk;
    auto times = pivot.times;
    auto pivots_num = pivot.pivots_num;
    auto dim = points.dimension();
    auto m = points.size();
    assert(dim % (chunk * sub_chunk) == 0);

    std::streamsize old_precision = std::cout.precision();

    // 同时运行几个pq算法
    uint stc = times * chunk * sub_chunk;
    uint td = times * dim;
    auto sub_chunk_dim = dim / (chunk * sub_chunk);
    size_t num_workers = parlay::num_workers();
    
    // 初始化------------------------------------------------------------------------------------
    size_t init_point_num = pivots_num * init_scale;

    if(init_point_num > points.size()){
        init_point_num = points.size();
    }
    if(max_cache_size == points.size()){
        max_cache_size = init_point_num;
    }
    if(init_point_num > max_cache_size){
        std::cout<<"max_cache_size must more or equal then "<<init_point_num<<std::endl;
        abort();
    }

    

    {
        // 开始初始化pivot_num个质心
        parlay::sequence<indexType> rand_indices = parlay::random_permutation<indexType>(static_cast<indexType>(m), parlay::random(0));
        points.load2cache(rand_indices.data(), init_point_num);
        // 初始化第一个轴
        parlay::parallel_for(0, td, [&](size_t j){
            pivots[j] = points[rand_indices[0]][j % dim];
        });
        // 计算每个向量距离之前轴的最远距离
        std::vector<float> all_dist = std::vector<float>(stc * init_point_num, 0);

        // todo 不知道为什么，明明是bug，但能让结果更好，以后再查
        // std::vector<float> tmp_dist = std::vector<float>(stc * num_workers);
        std::vector<float> tmp_dist = std::vector<float>(stc);

        // 用距离之前的轴最远的向量初始化后面的轴
        for(size_t i = 1; i < pivots_num; ++i){
            float* pre_pivot = pivots.data() + (i-1) * td;

            // 计算某个向量距离前轴的最近距离
            parlay::parallel_for(i, init_point_num, [&](size_t jj){
                // todo 不知道为什么，明明是bug，但能让结果更好，以后再查
                // auto* cur_tmp_dist = tmp_dist.data() + stc * parlay::worker_id();
                auto* cur_tmp_dist = tmp_dist.data() ;

                points[rand_indices[jj]].pq_distance(pre_pivot, order_ids.data(), times, chunk * sub_chunk, cur_tmp_dist);
                float* cur_dist = all_dist.data() + jj * stc;
                if(i == 1){
                    memcpy(cur_dist, cur_tmp_dist, stc * sizeof(float));
                } else {
                    for(auto cid = 0; cid < stc; ++cid)
                        cur_dist[cid] = std::min(cur_dist[cid], cur_tmp_dist[cid]);
                }
            });

            // 找到离之前所有轴最远的向量
            std::vector<size_t> max_ids = std::vector<size_t>(stc, i);
            for(size_t j = i + 1; j < init_point_num; ++j){
                float* cur_dist = all_dist.data() + j * stc;
                for(auto cid = 0; cid < stc; ++cid){
                    float pre_dis = all_dist[max_ids[cid] * stc + cid];
                    if(cur_dist[cid] > pre_dis)
                        max_ids[cid] = j;
                }
            }

            // 用最远向量填写当前轴
            float* cur_pivot = pivots.data() + i * td;
            for(auto cid = 0; cid < stc; ++cid){
                size_t index = rand_indices[max_ids[cid]];
                size_t tid = cid / (chunk * sub_chunk);
                for(auto j = cid * sub_chunk_dim; j < (cid+1)*sub_chunk_dim; ++j){
                    auto oj = order_ids[j];
                    cur_pivot[tid * dim + oj] = points[index][oj];
                }
                            
            }
            printProgressBar((i + 1) / (double)pivots_num);
        }             
    }

    // kmeans-----------------------------------------------------------------------
    size_t max_batch = std::min((size_t)pivots_num * 10000, m);
    
    for(uint epoch_id = 0; epoch_id < epoch_num; ++epoch_id){
        parlay::sequence<indexType> rperm = parlay::random_permutation<indexType>(static_cast<indexType>(m), parlay::random(epoch_id+1));
        // 统计最近某个轴的向量的数量
        std::vector<size_t> pivots_cnt(pivots_num * stc * num_workers, 0);
        // 统计最近某个轴的向量累加
        std::vector<double> pivots_sum(pivots_num * td * num_workers, 0);
        double sum_dis = 0;
        
        std::vector<float> min_dis = std::vector<float>(num_workers * stc);
        std::vector<float> tmp_dis = std::vector<float>(num_workers * stc);
        std::vector<uint> min_pivot_id = std::vector<uint>(num_workers * stc);
        
        for(size_t sid = 0; sid < max_batch; sid += max_cache_size){
            size_t cur_batch_size = (sid + max_cache_size) > max_batch ? max_batch - sid: max_cache_size;
            points.load2cache(rperm.data() + sid, cur_batch_size);
            std::vector<float> stat_dis = std::vector<float>(cur_batch_size, 0);
            
            parlay::parallel_for(sid, sid + cur_batch_size, [&](size_t i){
                auto worker_id = parlay::worker_id();
                // 计算最近轴
                auto worker_offset = worker_id * stc;
                auto* cur_min_dis = min_dis.data() + worker_offset;
                auto* cur_tmp_dis = tmp_dis.data() + worker_offset;
                auto* cur_min_pivot_id = min_pivot_id.data() + worker_offset;
                // std::vector<float> tmp_dist = std::vector<float>(tc);
                size_t index = rperm[i];
                points[index].prefetch();

                pivot.encode(points[index], cur_min_pivot_id, cur_min_dis, cur_tmp_dis);

                // 累加最近轴
                
                size_t* cur_pivots_cnt = pivots_cnt.data() + worker_id * pivots_num * stc;
                double* cur_pivots_sum = pivots_sum.data() + worker_id * pivots_num * td;
                for(auto cid = 0; cid < stc; ++cid){
                    auto pid = cur_min_pivot_id[cid];

                    size_t tid = cid / (chunk*sub_chunk);
                    cur_pivots_cnt[pid * stc + cid]++;
                    for(size_t j = cid * sub_chunk_dim; j < (cid+1) * sub_chunk_dim; ++j){
                        auto oj = order_ids[j];
                        cur_pivots_sum[pid * td + tid * dim + oj] += points[index][oj];
                    }
                }
                // 统计
                float sum_dis = 0;
                for(auto j = 0; j < stc; ++j)
                    sum_dis += cur_min_dis[j];
                stat_dis[i-sid] = sum_dis / td;
            }); 

            
            for(auto dis : stat_dis)
                sum_dis += dis;
            printProgressBar((sid + cur_batch_size) / (double)max_batch);
        }
        // 输出控制条会改变浮点数的输出精度，需要还原回去
        std::cout << std::setprecision(old_precision);
        std::cout<<"loss:"<<sum_dis / max_batch<<std::endl;

        for(uint i = 1; i < num_workers; ++i){
            size_t offset = i * pivots_num * stc;
            for(uint j = 0; j < pivots_num * stc; ++j)
                pivots_cnt[j] += pivots_cnt[offset + j];
            offset = i * pivots_num * td;
            for(uint j = 0; j < pivots_num * td; ++j)
                pivots_sum[j] += pivots_sum[offset + j];
        }
        parlay::parallel_for(0, pivots_num * stc, [&](size_t i){
            auto cnt = pivots_cnt[i]; // 当前t_chunk的命中数量
            auto cur_stc = i % stc; // 当前轴的第几个sub_t_chunk
            auto cur_pid = i / stc; // 当前轴id
            auto cur_time = cur_stc / (chunk * sub_chunk); // 当前tid
            // 当前轴
            auto* cur_pivots = pivots.data() + cur_pid * td + cur_time * dim;
            auto* sum_pivots = pivots_sum.data() + cur_pid * td + cur_time * dim;
            for(uint j = cur_stc * sub_chunk_dim; j < (cur_stc + 1) * sub_chunk_dim; ++j){
                auto oj = order_ids[j];
                cur_pivots[oj] = sum_pivots[oj] / (cnt + 1e-5);
            }
        });
        
    }
}


template <typename PR>
void infer(Pivot& pivot, PR& points, std::string quantFile, std::string quantLossFile)
{
    auto dim = points.dimension();
    size_t max_cache_size = points.getMaxCache();
    if(max_cache_size == points.size())
        max_cache_size /= 100;
    assert(dim % (pivot.chunk * pivot.sub_chunk) == 0);
    // 同时运行几个pq算法
    uint times = pivot.times;
    uint tc = times * pivot.chunk;
    uint sc = pivot.chunk * pivot.sub_chunk;
    uint stc = times * sc;
    uint td = times * dim;
    uint pivots_num = pivot.pivots_num;

    std::ofstream writer(quantFile.c_str());
    unsigned int num_points = points.size();
    unsigned int d = pivot.chunk * times;
    writer.write((char*)(&num_points), sizeof(unsigned int));
    writer.write((char*)(&d), sizeof(unsigned int));

    std::vector<float> min_dis = std::vector<float>(parlay::num_workers() * stc);
    std::vector<float> tmp_dis = std::vector<float>(parlay::num_workers() * stc);  
    std::vector<uint8_t> min_pivot_id = std::vector<uint8_t>(tc * max_cache_size);

    // 记录loss--------------------------------------------------------------------
    std::ofstream loss_writer;
    std::vector<float> loss;
    if(quantLossFile != ""){
        loss_writer.open(quantLossFile.c_str(), std::ios::binary | std::ios::out);
        loss.resize(times * max_cache_size);
        loss_writer.write((char*)(&num_points), sizeof(unsigned int));
        loss_writer.write((char*)(&times), sizeof(unsigned int));
    }
    // ---------------------------------------------------------------------------

    for(size_t sid = 0; sid < points.size(); sid += max_cache_size){
        size_t cur_batch_size = (sid + max_cache_size) > points.size() ? points.size() - sid: max_cache_size;
        points.load2cache(sid, sid + cur_batch_size);

        if(loss.size() > 0){
            memset(loss.data(), 0, cur_batch_size * times * sizeof(float));
        }

        parlay::parallel_for(sid, sid + cur_batch_size, [&](size_t i){
            // 计算最近轴
            auto worker_offset = parlay::worker_id() * stc;
            auto* cur_min_dis = min_dis.data() + worker_offset;
            auto* cur_tmp_dis = tmp_dis.data() + worker_offset;
            auto* cur_min_pivot_id = min_pivot_id.data() + (i - sid) * tc;

            pivot.compress_encode(points[i], cur_min_pivot_id, cur_min_dis, cur_tmp_dis);

            if(loss.size() > 0){
                // 需要写入loss
                auto* cur_loss = loss.data() + (i - sid) * times;
                for(uint j = 0; j < stc; ++j){
                    cur_loss[j / sc] += cur_tmp_dis[j];
                }
            }

        });  

        if(loss.size() > 0){
            loss_writer.write((char*)loss.data(), cur_batch_size * times * sizeof(float));
        }
        
            
        writer.write((char*)min_pivot_id.data(), tc * cur_batch_size * sizeof(uint8_t));
        printProgressBar((sid + cur_batch_size) / (double)points.size());
    }
    writer.close();

    // 写入loss------------------------
    if(quantLossFile != ""){
        loss_writer.close();
    }
    
}

int main(int argc, char *argv[])
{
    CLI::App app{"pq"};

    // train--------------------------------------------------------------------
    auto train_cmd = app.add_subcommand("train", "train");

    std::string iFile;
    train_cmd->add_option("--ifile", iFile, "File path for input vectors");

    std::string pivotFile;
    train_cmd->add_option("--pivotFile", pivotFile, "File path for save pivots");

    uint pivots_num = 256;
    train_cmd->add_option("--pivots_num", pivots_num, "num centroids");

    uint chunk = 32;
    train_cmd->add_option("--chunk", chunk, "num chunks");

    uint sub_chunk = 1;
    train_cmd->add_option("--sub_chunk", sub_chunk, "num sub chunks");

    uint max_reps = 8;
    train_cmd->add_option("--max_reps", max_reps, "max reps");

    uint times = 1;
    train_cmd->add_option("--times", times, "pq times");

    std::string df = "Euclidian";
    train_cmd->add_option("--dist_func", df, "distance function");

    std::string tp = "uint8";
    train_cmd->add_option("--data_type", tp, "data type");

    uint max_cache = 100000;
    train_cmd->add_option("--max_cache", max_cache, "max points cache");

    auto infer_cmd = app.add_subcommand("infer", "infer");

    infer_cmd->add_option("--ifile", iFile, "File path for input vectors");
    infer_cmd->add_option("--pivotFile", pivotFile, "File path for save pivots");

    std::string quantFile;
    infer_cmd->add_option("--quantFile", quantFile, "File path for save quant");

    std::string quantLossFile = "";
    infer_cmd->add_option("--quantLossFile", quantLossFile, "File path for save quant loss");

    infer_cmd->add_option("--dist_func", df, "distance function");
    infer_cmd->add_option("--data_type", tp, "data type");
    infer_cmd->add_option("--max_cache", max_cache, "max points cache");

    CLI11_PARSE(app, argc, argv);

    assert(pivots_num <= 256);

    // std::vector<float> pivots;
    // std::vector<uint> order_ids;


    if(*train_cmd){
        parlay::internal::timer t_train("ANN");
        uint dim;
        if(tp == "uint8"){
            if(df == "Euclidian"){
                using PR = PointRange<Euclidian_Point<uint8_t>>;
                PR Points(iFile.c_str(), max_cache);
                std::vector<typename PR::byte*> index_cache(Points.size());
                Points.setIndexCache(index_cache.data());
                dim = Points.dimension();
                Pivot pivot = Pivot(times, pivots_num, dim, chunk, sub_chunk);
                train(pivot, Points, max_reps); 
                pivot.save(pivotFile.c_str());
            }
        } else if(tp == "float"){
             if(df == "Euclidian"){
                using PR = PointRange<Euclidian_Point<float>>;
                PR Points(iFile.c_str(), max_cache);
                std::vector<typename PR::byte*> index_cache(Points.size());
                Points.setIndexCache(index_cache.data());
                dim = Points.dimension();
                Pivot pivot = Pivot(times, pivots_num, dim, chunk, sub_chunk);
                train(pivot, Points, max_reps); 
                pivot.save(pivotFile.c_str());
            }           
        }
        std::cout<<"[Timer 1]:train pq:"<<t_train.next_time()<<std::endl;

        // Pivot::save(pivotFile.c_str(), pivots.data(), order_ids.data(), dim, times, pivots_num, chunk);
    }

    if(*infer_cmd){
        uint dim;

        auto pivot = Pivot(pivotFile.c_str());
        parlay::internal::timer t_infer("ANN");
        if(tp == "uint8"){
            if(df == "Euclidian"){
                using PR = PointRange<Euclidian_Point<uint8_t>>;
                PR Points(iFile.c_str(), max_cache);
                infer(pivot, Points, quantFile, quantLossFile);
            }
        } else if(tp == "float"){
             if(df == "Euclidian"){
                using PR = PointRange<Euclidian_Point<float>>;
                PR Points(iFile.c_str(), max_cache);
                infer(pivot, Points, quantFile, quantLossFile);
            }           
        }
        std::cout<<"[Timer 2]:infer pq:"<<t_infer.next_time()<<std::endl;
        
    }


    return 0;
}