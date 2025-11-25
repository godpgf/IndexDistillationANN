#pragma once
#include <limits>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <set>
#include "pivot.h"

namespace parlayANN
{
    template<typename indexType>
    struct IVFWriter{
        
        IVFWriter(std::string ivf_path, size_t keyNum, size_t valueNum, size_t max_cache_size=1000000) 
            : ivf_path(ivf_path), 
            keyNum(keyNum), 
            valueNum(valueNum), 
            max_cache_size(max_cache_size), 
            nextPos(keyNum), 
            cur_index(0){
            if(ivf_path != ""){
                init_writer();
            }
        }

        template<typename rangeType>
        size_t fill(const rangeType& r){
            if(nextPos.size() < cur_index){
                std::cout<<"IVFWrite ERROR: nextPos.size() < cur_index"<<std::endl;
                abort();
            }
            auto l = r.size() + (cur_index == 0 ? 0 : nextPos[cur_index-1]);
            if(cur_index == nextPos.size()){
                nextPos.push_back(l);
            } else{
                nextPos[cur_index] = l;
            }
            
            for(auto d : r)
                ivf.push_back(d);
            if(ivf_path != "" && ivf.size() > max_cache_size){
                writer.write((char*)ivf.data(), ivf.size() * sizeof(indexType));
                ivf.clear();
            }
            return cur_index++;
        }

        void finish(){
            if(ivf_path != ""){
                if(ivf.size() > 0){
                    writer.write((char*)ivf.data(), ivf.size() * sizeof(indexType));
                    ivf.clear();
                }
                writer.close();

                if(keyNum == 0){
                    keyNum = nextPos.size();
                    writer.open(ivf_path.c_str(), std::ios::binary | std::ios::out);
                    writer.write((char*)&keyNum, sizeof(size_t));
                    writer.write((char*)&valueNum, sizeof(size_t));
                    writer.write((char*)nextPos.data(), nextPos.size() * sizeof(size_t));
                    // 当前是动态添加的数据
                    std::ifstream reader;
                    reader.open((ivf_path + ".tmp").c_str(), std::ios::binary | std::ios::in);
                    ivf.resize(max_cache_size);
                    
                    for(size_t i = 0; i < nextPos[keyNum-1]; i += max_cache_size){
                        size_t cur_batch_size = (i + max_cache_size > nextPos[keyNum-1]) ? nextPos[keyNum-1] - i : max_cache_size;
                        reader.read((char*)ivf.data(), cur_batch_size * sizeof(indexType));
                        writer.write((char*)ivf.data(), cur_batch_size * sizeof(indexType));
                    }
                } else {
                    // 重新写入
                    writer.open(ivf_path.c_str(), std::ios::binary | std::ios::out | std::ios::in);
                    writer.seekp(sizeof(size_t) * 2, std::ios::beg);
                    writer.write((char*)nextPos.data(), nextPos.size() * sizeof(size_t));
                }

                writer.close();
            }
        }

        template<typename invIndexType>
        void inv_ivf(std::string inv_ivf_path, size_t epoch_num=10){
            // 如v2c转c2v
            if(ivf_path == "")
                return;
            std::ofstream inv_writer;
            std::vector<size_t> inv_nextPos = std::vector<size_t>(valueNum);
            // 写入尺寸
            inv_writer.open(inv_ivf_path.c_str(), std::ios::binary | std::ios::out);        
            inv_writer.write((char*)&valueNum, sizeof(size_t));
            inv_writer.write((char*)&keyNum, sizeof(size_t));
            inv_writer.write((char*)inv_nextPos.data(), valueNum * sizeof(size_t));

            std::ifstream ivf_reader;
            ivf_reader.open(ivf_path.c_str(), std::ios::binary | std::ios::in);
            size_t max_batch_size = valueNum / epoch_num;

            // 开始写入
            for(size_t si = 0; si < valueNum; si += max_batch_size){
                size_t cur_batch_size = (si + max_batch_size > valueNum) ? valueNum - si : max_batch_size; 
                std::vector<std::vector<invIndexType>> new_out_(cur_batch_size);
                
                // 读取所有数据
                ivf_reader.seekg((2 + keyNum) * sizeof(size_t));
                for(auto vid = 0; vid < nextPos.size(); ++vid){
                    size_t pre_pos = (vid == 0) ? 0 : nextPos[vid - 1];
                    std::vector<indexType> cache_data(nextPos[vid] - pre_pos);
                    ivf_reader.read((char*)cache_data.data(), cache_data.size() * sizeof(indexType));
                    // 筛选出内容在范围内的数据
                    for(auto cid : cache_data){
                        if(cid >= si && cid < si + cur_batch_size){
                            new_out_[cid - si].push_back(vid);
                        }
                    }
                }

                // 写入ivf
                for(size_t i = 0; i < cur_batch_size; ++i){
                    auto& cur_out = new_out_[i];
                    inv_writer.write((char*)cur_out.data(), cur_out.size() * sizeof(invIndexType));
                    auto cur_index = si + i;
                    inv_nextPos[cur_index] = cur_out.size() + ((cur_index == 0) ? 0 : inv_nextPos[cur_index - 1]);
                }  
            }
            inv_writer.close();
            ivf_reader.close();

            // 重新写入
            inv_writer.open(inv_ivf_path.c_str(), std::ios::binary | std::ios::out | std::ios::in);
            inv_writer.seekp(sizeof(size_t) * 2, std::ios::beg);
            inv_writer.write((char*)inv_nextPos.data(), inv_nextPos.size() * sizeof(size_t));
            inv_writer.close();
        }

        std::ofstream writer;
        std::string ivf_path;
        size_t keyNum;
        size_t valueNum;
        // 最多缓存多少ivf，当超过时就存盘
        size_t max_cache_size;
        // 记录当前写入的是第几个元素
        size_t cur_index;

        std::vector<size_t> nextPos;
        std::vector<indexType> ivf;

        protected:

        void init_writer(){
            if(nextPos.size() == 0){
                // 动态添加数据
                writer.open((ivf_path + ".tmp").c_str(), std::ios::binary | std::ios::out);
            } else {
                writer.open(ivf_path.c_str(), std::ios::binary | std::ios::out);
                writer.write((char*)&keyNum, sizeof(size_t));
                writer.write((char*)&valueNum, sizeof(size_t));
                // 先占个位置
                writer.write((char*)nextPos.data(), keyNum * sizeof(size_t));
            }
        }

    };

    template<typename indexType>
    struct IVFReader{
        IVFReader(std::string ivf_path="", size_t initIvfCacheSize=0) : ivf_path(ivf_path){
            if(ivf_path != ""){
                reader.open(ivf_path.c_str(), std::ios::binary | std::ios::in);
                reader.read((char*)&keyNum, sizeof(size_t));
                reader.read((char*)&valueNum, sizeof(size_t));
                nextPos.resize(keyNum);
                reader.read((char*)nextPos.data(), nextPos.size() * sizeof(size_t));

                if(initIvfCacheSize==0 || initIvfCacheSize>=nextPos[keyNum-1]){
                    ivf.resize(nextPos[keyNum-1]);
                    reader.read((char*)ivf.data(), nextPos[keyNum-1] * sizeof(indexType)); 
                    // std::cout<<"read size:"<<nextPos[keyNum-1]<<" ivf[1700267]:"<<ivf[1700267]<<std::endl;      
                } else {
                    // 不能全部缓存所有数据
                    indexCache.resize(nextPos.size());
                    ivf.resize(initIvfCacheSize);
                }
            }
        }

        // IVFReader<indexType>* cmpIVF;

        bool is_use_cache(){
            return ivf.size() < nextPos[keyNum-1];
        }

        void load2cache(size_t sid, size_t elementNum){
            if(!is_use_cache())
                return;
            if(sid + elementNum > nextPos.size()){
                std::cout<<"load2cache ERROR: index out of "<<nextPos.size()<<std::endl;
                abort();
            }
            size_t pre_pos = (sid == 0) ? 0 : nextPos[sid-1];
            size_t load_num = nextPos[sid + elementNum - 1] - pre_pos;
            if(load_num == nextPos[keyNum-1]){
                // 缓存全部数据
                ivf.resize(nextPos[keyNum-1]);
                reader.seekg((2 + keyNum) * sizeof(size_t));
                reader.read((char*)ivf.data(), nextPos[keyNum-1] * sizeof(indexType));   
                indexCache.resize(0);               
            } else {
                if(ivf.size() < load_num){
                    ivf.resize(load_num);
                }
                reader.seekg((2 + keyNum) * sizeof(size_t) + pre_pos * sizeof(indexType));
                reader.read((char*)ivf.data(), load_num * sizeof(indexType));
                // 填写索引
                size_t cur_offset = 0;
                for(size_t i = 0; i < elementNum; ++i){
                    indexCache[sid + i] = ivf.data() + cur_offset;
                    cur_offset += nextPos[sid + i] - (sid + i == 0 ? 0 : nextPos[sid + i - 1]);
                }
            }

        }

        template<typename ivfIndexType>
        void load2cache(ivfIndexType* ids, size_t idsNum){
            if(!is_use_cache())
                return;
            size_t cur_offset = 0;
            for(size_t i = 0; i < idsNum; ++i){
                auto sid = ids[i];
                size_t pre_pos = (sid == 0) ? 0 : nextPos[sid-1];
                size_t load_num = nextPos[sid] - pre_pos;
                cur_offset += load_num;
            } 
            if(cur_offset >= nextPos[keyNum-1]){
                // 缓存全部数据
                ivf.resize(nextPos[keyNum-1]);
                reader.seekg((2 + keyNum) * sizeof(size_t));
                reader.read((char*)ivf.data(), nextPos[keyNum-1] * sizeof(indexType));   
                indexCache.resize(0);    

            } else {
                if(cur_offset > ivf.size()){
                    ivf.resize(cur_offset);
                }
                cur_offset = 0;
                for(size_t i = 0; i < idsNum; ++i){
                    auto sid = ids[i];
                    size_t pre_pos = (sid == 0) ? 0 : nextPos[sid-1];
                    size_t load_num = nextPos[sid] - pre_pos;
                    reader.seekg((2 + keyNum) * sizeof(size_t) + pre_pos * sizeof(indexType), std::ios::beg);
                    reader.read((char*)(ivf.data() + cur_offset), load_num * sizeof(indexType));
                    indexCache[sid] = ivf.data() + cur_offset;
                    cur_offset += load_num;
                }  
            }

          
        }

        std::pair<indexType*, size_t> operator [](size_t i){
            size_t pre_pos = (i == 0) ? 0 : nextPos[i-1];
            if(!is_use_cache()){
                return std::pair<indexType*, size_t>(ivf.data() + pre_pos, nextPos[i] - pre_pos);
            } else {
                return std::pair<indexType*, size_t>(indexCache[i], nextPos[i] - pre_pos);
            }
        }

        std::ifstream reader;
        std::string ivf_path;
        size_t keyNum;
        size_t valueNum;
        // size_t max_cache_size;
        std::vector<size_t> nextPos;
        std::vector<indexType> ivf;
        std::vector<indexType*> indexCache;
    };

    struct IVF{
        template<typename indexType>
        static void make_ivf(const std::vector<indexType>& id2cls, std::vector<size_t>& nextPos, std::vector<indexType>& ivf){
            if(nextPos.size() > 0 || ivf.size() > 0){
                std::cout<<"make_ivf:ERROR nextPos or ivf list must empty!"<<std::endl;
                abort();
            }

            // 申请内存
            auto max_it = std::max_element(id2cls.begin(), id2cls.end());
            auto cls_num = (*max_it) + 1;
            nextPos.resize(cls_num);
            ivf.resize(id2cls.size());

            // 统计每个分类中元素的数量
            std::vector<size_t>& cls_size = nextPos;
            for(auto cid : id2cls)
                cls_size.data()[cid]++;
            
            // 计算每个聚类的下一个偏移
            for (size_t cid = 1; cid < cls_num; ++cid)
            {
                nextPos[cid] = nextPos[cid] + nextPos[cid - 1];
            }
            assert(nextPos[cls_num - 1] == id2cls.size());

            // 开始填写归属每个节点的向量id，记录当前填写到第几个
            std::vector<indexType> node_offset(cls_num, 0);
            for (indexType i = 0; i < id2cls.size(); ++i)
            {
                auto cid = id2cls[i];
                size_t pre_offset = (cid == 0 ? 0 : nextPos[cid - 1]);
                ivf[pre_offset + node_offset[cid]] = i;
                node_offset[cid] += 1;
            }
        }

        template<typename indexType>
        static void make_ivf(const std::vector<size_t>& fromNextPos, const std::vector<indexType>& fromIvf, std::vector<size_t>& toNextPos, std::vector<indexType>& toIvf){
            if(toNextPos.size() > 0 || toIvf.size() > 0){
                std::cout<<"make_ivf:ERROR nextPos or ivf list must empty!"<<std::endl;
                abort();
            }  
            
            // 申请内存
            auto max_it = std::max_element(fromIvf.begin(), fromIvf.end());
            auto cls_num = (*max_it) + 1;
            toNextPos.resize(cls_num);
            toIvf.resize(fromIvf.size());    
            
            // 统计每个分类中元素的数量
            std::vector<size_t>& cls_size = toNextPos;
            for(auto cid : fromIvf)
                cls_size.data()[cid]++;

            // 计算每个聚类的下一个偏移
            for (size_t cid = 1; cid < cls_num; ++cid)
            {
                toNextPos[cid] = toNextPos[cid] + toNextPos[cid - 1];
            }
            assert(toNextPos[cls_num - 1] == fromIvf.size());

            // 开始填写归属每个节点的向量id，记录当前填写到第几个
            std::vector<indexType> node_offset(cls_num, 0);
            for(indexType i = 0; i < fromNextPos.size(); ++i){
                size_t fid = (i == 0) ? 0 : fromNextPos[i-1];
                size_t eid = fromNextPos[i];
                for(size_t j = fid; j < eid; ++j){
                    indexType cid = fromIvf[j];
                    size_t pre_offset = (cid == 0 ? 0 : toNextPos[cid - 1]);
                    toIvf[pre_offset + node_offset[cid]] = i;
                    node_offset[cid] += 1;
                }
            }
        }

        template<typename PR, typename PQPR, typename indexType>
        static void split_ivf(IVFReader<indexType>& fromIvf, const std::vector<indexType>& fromCentroidIds, Pivot& pivot, PR& Points, PQPR& quants, size_t maxIVFLen, IVFWriter<indexType>& toIvf, std::vector<indexType>& toCentroidIds){
            if(toIvf.nextPos.size() > 0 || toIvf.ivf.size() > 0 || toCentroidIds.size() > 0){
                std::cout<<"make_ivf:ERROR nextPos or ivf list must empty!"<<std::endl;
                abort();
            }  
            
            if(fromIvf.nextPos.size() != fromCentroidIds.size()){
                std::cout<<"make_ivf:ERROR fromNextPos.size() != fromCentroidIds.size()"<<std::endl;
                abort();                
            }

            // 记录已经使用过的质心
            std::set<indexType> has_seen;

            for(size_t i = 0; i < fromIvf.nextPos.size(); ++i){
                size_t pre_offset = i > 0 ? fromIvf.nextPos[i-1] : 0;
                size_t ivf_size = fromIvf.nextPos[i] - pre_offset;
                if(ivf_size == 0)
                    continue;

                fromIvf.load2cache(i, 1);
                auto [ivf_data, data_size] = fromIvf[i];
                assert(data_size == ivf_size);
    
                if(ivf_size < maxIVFLen && has_seen.find(fromCentroidIds[i]) == has_seen.end()){
                    // 如果当前的ivf没有超长
                    has_seen.insert(fromCentroidIds[i]);
                    toCentroidIds.push_back(fromCentroidIds[i]);

                    toIvf.fill(parlay::tabulate(ivf_size, [&](size_t j){return ivf_data[j];}));
                    // toNextPos.push_back(ivf_size);
                    // for(size_t j = pre_offset; j < fromNextPos[i]; ++j)
                        // toIvf.push_back(fromIvf[j]);
                } else {
                    // ivf超长，那么丢弃之前的质心，开始随机分裂！！！

                    // 1、随机选择split_num个质心
                    std::vector<indexType> randCentroidIds;
                    for(size_t j = 0; j < ivf_size; ++j){
                        if(has_seen.find(ivf_data[j]) == has_seen.end())
                            randCentroidIds.push_back(ivf_data[j]);
                    }

                    // 所有质心都被选择过，将丢弃这个质心的ivf
                    if(randCentroidIds.size() == 0)
                        continue;

                    // 随机选出split_num个新质心
                    uint split_num = ivf_size / maxIVFLen;
                    if(split_num > randCentroidIds.size())
                        split_num = randCentroidIds.size();
                    if(split_num == 0)
                        split_num = 1;
           
                    if(split_num < randCentroidIds.size()){
                        std::mt19937_64 rng{std::random_device{}()}; // 64 位引擎
                        std::shuffle(randCentroidIds.begin(), randCentroidIds.end(), rng);   // 洗牌
                        randCentroidIds.resize(split_num);                               // 只保留前 split_num 个                        
                    }

                    // 预加载数据
                    std::vector<typename PQPR::byte> quant_bytes = std::vector<typename PQPR::byte>(split_num * quants.get_dims());
                    quants.load2cache(randCentroidIds.data(), randCentroidIds.size());
                    parlay::parallel_for(0, randCentroidIds.size(), [&](size_t ii){
                        memcpy(quant_bytes.data() + ii * quants.get_dims(), quants[randCentroidIds[ii]].data(), quants.get_dims() * sizeof(typename PQPR::byte));
                    });

                    Points.load2cache(ivf_data, data_size);
                 
                    // 记录ivf中第几个元素所对应的“新质心的下标”，下面称为id
                    std::vector<indexType> id2cls = std::vector<indexType>(ivf_size, 0);
                    parlay::parallel_for(0, ivf_size, [&](size_t ii){
                        // size_t indexOffset = pre_offset + ii;
                        float minDis = std::numeric_limits<float>::max();
                        size_t min_j;
                        
                        for(size_t j = 0; j < split_num; ++j){
                            //indexType centroidIndex = randCentroidIds[j];
                            //auto* qnt = quants[centroidIndex].data();
                            auto* qnt = quant_bytes.data() + j * quants.get_dims();
                            float dis = pivot.pq_distance_sum(Points[ivf_data[ii]].data(), qnt);
                            // float dis = Points[ivf_data[ii]].pq_distance_sum(pivot.pivots.data(), pivot.order_ids.data(), pivot.times, pivot.chunk, qnt);
                            if(dis < minDis){
                                minDis = dis;
                                min_j = j;
                            }
                        }
                        id2cls[ii] = min_j;
                    });

                    // tmp_ivf记录每个随机质心包含的id，0<=id<ivf_size
                    std::vector<size_t> tmp_nextPos;
                    std::vector<indexType> tmp_ivf;
                    make_ivf(id2cls, tmp_nextPos, tmp_ivf);

                    // 3、写入结果
                    /*if(tmp_nextPos.size() != split_num){
                        std::cout<<"tmp_nextPos.size() ERROR! tmp_nextPos.size()="<<tmp_nextPos.size()<<" split_num="<<split_num<<std::endl;
                        abort();
                    }*/
                    for(size_t j = 0; j < tmp_nextPos.size(); ++j){
                        // 遍历所有新质心
                        size_t tmp_pre_offset = j > 0 ? tmp_nextPos[j-1] : 0;
                        size_t tmp_ivf_size = tmp_nextPos[j] - tmp_pre_offset;
                        if(tmp_ivf_size == 0)
                            continue;  
                        // 插入新质心和它包含的元素数量
                        // has_seen.insert(fromCentroidIds[i]);
                        has_seen.insert(randCentroidIds[j]);
                        toCentroidIds.push_back(randCentroidIds[j]);   

                        // 插入新质心包含的元素 
                        toIvf.fill(parlay::tabulate(tmp_ivf_size, [&](size_t jj){
                            auto cur_id = tmp_ivf[tmp_pre_offset + jj];
                            return ivf_data[cur_id];
                        }));
                                     
                    }
                }
            }

            toIvf.finish();  
        }
    };

}