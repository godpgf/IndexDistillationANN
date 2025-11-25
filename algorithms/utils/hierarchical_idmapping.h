#pragma once
#include <string.h>
#include <fstream>
#include <cmath>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"

namespace parlayANN
{
    struct HIdMapping
    {
        HIdMapping(size_t max_node_num) : max_node_num(max_node_num), cur_node_num(0)
        {
            id_in_pre_level = new int64_t[max_node_num];
        }

        HIdMapping(const char* file_path){
            std::ifstream reader;
            reader.open(file_path, std::ios::binary | std::ios::in);
            reader.read((char*)&max_node_num, sizeof(size_t));
            reader.read((char*)&cur_node_num, sizeof(size_t));
            reader.read((char*)&sp, sizeof(size_t));
            id_in_pre_level = new int64_t[max_node_num];
            reader.read((char*)id_in_pre_level, sizeof(int64_t) * cur_node_num);
            reader.close();
        }

        void save(const char* file_path){

            std::ofstream writer;
            writer.open(file_path, std::ios::binary | std::ios::out);
            writer.write((char*)&max_node_num, sizeof(size_t));
            writer.write((char*)&cur_node_num, sizeof(size_t));
            writer.write((char*)&sp, sizeof(size_t));
            writer.write((char*)id_in_pre_level, sizeof(int64_t) * cur_node_num);
            writer.close();
        }

        ~HIdMapping()
        {
            delete[] id_in_pre_level;
        }

        uint get_level(int64_t node_id){
            uint l = 0;
            assert(node_id >= 0);
            int64_t pre_id = get_id_in_pre_level(node_id);
            while(pre_id >= 0){
                pre_id = get_id_in_pre_level(pre_id);
                ++l;
            }
            return l;
        }


        int64_t get_id_in_pre_level(int64_t node_id){
            assert(node_id < cur_node_num);
            return id_in_pre_level[node_id];
        }

        void set_id_in_pre_level(int64_t node_id, int64_t pre_id){
            assert(node_id != pre_id && node_id < cur_node_num);
            id_in_pre_level[node_id] = pre_id;
            // assert(node_id < cur_node_num);
        }

        void set_sp(size_t start_point){
            this->sp = start_point;
        }

        size_t get_sp(){
            return sp;
        }

        size_t get_root_id(int64_t node_id){
            int64_t pre_id = node_id;
            while(pre_id >= 0)
                pre_id = id_in_pre_level[pre_id];
            return -pre_id-1;
        }

        // 注意，节点数不包含level=0的节点，level=0的节点id直接对应真实的向量id
        size_t max_node_num;
        size_t cur_node_num;
        size_t sp;

        int64_t* id_in_pre_level;
    };

}