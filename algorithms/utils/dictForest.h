#pragma once
#include <string.h>
#include <map>
#include <fstream>
#include <cassert>
#include <iostream>
#include "parlay/parallel.h"

namespace parlayANN
{
    template<typename T>
    struct DictForest{
        struct TreeNode
        {
            TreeNode(char q=0) : q(q), childId(-1), brotherId(-1) {}
            T v;
            char q;
            size_t childId;
            size_t brotherId;
        };
        DictForest(uint tree_num) : tree_num(tree_num) {
            for(uint i = 0; i < tree_num; ++i){
                auto dummy_node = TreeNode(0);
                nodes.push_back(dummy_node);                
            }
        }

        DictForest(const char* filename){
            if(filename == nullptr){
                std::cout << "ERROR: load filename is nullptr!" << std::endl;
                abort();                
            }
            std::ifstream reader(filename);
            reader.read((char*)(&tree_num), sizeof(unsigned int));
            uint n;
            reader.read((char*)(&n), sizeof(uint));  
            nodes = std::vector<TreeNode>(n);
            reader.read((char*)nodes.data(), n * sizeof(TreeNode));
            reader.close();
        }

        void save(const char* filename){
            if (filename == NULL) {
                std::cout << "ERROR: save filename is nullptr!" << std::endl;
                abort();
            }
            std::ofstream writer(filename);
            uint n = nodes.size();
            writer.write((char*)(&tree_num), sizeof(unsigned int));
            writer.write((char*)(&n), sizeof(uint));  
            writer.write((char*)nodes.data(), n * sizeof(TreeNode));
            writer.close();             
        }

        T* getValue(uint tree_id, const char* text, uint char_num){
            size_t pre_node_id = tree_id;
            size_t pre_node_size = nodes.size();
            for(uint j = 0; j < char_num; ++j){
                if (nodes[pre_node_id].childId == -1)
                {
                    return nullptr;
                }
                else
                {
                    size_t child_id = nodes[pre_node_id].childId;
                    while (nodes[child_id].q != text[j] && nodes[child_id].brotherId != -1)
                    {
                        child_id = nodes[child_id].brotherId;
                    }
                    if (nodes[child_id].q == text[j])
                    {
                        pre_node_id = child_id;
                    }
                    else
                    {
                        return nullptr;
                    }
                }
            }
            return &nodes[pre_node_id].v;          
        }

        void insert(uint tree_id, const char* text, uint char_num, T v, std::function<bool(T)> is_replace){
            size_t pre_node_id = tree_id;
            size_t pre_node_size = nodes.size();
            for(uint j = 0; j < char_num; ++j){
                if (nodes[pre_node_id].childId == -1)
                {
                    // new child node
                    auto tn = TreeNode(text[j]);
                    nodes[pre_node_id].childId = nodes.size();
                    pre_node_id = nodes.size();
                    nodes.push_back(tn);
                }
                else
                {
                    size_t child_id = nodes[pre_node_id].childId;
                    while (nodes[child_id].q != text[j] && nodes[child_id].brotherId != -1)
                    {
                        child_id = nodes[child_id].brotherId;
                    }
                    if (nodes[child_id].q == text[j])
                    {
                        pre_node_id = child_id;
                    }
                    else
                    {
                        // new bro node
                        auto tn = TreeNode(text[j]);
                        nodes[child_id].brotherId = nodes.size();
                        pre_node_id = nodes.size();
                        nodes.push_back(tn);
                    }
                }
            }
            if(pre_node_size == nodes.size()){
                // 替换节点
                if(is_replace(nodes[pre_node_id].v))
                    nodes[pre_node_id].v = v;
            } else {
                // 填写值
                nodes[pre_node_id].v = v;
            }
        }

        size_t getElementNum(){
            size_t num = 0;
            for(auto n : nodes){
                if(n.childId == -1)
                    num++;
            }
            return num;
        }

        template<typename PR>
        size_t fillQuants(PR& quants){
            std::vector<uint8_t> cache = std::vector<uint8_t>(quants.dimension());
            size_t quant_id = 0;
            for(size_t i = 0; i < tree_num; ++i){
                _fill_quant<PR>(quants, cache.data(), 0, i, quant_id);
            }
            return quant_id;
        } 

        std::vector<TreeNode> nodes;
        uint tree_num;

        template<typename PR>
        void _fill_quant(PR& quants, uint8_t* cache, int step, size_t nid, size_t& quant_id){
            // 先填写当前量化缓存
            if(step > 0){
                cache[step - 1] = (uint8_t)nodes[nid].q;
            }
            // 量化缓存填满时填写量化值
            if(nodes[nid].childId == -1){
                memcpy(quants[quant_id++].data(), cache, step * sizeof(uint8_t));
            } else {
                // 填写下一个量化缓存
                _fill_quant(quants, cache, step+1, nodes[nid].childId, quant_id);
            }
            // 填写兄弟缓存
            if(nodes[nid].brotherId != -1){
                _fill_quant(quants, cache, step, nodes[nid].brotherId, quant_id);
            } 
        }
    };
}