// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
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

#include <iostream>
#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parse_command_line.h"
#include "time_loop.h"
#include "../utils/NSGDist.h"
#include "../utils/euclidian_point.h"
#include "../utils/point_range.h"
#include "../utils/mips_point.h"
#include "../utils/graph.h"


#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

using namespace parlayANN;

// *************************************************************
//  TIMING
// *************************************************************

using uint = unsigned int;


template<typename Point, typename PointRange, typename indexType>
void timeNeighbors(Graph<indexType> &G,
		   PointRange &Query_Points, long k,
		   BuildParams &BP, char* outFile,
		   groundTruth<indexType> GT, char* meta_file, char* res_file, bool graph_built, PointRange &Points)
{


    time_loop(1, 0,
      [&] () {},
      [&] () {
        ANN<Point, PointRange, indexType>(G, k, BP, Query_Points, GT, meta_file, res_file, graph_built, Points);
      },
      [&] () {});

    if(outFile != NULL) {
      G.save(outFile);
    }

}


int main(int argc, char* argv[]) {
    commandLine P(argc,argv,
    "[-a <alpha>] [-d <delta>] [-R <deg>]"
        "[-L <bm>] [-k <k> ]  [-gt_path <g>] [-query_path <qF>]"
        "[-graph_path <gF>] [-graph_outfile <oF>] [-res_path <rF>]" "[-num_passes <np>]"
        "[-memory_flag <algoOpt>] [-mst_deg <q>] [-num_clusters <nc>] [-cluster_size <cs>]"
        "[-data_type <tp>] [-dist_func <df>] [-base_path <b>] <inFile>");

  char* alg_type = P.getOptionValue("-alg_type");
  const char* iFile = P.getOptionValue("-base_path");
  char* oFile = P.getOptionValue("-graph_outfile");
  //char* t_oFile = P.getOptionValue("-tree_outfile");
  char* gFile = P.getOptionValue("-graph_path");
  const char* qFile = P.getOptionValue("-query_path");
  char* cFile = P.getOptionValue("-gt_path");
  char* rFile = P.getOptionValue("-res_path");
  char* metaFile = P.getOptionValue("-meta_path");
  char* vectype = P.getOptionValue("-data_type");
  long Q = P.getOptionIntValue("-Q", 0);
  long R = P.getOptionIntValue("-R", 0);
  if(R<0) P.badArgument();
  long PR = P.getOptionIntValue("-PR", 0);
  if(PR < R)
    PR = R;
  long L = P.getOptionIntValue("-L", 0);
  if(L<0) P.badArgument();

  long MST_deg = P.getOptionIntValue("-mst_deg", 0);
  if(MST_deg < 0) P.badArgument();
  long num_clusters = P.getOptionIntValue("-num_clusters", 0);
  if(num_clusters<0) P.badArgument();
  long cluster_size = P.getOptionIntValue("-cluster_size", 0);
  if(cluster_size<0) P.badArgument();
  double radius  = P.getOptionDoubleValue("-radius", 0.0);
  double radius_2  = P.getOptionDoubleValue("-radius_2", radius);
  long k = P.getOptionIntValue("-k", 0);
  if (k > 1000 || k < 0) P.badArgument();
  double alpha = P.getOptionDoubleValue("-alpha", 1.0);
  double init_alpha = P.getOptionDoubleValue("-init_alpha", 1.0);
  double cos_angle = P.getOptionDoubleValue("-cos_angle", 0.0);
  double init_cos_angle = P.getOptionDoubleValue("-init_cos_angle", 0.4226);
  if(cos_angle < 1e-5){
    init_cos_angle = 0;
  }
  int num_passes = P.getOptionIntValue("-num_passes", 1);
  int two_pass = P.getOptionIntValue("-two_pass", 0);
  if(two_pass > 1 | two_pass < 0) P.badArgument();
  if (two_pass == 1) num_passes = 2;
  double delta = P.getOptionDoubleValue("-delta", 0);
  if(delta<0) P.badArgument();
  char* dfc = P.getOptionValue("-dist_func");
  int quantize = P.getOptionIntValue("-quantize_bits", 0);
  int quantize_build = P.getOptionIntValue("-quantize_mode", 0);
  bool verbose = P.getOption("-verbose");
  bool normalize = P.getOption("-normalize");
  bool rebuild = P.getOption("-rebuild");
  double trim = P.getOptionDoubleValue("-trim", 0.0); // not used
  bool self = P.getOption("-self");
  int rerank_factor = P.getOptionIntValue("-rerank_factor", 100);
  bool range = P.getOption("-range");
  //bool to_tree = P.getOption("-to_tree");
  bool use_madvise = P.getOption("-use_madvise");

  // this integer represents the number of random edges to start with for
  // inserting in a single batch per round
  int single_batch = P.getOptionIntValue("-single_batch", 0);

  // start point type
  int sp_type = P.getOptionIntValue("-sp_type", 0);
    
  std::string df = std::string(dfc);
  std::string tp = std::string(vectype);

  BuildParams BP;
  double node_num_scale = 1;
  std::cout<<alg_type<<std::endl;
  if(strcmp(alg_type, "HNSW") == 0){
    double m_l = 1.0f / log(R);
    BP = BuildParams(R, PR, L, m_l, alpha, verbose);
    node_num_scale = exp(-1.0f / m_l) * 2 + 1;
  } else{
    BP = BuildParams(R, PR, L, alpha, init_alpha, cos_angle, init_cos_angle, num_passes, num_clusters, cluster_size, MST_deg, delta, verbose, quantize_build, radius, radius_2, self, range, single_batch, Q, trim, rerank_factor);
    BP.sp_type = sp_type;
  }
  BP.alg_type = alg_type;
  long maxDeg = BP.max_degree();

  if((tp != "uint8") && (tp != "int8") && (tp != "float")){
    std::cout << "Error: vector type not specified correctly, specify int8, uint8, or float" << std::endl;
    abort();
  }

  if(df != "Euclidian" && df != "mips"){
    std::cout << "Error: specify distance type Euclidian or mips" << std::endl;
    abort();
  }

  // has graph has builded
  bool graph_built = (gFile != NULL && rebuild == false);

  groundTruth<uint> GT = groundTruth<uint>(cFile);
  
  if(tp == "float"){
    if(df == "Euclidian"){
      PointRange<Euclidian_Point<float>> Points(iFile);
      PointRange<Euclidian_Point<float>> Query_Points(qFile);
      if (normalize) {
        std::cout << "normalizing data" << std::endl;
        for (int i=0; i < Points.size(); i++) 
          Points[i].normalize();
        for (int i=0; i < Query_Points.size(); i++) 
          Query_Points[i].normalize();
      }
      Graph<unsigned int> G;
      if(gFile == NULL) G = Graph<unsigned int>(maxDeg, (size_t)(Points.size() * node_num_scale), use_madvise);
      else G = Graph<unsigned int>(gFile, use_madvise);
      if (quantize == 8) {
        std::cout << "quantizing data to 1 byte" << std::endl;
        using QT = uint8_t;
        using QPoint = Euclidian_Point<QT>;
        using PR = PointRange<QPoint>;
        PR Points_(Points);
        PR Query_Points_(Query_Points, Points_.params);
        timeNeighbors<QPoint, PR, uint>(G, Query_Points_, k, BP, oFile, GT, metaFile, rFile, graph_built, Points_);
      } else if (quantize == 16) {
        std::cout << "quantizing data to 2 bytes" << std::endl;
        using Point = Euclidian_Point<uint16_t>;
        using PR = PointRange<Point>;
        PR Points_(Points);
        PR Query_Points_(Query_Points, Points_.params);
        timeNeighbors<Point, PR, uint>(G, Query_Points_, k, BP, oFile, GT, metaFile, rFile, graph_built, Points_);
      } else {
        using Point = Euclidian_Point<float>;
        using PR = PointRange<Point>;
        timeNeighbors<Point, PR, uint>(G, Query_Points, k, BP, oFile, GT, metaFile, rFile, graph_built, Points);
      }
    } else if(df == "mips"){
      PointRange<Mips_Point<float>> Points(iFile);
      PointRange<Mips_Point<float>> Query_Points(qFile);
      if (normalize) {
        std::cout << "normalizing data" << std::endl;
        for (int i=0; i < Points.size(); i++) 
          Points[i].normalize();
        for (int i=0; i < Query_Points.size(); i++) 
          Query_Points[i].normalize();
      }
      Graph<unsigned int> G;
      if(gFile == NULL) G = Graph<unsigned int>(maxDeg, (size_t)(Points.size() * node_num_scale), use_madvise);
      else G = Graph<unsigned int>(gFile, use_madvise);
      if (quantize == 8) {
        std::cout << "quantizing data to 1 byte" << std::endl;
        using QT = int8_t;
        using Point = Quantized_Mips_Point<8>;
        using PR = PointRange<Point>;
        PR Points_(Points);
        PR Query_Points_(Query_Points, Points_.params);
        timeNeighbors<Point, PR, uint>(G, Query_Points_, k, BP, oFile, GT, metaFile, rFile, graph_built, Points_);
      } else if (quantize == 16) {
        std::cout << "quantizing data to 2 bytes" << std::endl;
        using QT = int16_t;
        using Point = Quantized_Mips_Point<16>;
        using PR = PointRange<Point>;
        PR Points_(Points);
        PR Query_Points_(Query_Points, Points_.params);
        timeNeighbors<Point, PR, uint>(G, Query_Points_, k, BP, oFile, GT, metaFile, rFile, graph_built, Points_);
      } else {
        using Point = Mips_Point<float>;
        using PR = PointRange<Point>;
        timeNeighbors<Point, PR, uint>(G, Query_Points, k, BP, oFile, GT, metaFile, rFile, graph_built, Points);
      }
    }
  } else if(tp == "uint8"){
    if(df == "Euclidian"){
      PointRange<Euclidian_Point<uint8_t>> Points(iFile);
      PointRange<Euclidian_Point<uint8_t>> Query_Points(qFile);
      Graph<unsigned int> G;
      if(gFile == NULL) G = Graph<unsigned int>(maxDeg, (size_t)(Points.size() * node_num_scale), use_madvise);
      else G = Graph<unsigned int>(gFile, use_madvise);
      timeNeighbors<Euclidian_Point<uint8_t>, PointRange<Euclidian_Point<uint8_t>>, uint>(G, Query_Points, k, BP,
        oFile, GT, metaFile, rFile, graph_built, Points);
    } else if(df == "mips"){
      PointRange<Mips_Point<uint8_t>> Points(iFile);
      PointRange<Mips_Point<uint8_t>> Query_Points(qFile);
      Graph<unsigned int> G;
      if(gFile == NULL) G = Graph<unsigned int>(maxDeg, (size_t)(Points.size() * node_num_scale), use_madvise);
      else G = Graph<unsigned int>(gFile, use_madvise);
      timeNeighbors<Mips_Point<uint8_t>, PointRange<Mips_Point<uint8_t>>, uint>(G, Query_Points, k, BP,
        oFile, GT, metaFile, rFile, graph_built, Points);
    }
  } else if(tp == "int8"){
    if(df == "Euclidian"){
      PointRange<Euclidian_Point<int8_t>> Points(iFile);
      PointRange<Euclidian_Point<int8_t>> Query_Points(qFile);
      Graph<unsigned int> G;
      if(gFile == NULL) G = Graph<unsigned int>(maxDeg, (size_t)(Points.size() * node_num_scale), use_madvise);
      else G = Graph<unsigned int>(gFile, use_madvise);
      timeNeighbors<Euclidian_Point<int8_t>, PointRange<Euclidian_Point<int8_t>>, uint>(G, Query_Points, k, BP,
        oFile, GT, metaFile, rFile, graph_built, Points);
    } else if(df == "mips"){
      PointRange<Mips_Point<int8_t>> Points(iFile);
      PointRange<Mips_Point<int8_t>> Query_Points(qFile);
      Graph<unsigned int> G;
      if(gFile == NULL) G = Graph<unsigned int>(maxDeg, (size_t)(Points.size() * node_num_scale), use_madvise);
      else G = Graph<unsigned int>(gFile, use_madvise);
      timeNeighbors<Mips_Point<int8_t>, PointRange<Mips_Point<int8_t>>, uint>(G, Query_Points, k, BP,
        oFile, GT, metaFile, rFile, graph_built, Points);
    }
  }
  
  return 0;
}


