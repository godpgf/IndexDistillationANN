#include <iostream>
#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/io.h"


auto convert_uvec(const char* infile, const char* outfile) {
  auto str = parlay::chars_from_file(infile);
  std::ofstream writer;
  writer.open(outfile, std::ios::binary | std::ios::out);


  int dims = *((int *) str.data());
  int n = str.size()/(4*dims+4);
  std::cout<< "n = " << n << " d = " << dims << std::endl;
  auto vects = parlay::tabulate(n, [&] (size_t i) {
		return parlay::to_sequence(str.cut(4 + i * (4 + 4*dims), (i+1) * (4 + 4*dims)));});
  auto flat_vects = parlay::flatten(vects);
  std::cout<<flat_vects.size()<<std::endl;
  char vec[4];
  float min_f = 10000000;
  float max_f = -10000000;
  for(int i = 0; i < n; ++i){
    writer.write((char*)&dims, 4);    
    for(int j = 0; j < dims; ++j){
      for(int k = 0; k < 4; ++k) vec[k] = flat_vects[i * dims * 4 + j * 4 + k];
      float v = *(float*)vec;
      unsigned char iv = (unsigned char)v;
      writer.write((char*)&iv, 1);
      if(min_f > iv) min_f = iv;
      if(max_f < iv) max_f = iv; 
    }
    
  }
  writer.close();
  std::cout<<min_f<<" "<<max_f<<std::endl;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "usage: fvec_to_uvec <infile> <outfile>" << std::endl;
    return 1;
  }
  convert_uvec(argv[1], argv[2]);
  return 0;
}
