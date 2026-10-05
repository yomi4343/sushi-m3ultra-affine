#include "mlx/mlx.h"
#include "mlx/c/fast.h"
#include "mlx/c/stream.h"
#include "mlx/c/private/mlx.h"
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
namespace mx=mlx::core;
std::string read(const std::string&p){std::ifstream f(p);return std::string(std::istreambuf_iterator<char>(f),{});}
int main(int argc,char**argv){if(argc!=2)return 2;
 std::string dir=argv[1];mx::set_default_device(mx::Device(mx::Device::gpu));
 const char* in_names[]={"x","w","sc","bi","K_size","N_size"};const char* out_names[]={"y"};
 auto ivn=mlx_vector_string_new_data(in_names,6),ovn=mlx_vector_string_new_data(out_names,1);
 auto body=read(dir+"/body.metal");
 setenv("SUSHI_MTP_QMV8_SG2","0",1);auto base=mlx_fast_metal_kernel_new("mtp_serial_qmv8_rows",ivn,ovn,body.c_str(),"",true,false);
 setenv("SUSHI_MTP_QMV8_SG2","1",1);auto candidate=mlx_fast_metal_kernel_new("mtp_serial_qmv8_rows",ivn,ovn,body.c_str(),"",true,false);
 if(!base.ctx || !candidate.ctx)return 4;
 auto stream=mlx_default_gpu_stream_new();
 for(auto nk:{std::pair<int,int>{320,10240},{10240,320},{10240,2560},{6144,2560},{2560,6144},{512,2560},{640,2560},{2560,640},{12288,2560}}){
  int N=nk.first,K=nk.second;
  auto w=mx::random::bits({N,K/4},mx::random::key(N+K));
  auto sc=mx::random::uniform(.002f,.006f,{N,K/64},mx::bfloat16,mx::random::key(N+1));
  auto bi=mx::random::uniform(-.03f,-.01f,{N,K/64},mx::bfloat16,mx::random::key(N+2));mx::eval({w,sc,bi});
  for(int M:{1,2,3,4,5})for(int rank:{2,3,4,8}){
   mx::Shape input_shape(rank,1);input_shape[rank-2]=M;input_shape[rank-1]=K;auto x=mx::random::normal(input_shape,mx::bfloat16,mx::random::key(M+3));auto ka=mx::array(K),na=mx::array(N);mx::eval({x,ka,na});
   auto cfg=mlx_fast_metal_kernel_config_new();auto shape=input_shape;shape[rank-1]=N;
   if(mlx_fast_metal_kernel_config_add_output_arg(cfg,shape.data(),shape.size(),MLX_BFLOAT16))return 5;
   mlx_fast_metal_kernel_config_set_grid(cfg,32,(N/8)*2,1);mlx_fast_metal_kernel_config_set_thread_group(cfg,32,2,1);
   mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg,"T",MLX_BFLOAT16);
   for(auto p:{std::pair<const char*,int>{"M",M},{"GS",64},{"NV",M},{"FAST",K%256==0?1:0}})mlx_fast_metal_kernel_config_add_template_arg_int(cfg,p.first,p.second);
   auto inputs=mlx_vector_array_new_(std::vector<mx::array>{x,w,sc,bi,ka,na});
   auto create=[&](bool on){auto outputs=mlx_vector_array_new_();if(mlx_fast_metal_kernel_apply(&outputs,on?candidate:base,inputs,cfg,stream))throw std::runtime_error("CAPIapplyfailed");auto y=mlx_vector_array_get_(outputs)[0];mlx_vector_array_free(outputs);return y;};
   auto run=[&](bool on){auto y=create(on);mx::eval(y);mx::synchronize();return y;};
   auto a=run(false),b=run(true);bool eq=mx::all(mx::equal(a,b)).item<bool>();
   std::cout<<"{\"kind\":\"numerical\",\"N\":"<<N<<",\"K\":"<<K<<",\"M\":"<<M<<",\"rank\":"<<rank<<",\"exact\":"<<(eq?"true":"false")<<"}"<<std::endl;if(!eq)return 3;
   mlx_vector_array_free(inputs);mlx_fast_metal_kernel_config_free(cfg);
  }
 }
}
