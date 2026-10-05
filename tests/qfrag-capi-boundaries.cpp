#include "mlx/mlx.h"
#include "mlx/c/private/mlx.h"
#include "mlx/c/fast.h"
#include <iostream>
#include <fstream>
#include <iterator>
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
namespace mx=mlx::core;
std::string read(const std::string&p){std::ifstream f(p);return std::string(std::istreambuf_iterator<char>(f),{});}
int main(int argc,char**argv){
 if(argc!=3)return 2;Dl_info loaded{};if(!dladdr((void*)mlx_fast_metal_kernel_new,&loaded)||std::filesystem::canonical(loaded.dli_fname)!=std::filesystem::canonical(argv[2])) {std::cerr<<"Incorrect CAPI loaded"<<std::endl;return 5;}std::string dir=argv[1];auto g=mx::default_stream(mx::Device(mx::Device::gpu));auto cs=mlx_stream_new_(g);mx::set_default_device(g.device);
 struct Case{int S,K,packed,strided,gqa,ratio;bool timed;};int index=0;
 for(auto c:{Case{16,19,8,0,12,4,false},Case{17,33,8,1,12,4,false},Case{64,69,8,0,12,4,false},Case{64,69,0,1,12,4,false},{65,8193,8,1,12,4,false},{1,8192,8,0,12,4,false},
             {33,8192,0,1,8,4,false},{17,33,0,0,12,1,false},{33,8192,4,1,12,4,false},
             {4096,8192,0,0,12,4,true},{4096,8192,8,1,12,4,true},
             {4096,16384,0,1,12,4,true},{4096,16384,8,0,12,4,true},
             {4096,32768,0,0,12,4,true},{4096,32768,8,1,12,4,true}}){
  int Hk=2,Hq=Hk*c.gqa,KB=512,NSG=(c.gqa+7)/8;auto q0=mx::random::normal({1,c.S,Hq,256},mx::bfloat16,mx::random::key(100+index));auto q=mx::transpose(q0,{0,2,1,3});
  auto k0=mx::random::normal(c.strided?mx::Shape{1,c.K,Hk,256}:mx::Shape{1,Hk,c.K,256},mx::bfloat16,mx::random::key(200+index));auto v0=mx::random::normal(k0.shape(),mx::bfloat16,mx::random::key(300+index));
  std::vector<mx::array> in;auto scl=mx::reshape(mx::array(.0625f),{1});std::vector<int> selection(c.S*KB,2147483647);
  for(int s=0;s<c.S;s++){int complete=(c.K-c.S+s+1)/c.ratio,n=std::min(complete,KB);for(int j=0;j<n;j++)selection[s*KB+j]=(j*complete)/n;}
  auto blocks=mx::array(selection.data(),{1,c.S,KB});
  if(c.packed){auto kq=mx::quantize(k0,64,c.packed),vq=mx::quantize(v0,64,c.packed);if(c.strided){for(auto&x:kq)x=mx::transpose(x,{0,2,1,3});for(auto&x:vq)x=mx::transpose(x,{0,2,1,3});}in={q,kq[0],vq[0],scl,blocks,kq[1],kq[2],vq[1],vq[2]};}
  else {auto k=c.strided?mx::transpose(k0,{0,2,1,3}):k0,v=c.strided?mx::transpose(v0,{0,2,1,3}):v0;in={q,k,v,scl,blocks};}
  mx::eval(in);std::vector<const char*> names={"q","k","v","scl","blocks"};if(c.packed)names.insert(names.end(),{"ksc","kbi","vsc","vbi"});const char* outs[]={"out"};auto iv=mlx_vector_string_new_data(names.data(),names.size()),ov=mlx_vector_string_new_data(outs,1);auto body=read(dir+"/qsa-stock-source.metal"),header=read(dir+(c.packed?"/qsa-stock-packed-header.metal":"/qsa-stock-header.metal"));
  mlx_fast_metal_kernel kernels[2];for(int flag=0;flag<2;flag++){setenv("SUSHI_QSA_QFRAG_BF16",flag?"1":"0",1);kernels[flag]=mlx_fast_metal_kernel_new(c.packed?"sushi_attn_qsa256_packed":"sushi_attn_qsa256",iv,ov,body.c_str(),header.c_str(),false,false);if(!kernels[flag].ctx)return 3;}
  auto cfg=mlx_fast_metal_kernel_config_new();int shape[]={1,Hq,c.S,256};
  mlx_fast_metal_kernel_config_add_output_arg(cfg,shape,4,MLX_BFLOAT16);mlx_fast_metal_kernel_config_set_grid(cfg,c.S*32,Hk*NSG,1);mlx_fast_metal_kernel_config_set_thread_group(cfg,32,NSG,1);mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg,"T",MLX_BFLOAT16);
  for(auto nv:{std::pair<const char*,int>{"NSG",NSG},{"BK",32},{"RATIO",c.ratio}})mlx_fast_metal_kernel_config_add_template_arg_int(cfg,nv.first,nv.second);if(c.packed){mlx_fast_metal_kernel_config_add_template_arg_int(cfg,"BITS",c.packed);mlx_fast_metal_kernel_config_add_template_arg_int(cfg,"GS",64);}
  auto input=mlx_vector_array_new_(in);auto create=[&](bool on){auto output=mlx_vector_array_new_();if(mlx_fast_metal_kernel_apply(&output,kernels[on?1:0],input,cfg,cs))throw std::runtime_error("QSA apply failed");auto y=mlx_vector_array_get_(output).at(0);mlx_vector_array_free_(output);return y;};
  auto a=create(false),b=create(true);mx::eval({a,b});bool exact=mx::all(mx::equal(mx::view(a,mx::uint16),mx::view(b,mx::uint16))).item<bool>();bool finite=mx::all(mx::isfinite(a)).item<bool>()&&mx::all(mx::isfinite(b)).item<bool>();
  std::cout<<"{\"kind\":\"numerical\",\"S\":"<<c.S<<",\"KV\":"<<c.K<<",\"packed_bits\":"<<c.packed<<",\"strided\":"<<c.strided<<",\"gqa\":"<<c.gqa<<",\"ratio\":"<<c.ratio<<",\"bit_exact\":"<<(exact?"true":"false")<<",\"finite\":"<<(finite?"true":"false")<<",\"admitted\":"<<((c.gqa==12&&c.ratio==4&&(c.packed==0||c.packed==8))?"true":"false")<<"}"<<std::endl;if(!exact||!finite)return 4;

  mlx_vector_array_free_(input);mlx_fast_metal_kernel_config_free(cfg);for(auto k:kernels)mlx_fast_metal_kernel_free(k);mlx_vector_string_free(iv);mlx_vector_string_free(ov);index++;
 }
 mlx_stream_free_(cs);return 0;
}
