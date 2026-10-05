#include "mlx/mlx.h"
#include "mlx/c/private/mlx.h"
#include "mlx/c/ops.h"
#include <chrono>
#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
namespace mx=mlx::core;
double stamp(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
int main(int argc,char**argv){if(argc!=2)return 2;int failures=0;Dl_info actual{};if(!dladdr(dlsym(RTLD_DEFAULT,"mlx_gather_qmm"),&actual))return 3;if(std::filesystem::canonical(actual.dli_fname)!=std::filesystem::canonical(argv[1]))return 4;std::cerr<<"[actual-CAPI] "<<actual.dli_fname<<std::endl;auto dev=mx::Device(mx::Device::gpu);mx::set_default_device(dev);auto cs=mlx_stream_new_(mx::default_stream(dev));for(int bits:{3,4})for(auto dims:{std::pair<int,int>{2560,640},{640,2560}}){int K=dims.first,N=dims.second,E=512;auto w=mx::random::bits({E,N,K*bits/32},mx::random::key(43+bits));auto sc=mx::random::uniform(.005f,.015f,{E,N,K/64},mx::bfloat16,mx::random::key(71+bits));auto bi=mx::random::uniform(-.03f,.03f,{E,N,K/64},mx::bfloat16,mx::random::key(99+bits));mx::eval({w,sc,bi});
for(int M:{4095,4096,4097,40959,40960,40961}){std::vector<uint32_t> ids(M);for(int i=0;i<M;i++)ids[i]=std::min(511,(i*511/M+(i*511/M)/7)%512);std::sort(ids.begin(),ids.end());auto idx=mx::array(ids.data(),{M},mx::uint32);auto x=mx::random::normal({M,1,K},mx::bfloat16,mx::random::key(120+bits));mx::eval({x,idx});auto xc=mlx_array_new_(x),wc=mlx_array_new_(w),sc_=mlx_array_new_(sc),bi_=mlx_array_new_(bi),ic=mlx_array_new_(idx);auto none=mlx_array{nullptr};mlx_optional_int gs{};gs.value=64;gs.has_value=true;mlx_optional_int bp{};bp.value=bits;bp.has_value=true;
auto build=[&](bool on){setenv("SUSHI_MOE_ALIGNED32_TAIL16",on?"1":"0",1);auto out=mlx_array_new_();if(mlx_gather_qmm(&out,xc,wc,sc_,bi_,none,ic,true,gs,bp,"affine",true,cs))throw std::runtime_error("gatherQMM failed");auto y=mlx_array_get_(out);mlx_array_free_(out);return y;};auto a=build(false),b=build(true);mx::eval({a,b});mx::synchronize();bool exact=mx::all(mx::equal(mx::view(a,mx::uint16),mx::view(b,mx::uint16))).item<bool>();if(!exact)failures++;float diff=mx::max(mx::abs(mx::subtract(mx::astype(a,mx::float32),mx::astype(b,mx::float32)))).item<float>();std::cout<<"{\"kind\":\"numerical\",\"bits\":"<<bits<<",\"M\":"<<M<<",\"N\":"<<N<<",\"K\":"<<K<<",\"bit_exact\":"<<(exact?"true":"false")<<",\"maxdiff\":"<<diff<<"}"<<std::endl;

for(auto v:{xc,wc,sc_,bi_,ic})mlx_array_free_(v);}}
mlx_stream_free_(cs);return failures?2:0;}
