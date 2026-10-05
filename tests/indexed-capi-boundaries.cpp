#include "mlx/mlx.h"
#include "mlx/c/private/mlx.h"
#include "mlx/c/ops.h"
#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
namespace mx=mlx::core;
double stamp(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
int main(int argc,char**argv){if(argc!=2)return 2;int failures=0;Dl_info info{};if(!dladdr(dlsym(RTLD_DEFAULT,"mlx_gather_qmm"),&info))return 3;std::fprintf(stderr,"[actual-CAPI] %s\n",info.dli_fname);if(std::filesystem::canonical(info.dli_fname)!=std::filesystem::canonical(argv[1]))return 4;mx::set_default_device(mx::Device(mx::Device::gpu));auto st=mx::default_stream(mx::Device(mx::Device::gpu));auto cs=mlx_stream_new_(st);setenv("SUSHI_MOE_ALIGNED32_TAIL16","1",1);std::mt19937 rng(19841);
for(auto dtype:{mx::uint32,mx::int32,mx::uint64,mx::int64})for(int tokens:{409,410,4096,4097})for(int skew:{0,1}){const int M=tokens*10,K=2560,N=640,E=512;std::vector<uint32_t> ids(M),order(M),lhs(M),expert(M);for(int t=0;t<tokens;t++){bool used[512]={};for(int k=0;k<10;k++){uint32_t e;if(skew&&k==0)e=0;else{do{e=1+rng()%511;}while(used[e]);}used[e]=true;ids[t*10+k]=e;}}std::iota(order.begin(),order.end(),0);std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return ids[a]<ids[b];});for(int i=0;i<M;i++){lhs[i]=order[i]/10;expert[i]=ids[order[i]];}auto X=mx::random::normal({tokens,K},mx::bfloat16,mx::random::key(51+skew));auto L=mx::astype(mx::array(lhs.data(),{M},mx::uint32),dtype);if(dtype==mx::int32||dtype==mx::int64)L=mx::where(mx::equal(mx::remainder(mx::arange(M),mx::array(2)),mx::array(0)),mx::subtract(L,mx::array(tokens,dtype)),L);auto I=mx::array(expert.data(),{M},mx::uint32);mx::eval({X,L,I});auto copy=[&](){return mx::take(X,L,0,st);};mx::eval(copy());mx::synchronize();

for(int bits:{3,4}){auto W=mx::random::bits({E,N,K*bits/32},mx::random::key(43+bits)),U=mx::random::bits({E,N,K*bits/32},mx::random::key(49+bits));auto S=mx::random::uniform(.005f,.015f,{E,N,K/64},mx::bfloat16,mx::random::key(71+bits)),B=mx::random::uniform(-.03f,.03f,{E,N,K/64},mx::bfloat16,mx::random::key(99+bits));mx::eval({W,U,S,B});auto ic=mlx_array_new_(I),wc=mlx_array_new_(W),uc=mlx_array_new_(U),sc=mlx_array_new_(S),bc=mlx_array_new_(B);auto none=mlx_array{nullptr};mlx_optional_int gs{};gs.value=64;gs.has_value=true;mlx_optional_int bp{};bp.value=bits;bp.has_value=true;
auto build=[&](bool indexed){setenv("SUSHI_MOE_INDEXED_INPUT",indexed?"1":"0",1);auto x=mx::reshape(copy(),{M,1,K});auto xc=mlx_array_new_(x);std::vector<mx::array> out;for(auto w:{wc,uc}){auto y=mlx_array_new_();if(mlx_gather_qmm(&y,xc,w,sc,bc,none,ic,true,gs,bp,"affine",true,cs))throw std::runtime_error("gatherQMM");out.push_back(mlx_array_get_(y));mlx_array_free_(y);}mlx_array_free_(xc);return out;};auto aa=build(false),bb=build(true);std::vector<mx::array> both=aa;both.insert(both.end(),bb.begin(),bb.end());mx::eval(both);mx::synchronize();for(int j=0;j<2;j++){bool exact=mx::all(mx::equal(mx::view(aa[j],mx::uint16),mx::view(bb[j],mx::uint16))).item<bool>();float diff=mx::max(mx::abs(mx::subtract(mx::astype(aa[j],mx::float32),mx::astype(bb[j],mx::float32)))).item<float>();if(!exact)failures++;std::cout<<"{\"kind\":\"numerical\",\"tokens\":"<<tokens<<",\"index_type\":\""<<(dtype==mx::uint32?"uint32":dtype==mx::int32?"int32":dtype==mx::uint64?"uint64":"int64")<<"\",\"skew\":"<<skew<<",\"bits\":"<<bits<<",\"projection\":"<<j<<",\"bit_exact\":"<<(exact?"true":"false")<<",\"maxdiff\":"<<diff<<"}"<<std::endl;}for(auto a:{ic,wc,uc,sc,bc})mlx_array_free_(a);}}
mlx_stream_free_(cs);return failures?2:0;}
