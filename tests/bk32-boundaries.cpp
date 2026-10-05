#include "mlx/mlx.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
namespace mx=mlx::core;
int main(){
 mx::set_default_device(mx::Device(mx::Device::gpu));
 int count=0;
 for(auto nk:{std::pair<int,int>{512,2560},{640,2560},{2560,2560},{6144,2560},{10240,2560},{12288,2560},{2560,6144},{2560,640},{10240,320}}){
  int N=nk.first,K=nk.second;
  for(int M:{2047,2048,4001,4008,4095,4096,16384,16385}){
   auto x=mx::random::normal({M,K},mx::bfloat16,mx::random::key(200+count)),w=mx::random::normal({N,K},mx::bfloat16,mx::random::key(300+count));count++;
   auto wt=mx::transpose(w);mx::eval({x,w});
   auto run=[&](bool on){setenv("SUSHI_GEMM_PREFILL_BK32",on?"1":"0",1);auto y=mx::matmul(x,wt);mx::eval(y);mx::synchronize();return y;};
   auto a=run(false),b=run(true);auto eq=mx::all(mx::equal(a,b));auto dif=mx::max(mx::abs(mx::subtract(mx::astype(a,mx::float32),mx::astype(b,mx::float32))));mx::eval({eq,dif});bool exact=eq.item<bool>();float maxabs=dif.item<float>();
   std::cout<<"{\"kind\":\"numerical\",\"M\":"<<M<<",\"N\":"<<N<<",\"K\":"<<K<<",\"exact\":"<<(exact?"true":"false")<<",\"max_abs\":"<<maxabs<<"}"<<std::endl;if(!exact)return 3;

  }
 }
}
