#include "mlx/c/fast.h"
#include "mlx/c/vector.h"
#include "mlx/c/error.h"
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <iostream>
#include <string>
static int errors=0;
static void handle(const char*,void*) { ++errors; }
int main(int argc,char**argv) {
  if(argc!=2)return 2;
  if(argc!=2)return 2;
  std::ifstream f(argv[1]); std::string body{std::istreambuf_iterator<char>(f),{}};
  const char* inputs[]={"x","w","sc","bi","K_size","N_size"};const char* outputs[]={"y"};
  auto in=mlx_vector_string_new_data(inputs,6),out=mlx_vector_string_new_data(outputs,1);
  mlx_set_error_handler(handle,nullptr,nullptr);
  struct Case {const char* label;bool on;const char* name;bool changed;const char* header;bool contiguous;bool atomic;bool expected;};
  Case cases[]={
    {"OFF-standard",false,"mtp_serial_qmv8_rows",false,"",true,false,true},
    {"ON-strict-match",true,"mtp_serial_qmv8_rows",false,"",true,false,true},
    {"ON-body-changed",true,"mtp_serial_qmv8_rows",true,"",true,false,false},
    {"ON-header-changed",true,"mtp_serial_qmv8_rows",false,"//changed",true,false,false},
    {"ON-contiguous-changed",true,"mtp_serial_qmv8_rows",false,"",false,false,false},
    {"ON-atomic-changed",true,"mtp_serial_qmv8_rows",false,"",true,true,false},
    {"OFF-unsubstituted-body",false,"mtp_serial_qmv8_rows",true,"",true,false,true},
    {"ON-unmatched-name-standard",true,"other_kernel",true,"",true,false,true}
  };
  for(const auto& c:cases) {
    setenv("SUSHI_MTP_QMV8_SG2",c.on?"1":"0",1);errors=0;
    const auto source=body+(c.changed?"\n":"");
    auto k=mlx_fast_metal_kernel_new(c.name,in,out,source.c_str(),c.header,c.contiguous,c.atomic);
    const bool valid=k.ctx!=nullptr;
    const bool pass=valid==c.expected && errors==(c.expected?0:1);
    std::cout<<"{\"case\":\""<<c.label<<"\",\"factory_guard_pass\":"<<(pass?"true":"false")<<",\"GPUeval\":false}"<<std::endl;
    if(valid)mlx_fast_metal_kernel_free(k);
    if(!pass)return 3;
  }
  mlx_vector_string_free(in);mlx_vector_string_free(out);return 0;
}
