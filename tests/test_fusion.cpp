// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Validate fused kernels against an independent double scalar calculation,
// including strided GLU views and intermediates explicitly retained as outputs.
int main(int argc, char ** argv) {
    try {
        ggml_backend_load_all();
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
            ggml_backend_init_by_name(argc>1 ? argv[1] : "CPU",nullptr),ggml_backend_free);
        if (!backend) throw std::runtime_error("Backend unavailable");
        std::mt19937 rng(62197);
        std::uniform_real_distribution<float> random(-2,2);
        int cases=0;
        double worst=0;
        for (int mode=0;mode<6;++mode) for (int cols : {7,32,360,512,1024})
        for (int rows : {1,3,65}) for (bool exposed : {false,true}) {
            std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
                ggml_init({64*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true}),ggml_free);
            const int input_cols=mode==2 ? 2*cols : cols;
            auto * input=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,input_cols,rows);
            auto * weight=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,cols);
            auto * bias=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,cols);
            ggml_tensor * intermediate=nullptr; ggml_tensor * result=nullptr;
            if (mode==0) {
                intermediate=ggml_norm(ctx.get(),input,1e-5f);
                result=ggml_add(ctx.get(),ggml_mul(ctx.get(),intermediate,weight),bias);
            } else if (mode==1) {
                intermediate=ggml_mul(ctx.get(),input,weight);
                result=ggml_add(ctx.get(),intermediate,bias);
            } else if (mode==2) {
                auto * a=ggml_view_2d(ctx.get(),input,cols,rows,input->nb[1],0);
                auto * b=ggml_view_2d(ctx.get(),input,cols,rows,input->nb[1],cols*4);
                intermediate=ggml_sigmoid(ctx.get(),b);
                result=ggml_mul(ctx.get(),a,intermediate);
            } else {
                intermediate=ggml_add(ctx.get(),input,bias);
                result=mode==3 ? ggml_sigmoid(ctx.get(),intermediate) :
                    mode==4 ? ggml_silu(ctx.get(),intermediate) : ggml_leaky_relu(ctx.get(),intermediate,0.01f,false);
            }
            if (exposed) ggml_set_output(intermediate);
            ggml_set_output(result);
            auto * graph=ggml_new_graph(ctx.get());
            ggml_build_forward_expand(graph,result);
            std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(
                ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);
            if (!buffer) throw std::runtime_error("Allocation failed");
            std::vector<float> x(input_cols*rows),w(cols),b(cols),out(cols*rows),mid(cols*rows);
            for(auto &v:x) v=random(rng);
            for(auto &v:w) v=random(rng);
            for(auto &v:b) v=random(rng);
            ggml_backend_tensor_set(input,x.data(),0,x.size()*4);
            ggml_backend_tensor_set(weight,w.data(),0,w.size()*4);
            ggml_backend_tensor_set(bias,b.data(),0,b.size()*4);
            if(ggml_backend_graph_compute(backend.get(),graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("Compute failed");
            ggml_backend_tensor_get(result,out.data(),0,out.size()*4);
            if(exposed) ggml_backend_tensor_get(intermediate,mid.data(),0,mid.size()*4);
            for(int row=0;row<rows;++row) {
                double mean=0,var=0;
                if(mode==0) {
                    for(int j=0;j<cols;++j) mean+=x[row*cols+j];
                    mean/=cols;
                    for(int j=0;j<cols;++j) {double d=x[row*cols+j]-mean;var+=d*d;}
                    var/=cols;
                }
                for(int j=0;j<cols;++j) {
                    double value=x[row*input_cols+j],tmp=0,expected=0;
                    if(mode==0) {tmp=(value-mean)/std::sqrt(var+1e-5);expected=tmp*w[j]+b[j];}
                    else if(mode==1) {tmp=value*w[j];expected=tmp+b[j];}
                    else if(mode==2) {tmp=1/(1+std::exp(-double(x[row*input_cols+cols+j])));expected=value*tmp;}
                    else {tmp=value+b[j];expected=mode==3?1/(1+std::exp(-tmp)):mode==4?tmp/(1+std::exp(-tmp)):tmp>=0?tmp:tmp*0.01;}
                    auto check=[&](float got,double ref) {
                        double err=std::abs(got-ref);worst=std::max(worst,err);
                        if(!std::isfinite(got)||err>2e-5+2e-6*std::abs(ref)) throw std::runtime_error("Fusion mismatch mode="+std::to_string(mode));
                    };
                    check(out[row*cols+j],expected);
                    if(exposed) check(mid[row*cols+j],tmp);
                }
            }
            ++cases;
        }
        std::cout<<"PASS: "<<cases<<" fused/fallback cases, max error="<<worst<<'\n';
    } catch(const std::exception & e) {std::cerr<<e.what()<<'\n';return 1;}
}
