// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

using Context=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
Context context() { return Context(ggml_init({64*ggml_tensor_overhead()+ggml_graph_overhead(),nullptr,true}),ggml_free); }
struct Execution {
    ggml_backend_t backend;
    ggml_cgraph * graph;
    Buffer buffer;
    Execution(ggml_context * ctx, ggml_backend_t backend, ggml_tensor * output):backend(backend),
        graph(ggml_new_graph(ctx)),buffer(nullptr,ggml_backend_buffer_free) {
        ggml_set_output(output);
        ggml_build_forward_expand(graph,output);
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx,backend));
        if(!buffer) throw std::runtime_error("Allocation failed");
    }
    std::vector<float> run(ggml_tensor * result) {
        if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) throw std::runtime_error("Compute failed");
        std::vector<float> out(ggml_nelements(result));
        ggml_backend_tensor_get(result,out.data(),0,out.size()*4);
        return out;
    }
};
void upload(ggml_tensor * t,const std::vector<float> & v) {ggml_backend_tensor_set(t,v.data(),0,v.size()*4);}
void check(float got,double expected,double tolerance,const char * label) {
    if(!std::isfinite(got)||std::abs(got-expected)>tolerance) {
        std::cerr<<label<<" got="<<got<<" expected="<<expected<<" error="<<std::abs(got-expected)<<'\n';
        throw std::runtime_error(label);
    }
}
int main(int argc,char ** argv) {
    try {
        ggml_backend_load_all();
        std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(
            ggml_backend_init_by_name(argc>1?argv[1]:"CPU",nullptr),ggml_backend_free);
        if(!backend) throw std::runtime_error("Backend unavailable");
        std::mt19937 rng(120906);std::uniform_real_distribution<float> rnd(-1,1);
        int copies=0,convs=0,norms=0,glus=0,cols=0;
        for(int cols:{1,7,31,32,33,512}) for(int rows:{1,3,31,33,101}) for(bool transpose:{false,true}) {
            auto ctx=context();const int stride=(transpose?rows:cols)+3;
            auto * storage=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,stride,transpose?cols:rows);
            auto * view=ggml_view_2d(ctx.get(),storage,transpose?rows:cols,transpose?cols:rows,stride*4,4);
            if(transpose) view=ggml_transpose(ctx.get(),view);
            auto * out=ggml_cont(ctx.get(),view);Execution e(ctx.get(),backend.get(),out);
            std::vector<float> x(ggml_nelements(storage));for(auto &v:x)v=rnd(rng);
            upload(storage,x);auto y=e.run(out);
            for(int row=0;row<rows;++row) for(int col=0;col<cols;++col)
                if(y[row*cols+col]!=x[1+(transpose?col*stride+row:row*stride+col)]) throw std::runtime_error("Copy bit mismatch");
            ++copies;
        }
        for(int frames:{1,2,15,16,31,32,63,64,255,256,257,1101}) for(int channels:{1,3,32}) for(int batches:{1,2}) for(bool channel_layout:{false,true}) {
            auto ctx=context();auto * w=ggml_new_tensor_4d(ctx.get(),GGML_TYPE_F32,31,1,1,channels);
            auto * storage=ggml_new_tensor_4d(ctx.get(),GGML_TYPE_F32,channel_layout?channels:frames,channel_layout?frames:1,channel_layout?1:channels,batches);
            auto * x=channel_layout?ggml_permute(ctx.get(),storage,2,0,1,3):storage;
            auto * out=ggml_conv_2d_dw_direct(ctx.get(),w,x,1,1,15,0,1,1);
            if(channel_layout) {
                out->nb[0]=channels*4;out->nb[1]=channels*frames*4;out->nb[2]=4;out->nb[3]=channels*frames*4;
                out=ggml_permute(ctx.get(),out,1,2,0,3);
            }
            Execution e(ctx.get(),backend.get(),out);
            std::vector<float> a(31*channels),b(frames*channels*batches);for(auto &v:a)v=rnd(rng);for(auto &v:b)v=rnd(rng);
            upload(w,a);upload(storage,b);auto y=e.run(out);
            for(int n=0;n<batches;++n) for(int c=0;c<channels;++c) for(int t=0;t<frames;++t) {
                double ref=0;for(int k=0;k<31;++k) if(t+k-15>=0 && t+k-15<frames) ref+=double(a[c*31+k])*b[channel_layout?(n*frames+t+k-15)*channels+c:(n*channels+c)*frames+t+k-15];
                check(y[channel_layout?(n*frames+t)*channels+c:(n*channels+c)*frames+t],ref,1e-5,"Depthwise mismatch");
            }
            ++convs;
        }
        for(int frames:{1,31,32,33,101,1101}) for(int channels:{8,32,512}) for(int groups:{1,4}) for(int mode:{0,1,2}) {
            auto ctx=context();auto * x=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,frames,1,channels);
            auto * out=ggml_group_norm(ctx.get(),x,groups,1e-5f);Execution e(ctx.get(),backend.get(),out);
            std::vector<float> a(frames*channels);for(auto &v:a)v=mode==0?rnd(rng):mode==1?4.0f+rnd(rng)*0.02f:4.0f;
            upload(x,a);auto y=e.run(out);int count=frames*channels/groups;
            for(int g=0;g<groups;++g) {
                double mean=0,var=0;for(int i=0;i<count;++i)mean+=a[g*count+i];mean/=count;
                for(int i=0;i<count;++i){double d=a[g*count+i]-mean;var+=d*d;}var/=count;
                for(int i=0;i<count;++i)check(y[g*count+i],(a[g*count+i]-mean)/std::sqrt(var+1e-5),mode==1?1e-4:2e-5,"GroupNorm mismatch");
            }
            auto again=e.run(out);if(y!=again)throw std::runtime_error("GroupNorm nondeterminism");
            ++norms;
        }
        for(int frames:{1,2,31,32,33,101,1101}) for(int channels:{1,7,32,33,128}) {
            auto ctx=context();auto * x=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,frames,channels);
            auto * w=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,3,channels,8);
            auto * out=ggml_im2col(ctx.get(),w,x,1,0,1,0,1,0,false,GGML_TYPE_F32);
            Execution e(ctx.get(),backend.get(),out);
            std::vector<float> input(frames*channels);for(auto &v:input)v=rnd(rng);
            upload(x,input);auto y=e.run(out);
            for(int t=0;t<frames;++t)for(int c=0;c<channels;++c)for(int k=0;k<3;++k) {
                float expected=t+k-1>=0&&t+k-1<frames?input[c*frames+t+k-1]:0;
                if(y[t*channels*3+c*3+k]!=expected)throw std::runtime_error("im2col bit mismatch");
            }
            ++cols;
        }
        for(int hidden:{32,64}) for(int rows:{7,11,33,65}) for(int k:{64,512}) for(bool exposed:{false,true}) {
            auto ctx=context();
            auto * w=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,k,hidden*2);
            auto * x=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,k,rows);
            auto * bias=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,hidden*2);
            auto * mm=ggml_mul_mat(ctx.get(),w,x);ggml_mul_mat_set_prec(mm,GGML_PREC_F32);
            auto * add=ggml_add(ctx.get(),mm,bias);
            auto * a=ggml_view_2d(ctx.get(),add,hidden,rows,add->nb[1],0);
            auto * b=ggml_view_2d(ctx.get(),add,hidden,rows,add->nb[1],hidden*4);
            auto * out=ggml_mul(ctx.get(),a,ggml_sigmoid(ctx.get(),b));
            if(exposed)ggml_set_output(add);
            Execution e(ctx.get(),backend.get(),out);
            std::vector<float> weights(k*hidden*2),input(k*rows),biases(hidden*2);
            for(auto &v:weights)v=rnd(rng)*0.25f;for(auto &v:input)v=rnd(rng)*0.25f;for(auto &v:biases)v=rnd(rng);
            upload(w,weights);upload(x,input);upload(bias,biases);auto y=e.run(out);
            std::vector<float> mid(rows*hidden*2);
            if(exposed)ggml_backend_tensor_get(add,mid.data(),0,mid.size()*4);
            for(int r=0;r<rows;++r)for(int h=0;h<hidden;++h) {
                double values[2]={biases[h],biases[h+hidden]};
                for(int q=0;q<k;++q)for(int gate=0;gate<2;++gate)values[gate]+=double(weights[(h+gate*hidden)*k+q])*input[r*k+q];
                check(y[r*hidden+h],values[0]/(1+std::exp(-values[1])),1e-5,"Projection GLU mismatch");
                if(exposed)for(int gate=0;gate<2;++gate)check(mid[r*hidden*2+h+gate*hidden],values[gate],1e-5,"Exposed projection mismatch");
            }
            ++glus;
        }
        std::cout<<"PASS: "<<copies<<" bitwise copies, "<<convs<<" depthwise, "<<norms<<" GroupNorm, "<<glus<<" projection GLU, "<<cols<<" bitwise im2col cases\n";
    }catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}
}
