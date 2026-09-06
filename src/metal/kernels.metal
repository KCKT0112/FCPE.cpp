// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.

kernel void fcpe_transpose_f32(constant ulong4 & a [[buffer(0)]],
        device const float * src [[buffer(1)]], device float * dst [[buffer(2)]],
        uint2 group [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]]) {
    threadgroup float tile[32][33];
    const uint x = group.x*32, y = group.y*32;
    for (uint j=tid.y; j<32; j+=8)
        if (x+j<a.x && y+tid.x<a.y) tile[j][tid.x]=src[(x+j)*a.z+(y+tid.x)];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j=tid.y; j<32; j+=8)
        if (x+tid.x<a.x && y+j<a.y) dst[(y+j)*a.x+x+tid.x]=tile[tid.x][j];
}

kernel void fcpe_copy_rows_f32(constant ulong4 & a [[buffer(0)]],
        device const float * src [[buffer(1)]], device float * dst [[buffer(2)]],
        uint i [[thread_position_in_grid]]) {
    if (i<a.x*a.y) dst[i]=src[(i/a.x)*a.w+i%a.x];
}

kernel void fcpe_im2col3(constant uint2 & a [[buffer(0)]],
        device const float * x [[buffer(1)]],device float * out [[buffer(2)]],
        uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float tile[32][35];
    uint t0=group.x*32,c0=group.y*32;
    for(uint i=tid;i<32*34;i+=256) {
        uint c=i/34,j=i%34;int t=int(t0)+int(j)-1;
        tile[c][j]=(c0+c<a.y && t>=0 && t<int(a.x))?x[(c0+c)*a.x+uint(t)]:0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<32*32*3;i+=256) {
        uint t=i/96,c=(i%96)/3,k=i%3;
        if(t0+t<a.x && c0+c<a.y)out[(t0+t)*a.y*3+(c0+c)*3+k]=tile[c][t+k];
    }
}

struct fcpe_elementwise_args { uint count, cols, astride, bstride, mode; float alpha; };
kernel void fcpe_elementwise_fused(constant fcpe_elementwise_args & a [[buffer(0)]],
        device const float * x [[buffer(1)]], device const float * y [[buffer(2)]],
        device float * out [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i>=a.count) return;
    uint row=i/a.cols, col=i%a.cols;
    float v=x[row*a.astride+col];
    if (a.mode==0) { out[i]=y[row*a.bstride+col]/(1.0f+exp(-v)); return; }
    v+=y[col];
    out[i]=a.mode==1 ? 1.0f/(1.0f+exp(-v)) : a.mode==2 ? v/(1.0f+exp(-v)) : v>=0 ? v : a.alpha*v;
}

kernel void fcpe_depthwise31(constant uint2 & a [[buffer(0)]],
        device const float * w [[buffer(1)]], device const float * x [[buffer(2)]],
        device float * out [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float samples[286];
    uint channel=group.y%a.y;
    int start=int(group.x)*256-15;
    for (uint i=tid;i<286;i+=64) {
        int t=start+int(i);
        samples[i]=(t>=0 && t<int(a.x)) ? x[group.y*a.x+uint(t)] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 sum=0.0f;
    #pragma unroll
    for (uint k=0;k<31;++k) {
        uint base=tid*4+k;
        sum=fma(float4(samples[base],samples[base+1],samples[base+2],samples[base+3]),w[channel*31+k],sum);
    }
    uint t=group.x*256+tid*4;
    for(uint j=0;j<4;++j) if(t+j<a.x) out[group.y*a.x+t+j]=sum[j];
}

struct fcpe_group_args { uint count,chunks,groups,phase; float eps; };
inline float fcpe_reduce(float v, threadgroup float * shared, uint tid) {
    v=simd_sum(v);
    if ((tid&31)==0) shared[tid/32]=v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=tid<8 ? shared[tid] : 0.0f;
    total=simd_sum(total);
    if (tid==0) shared[8]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return shared[8];
}
kernel void fcpe_group_stats(constant fcpe_group_args & a [[buffer(0)]],
        device const float * x [[buffer(1)]], device float * tmp [[buffer(2)]],
        device float * out [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shared[9];
    float mean=0;
    if (a.phase==1) {
        float sum=0;
        for(uint i=tid;i<a.chunks;i+=256) sum+=tmp[group.y*a.chunks+i];
        mean=precise::divide(fcpe_reduce(sum,shared,tid),float(a.count));
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum=0;
    for(uint i=group.x*1024+tid;i<min((group.x+1)*1024,a.count);i+=256) {
        float v=x[group.y*a.count+i];
        if(a.phase==1) {v-=mean;v*=v;}
        sum+=v;
    }
    sum=fcpe_reduce(sum,shared,tid);
    if(tid==0) tmp[(a.phase*a.groups+group.y)*a.chunks+group.x]=sum;
}
kernel void fcpe_group_apply(constant fcpe_group_args & a [[buffer(0)]],
        device const float * x [[buffer(1)]], device const float * tmp [[buffer(2)]],
        device float * out [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shared[9];
    float sum=0,var=0;
    for(uint i=tid;i<a.chunks;i+=256) {
        sum+=tmp[group.y*a.chunks+i];
        var+=tmp[(a.groups+group.y)*a.chunks+i];
    }
    float mean=precise::divide(fcpe_reduce(sum,shared,tid),float(a.count));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale=rsqrt(precise::divide(fcpe_reduce(var,shared,tid),float(a.count))+a.eps);
    for(uint i=group.x*1024+tid;i<min((group.x+1)*1024,a.count);i+=256)
        out[group.y*a.count+i]=(x[group.y*a.count+i]-mean)*scale;
}

kernel void fcpe_group_small(constant fcpe_group_args & a [[buffer(0)]],
        device const float * x [[buffer(1)]], device float * out [[buffer(2)]],
        uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shared[9];
    float sum=0;
    for(uint i=tid;i<a.count;i+=256) sum+=x[group*a.count+i];
    float mean=precise::divide(fcpe_reduce(sum,shared,tid),float(a.count));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var=0;
    for(uint i=tid;i<a.count;i+=256) {float d=x[group*a.count+i]-mean;var+=d*d;}
    float scale=rsqrt(precise::divide(fcpe_reduce(var,shared,tid),float(a.count))+a.eps);
    for(uint i=tid;i<a.count;i+=256) out[group*a.count+i]=(x[group*a.count+i]-mean)*scale;
}

kernel void fcpe_depthwise_channels(constant uint2 & a [[buffer(0)]],
        device const float * w [[buffer(1)]],device const float * x [[buffer(2)]],
        device float * out [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sample[38][32];
    threadgroup float weight[31][32];
    uint t0=group.x*8,c0=group.y*32,c=tid%32;
    for(uint i=tid;i<38*32;i+=128) {
        int t=int(t0)+int(i/32)-15;uint ch=c0+i%32;
        sample[i/32][i%32]=(t>=0&&t<int(a.x)&&ch<a.y)?x[uint(t)*a.y+ch]:0.0f;
    }
    for(uint i=tid;i<31*32;i+=128) {
        uint ch=c0+i%32;
        weight[i/32][i%32]=ch<a.y?w[ch*31+i/32]:0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint t=tid/32;float2 sum=0;
    #pragma unroll
    for(uint k=0;k<31;++k)sum=fma(float2(sample[t+k][c],sample[t+4+k][c]),weight[k][c],sum);
    if(c0+c<a.y) {
        if(t0+t<a.x)out[(t0+t)*a.y+c0+c]=sum.x;
        if(t0+t+4<a.x)out[(t0+t+4)*a.y+c0+c]=sum.y;
    }
}

template<uint BM,uint BN,uint BK>
kernel void fcpe_linear_glu(constant uint4 & a [[buffer(0)]],
        device const float * w [[buffer(1)]],device const float * x [[buffer(2)]],
        device float * out [[buffer(3)]],device const float * bias [[buffer(4)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint THREADS=(BM/32)*(BN/16)*32;
    constexpr uint MEM=(BM+BN)*BK>BM*BN ? (BM+BN)*BK : BM*BN;
    threadgroup float mem[MEM];
    threadgroup float * sa=mem;
    threadgroup float * sb=mem+BM*BK;
    uint m0=group.y*(BM/2),n0=group.x*BN;
    uint sm=sg%(BM/32),sn=sg/(BM/32);
    simdgroup_float8x8 ma[4],mb[2],acc[8];
    for(uint j=0;j<8;++j)acc[j]=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(uint k0=0;k0<a.x;k0+=BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=tid;i<BM*(BK/16);i+=THREADS) {
            uint m=i/(BK/16),kbase=(i%(BK/16))*16;
            uint wm=m0+(m%(BM/2))+(m/(BM/2))*(a.y/2);
            float4 v[4];
            if(wm<a.y && k0+kbase+15<a.x) {
                for(uint q=0;q<4;++q)v[q]=*((device const float4 *)(w+wm*a.x+k0+kbase+q*4));
            } else {
                for(uint q=0;q<4;++q)for(uint j=0;j<4;++j)v[q][j]=(wm<a.y&&k0+kbase+q*4+j<a.x)?w[wm*a.x+k0+kbase+q*4+j]:0.0f;
            }
            #pragma unroll
            for(uint q=0;q<16;++q) {
                uint k=kbase+q;
                sa[(k/8)*BM*8+(m/8)*64+(k%8)*8+m%8]=v[q/4][q%4];
            }
        }

        for(uint i=tid;i<BN*(BK/8);i+=THREADS) {
            uint n=i/(BK/8),kbase=(i%(BK/8))*8;
            float4 v[2];
            if(n0+n<a.z && k0+kbase+7<a.x) {
                v[0]=*((device const float4 *)(x+(n0+n)*a.x+k0+kbase));
                v[1]=*((device const float4 *)(x+(n0+n)*a.x+k0+kbase+4));
            } else {
                for(uint q=0;q<2;++q)for(uint j=0;j<4;++j)v[q][j]=(n0+n<a.z&&k0+kbase+q*4+j<a.x)?x[(n0+n)*a.x+k0+kbase+q*4+j]:0.0f;
            }
            #pragma unroll
            for(uint q=0;q<8;++q) {
                uint k=kbase+q;
                sb[(k/8)*BN*8+(n/8)*64+(n%8)*8+k%8]=v[q/4][q%4];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for(uint k=0;k<BK/8;++k) {
            #pragma unroll
            for(uint i=0;i<4;++i)simdgroup_load(ma[i],sa+k*BM*8+sm*32*8+i*64,8,0,false);
            #pragma unroll
            for(uint i=0;i<2;++i)simdgroup_load(mb[i],sb+k*BN*8+sn*16*8+i*64,8,0,false);
            #pragma unroll
            for(uint i=0;i<8;++i)simdgroup_multiply_accumulate(acc[i],mb[i/4],ma[i%4],acc[i]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=0;i<8;++i)simdgroup_store(acc[i],mem+(sn*16+(i/4)*8)*BM+sm*32+(i%4)*8,BM,0,false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<(BM/2)*BN;i+=THREADS) {
        uint m=i%(BM/2),n=i/(BM/2);
        if(m0+m<a.y/2 && n0+n<a.z) {
            float value=mem[n*BM+m]+bias[m0+m];
            float gate=mem[n*BM+m+BM/2]+bias[m0+m+a.y/2];
            out[(n0+n)*(a.y/2)+m0+m]=value/(1.0f+exp(-gate));
        }
    }
}

template [[host_name("fcpe_linear_glu")]] kernel decltype(fcpe_linear_glu<64,32,32>) fcpe_linear_glu<64,32,32>;
