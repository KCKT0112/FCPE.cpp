// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. See LICENSE or https://mozilla.org/MPL/2.0/.

static ggml_metal_pipeline_with_params fcpe_metal_pipeline(ggml_metal_op_t ctx, const char * name) {
    auto p = ggml_metal_library_get_pipeline(ctx->lib, name);
    if (!p.pipeline) p = ggml_metal_library_compile_pipeline(ctx->lib, name, name, nullptr);
    return p;
}

static bool fcpe_metal_copy(ggml_metal_op_t ctx, int idx) {
    if (getenv("FCPE_METAL_LEGACY_COPY")) return false;
    auto * op = ctx->node(idx);
    auto * a = op->src[0];
    if (op->op != GGML_OP_CONT || !a || a->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 ||
        a->ne[2] != 1 || a->ne[3] != 1 || !ggml_is_contiguous(op) || !ggml_are_same_shape(a, op)) return false;
    const bool transpose = a->nb[1] == 4 && a->nb[0] >= 4*size_t(a->ne[1]);
    const bool rows = a->nb[0] == 4 && a->nb[1] >= 4*size_t(a->ne[0]);
    if ((!transpose && !rows) || uint64_t(ggml_nelements(op)) > UINT32_MAX) return false;
    uint64_t args[] = {uint64_t(a->ne[0]),uint64_t(a->ne[1]),a->nb[0]/4,a->nb[1]/4};
    const auto p = fcpe_metal_pipeline(ctx, transpose ? "fcpe_transpose_f32" : "fcpe_copy_rows_f32");
    ggml_metal_encoder_set_pipeline(ctx->enc, p);
    ggml_metal_encoder_set_bytes(ctx->enc, args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(a), 1);
    ggml_metal_encoder_set_buffer(ctx->enc, ggml_metal_get_buffer_id(op), 2);
    if (transpose)
        ggml_metal_encoder_dispatch_threadgroups(ctx->enc,(a->ne[0]+31)/32,(a->ne[1]+31)/32,1,32,8,1);
    else
        ggml_metal_encoder_dispatch_threadgroups(ctx->enc,(ggml_nelements(op)+255)/256,1,1,256,1,1);
    return true;
}

static bool fcpe_metal_matrix(const ggml_tensor * t) {
    return t && t->type == GGML_TYPE_F32 && t->ne[2] == 1 && t->ne[3] == 1 &&
        t->nb[0] == 4 && t->nb[1] >= size_t(t->ne[0])*4 && t->nb[1]/4 <= UINT32_MAX && uint64_t(ggml_nelements(t)) <= UINT32_MAX;
}
static bool fcpe_metal_bias(const ggml_tensor * t, int64_t cols) {
    return fcpe_metal_matrix(t) && t->ne[0] == cols && t->ne[1] == 1;
}
static bool fcpe_metal_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t ap = (uintptr_t)a->data, bp = (uintptr_t)b->data;
    return ap < bp + ggml_nbytes(b) && bp < ap + ggml_nbytes(a);
}
static bool fcpe_metal_im2col(ggml_metal_op_t ctx,int idx) {
    if(getenv("FCPE_METAL_LEGACY_IM2COL"))return false;
    auto * op=ctx->node(idx);auto * w=op->src[0];auto * x=op->src[1];const auto * p=(const int32_t *)op->op_params;
    if(!fcpe_metal_matrix(x)||!fcpe_metal_matrix(op)||!ggml_is_contiguous(x)||!ggml_is_contiguous(op)||
       w->ne[0]!=3||w->ne[1]!=x->ne[1]||p[0]!=1||p[2]!=1||p[4]!=1||p[6]!=0)return false;
    uint32_t args[]={uint32_t(x->ne[0]),uint32_t(x->ne[1])};
    auto pipeline=fcpe_metal_pipeline(ctx,"fcpe_im2col3");
    ggml_metal_encoder_set_pipeline(ctx->enc,pipeline);
    ggml_metal_encoder_set_bytes(ctx->enc,args,sizeof(args),0);
    ggml_metal_encoder_set_buffer(ctx->enc,ggml_metal_get_buffer_id(x),1);
    ggml_metal_encoder_set_buffer(ctx->enc,ggml_metal_get_buffer_id(op),2);
    ggml_metal_encoder_dispatch_threadgroups(ctx->enc,(args[0]+31)/32,(args[1]+31)/32,1,256,1,1);
    return true;
}
static bool fcpe_metal_depthwise(ggml_metal_op_t ctx, int idx) {
    if (getenv("FCPE_METAL_LEGACY_DW")) return false;
    auto * op=ctx->node(idx); auto * w=op->src[0]; auto * x=op->src[1];
    const int32_t * p=(const int32_t *)op->op_params;
    if(w->type==GGML_TYPE_F32 && x->type==GGML_TYPE_F32 && op->type==GGML_TYPE_F32 &&
       w->ne[0]==31 && w->ne[1]==1 && w->ne[2]==1 && x->ne[1]==1 && x->ne[3]==1 &&
       ggml_is_contiguous(w) && x->nb[2]==4 && x->nb[0]==size_t(x->ne[2])*4 &&
       op->nb[2]==4 && op->nb[0]==size_t(x->ne[2])*4 && p[0]==1 && p[1]==1 && p[2]==15 && p[3]==0 && p[4]==1 && p[5]==1 &&
       uint64_t(ggml_nelements(op))<=UINT32_MAX && x->ne[0]<=INT32_MAX) {
        uint32_t args[]={uint32_t(x->ne[0]),uint32_t(x->ne[2])};
        auto pipeline=fcpe_metal_pipeline(ctx,"fcpe_depthwise_channels");
        ggml_metal_encoder_set_pipeline(ctx->enc,pipeline);
        ggml_metal_encoder_set_bytes(ctx->enc,args,sizeof(args),0);
        ggml_metal_encoder_set_buffer(ctx->enc,ggml_metal_get_buffer_id(w),1);
        ggml_metal_encoder_set_buffer(ctx->enc,ggml_metal_get_buffer_id(x),2);
        ggml_metal_encoder_set_buffer(ctx->enc,ggml_metal_get_buffer_id(op),3);
        ggml_metal_encoder_dispatch_threadgroups(ctx->enc,(args[0]+7)/8,(args[1]+31)/32,1,128,1,1);
        return true;
    }

    if (w->type!=GGML_TYPE_F32 || x->type!=GGML_TYPE_F32 || op->type!=GGML_TYPE_F32 ||
        w->ne[0]!=31 || w->ne[1]!=1 || w->ne[2]!=1 || x->ne[1]!=1 ||
        !ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(op) ||
        p[0]!=1 || p[1]!=1 || p[2]!=15 || p[3]!=0 || p[4]!=1 || p[5]!=1 ||
        x->ne[0]>INT32_MAX || uint64_t(ggml_nelements(op))>UINT32_MAX) return false;
    uint32_t args[]={uint32_t(x->ne[0]),uint32_t(x->ne[2])};
    auto pipeline=fcpe_metal_pipeline(ctx,"fcpe_depthwise31");
    ggml_metal_encoder_set_pipeline(ctx->enc,pipeline);
    ggml_metal_encoder_set_bytes(ctx->enc,args,sizeof(args),0);
    ggml_metal_encoder_set_buffer(ctx->enc,ggml_metal_get_buffer_id(w),1);
    ggml_metal_encoder_set_buffer(ctx->enc,ggml_metal_get_buffer_id(x),2);
    ggml_metal_encoder_set_buffer(ctx->enc,ggml_metal_get_buffer_id(op),3);
    ggml_metal_encoder_dispatch_threadgroups(ctx->enc,(args[0]+255)/256,x->ne[2]*x->ne[3],1,64,1,1);
    return true;
}
static bool fcpe_metal_group_norm(ggml_metal_op_t ctx, int idx) {
    if (getenv("FCPE_METAL_LEGACY_GN")) return false;
    auto * op=ctx->node(idx); auto * x=op->src[0];
    const int groups=((const int32_t *)op->op_params)[0];
    if (x->type!=GGML_TYPE_F32 || op->type!=GGML_TYPE_F32 || !ggml_is_contiguous(x) ||
        !ggml_is_contiguous(op) || x->ne[3]!=1 || x->ne[2]%groups ||
        uint64_t(ggml_nelements(op))>UINT32_MAX ||
        fcpe_metal_overlap(x,op)) return false;
    const uint32_t count=uint32_t(ggml_nelements(op)/groups), chunks=(count+1023)/1024;
    struct { uint32_t count,chunks,groups,phase; float eps; } args {count,chunks,uint32_t(groups),0,0};
    memcpy(&args.eps,op->op_params+1,4);
    if(count<4096) {
        auto pipeline=fcpe_metal_pipeline(ctx,"fcpe_group_small");
        ggml_metal_encoder_set_pipeline(ctx->enc,pipeline);
        ggml_metal_encoder_set_bytes(ctx->enc,&args,sizeof(args),0);
        ggml_metal_encoder_set_buffer(ctx->enc,ggml_metal_get_buffer_id(x),1);
        ggml_metal_encoder_set_buffer(ctx->enc,ggml_metal_get_buffer_id(op),2);
        ggml_metal_encoder_dispatch_threadgroups(ctx->enc,groups,1,1,256,1,1);
        return true;
    }
    auto out=ggml_metal_get_buffer_id(op), scratch=out;
    scratch.offs+=GGML_PAD(ggml_nbytes(op),16);
    for (uint32_t phase=0;phase<3;++phase) {
        args.phase=phase;
        auto pipeline=fcpe_metal_pipeline(ctx,phase<2 ? "fcpe_group_stats" : "fcpe_group_apply");
        ggml_metal_encoder_set_pipeline(ctx->enc,pipeline);
        ggml_metal_encoder_set_bytes(ctx->enc,&args,sizeof(args),0);
        ggml_metal_encoder_set_buffer(ctx->enc,ggml_metal_get_buffer_id(x),1);
        ggml_metal_encoder_set_buffer(ctx->enc,scratch,2);
        ggml_metal_encoder_set_buffer(ctx->enc,out,3);
        ggml_metal_encoder_dispatch_threadgroups(ctx->enc,chunks,groups,1,256,1,1);
        if (phase<2) ggml_metal_encoder_memory_barrier(ctx->enc);
    }
    return true;
}
static int fcpe_metal_glu_project(ggml_metal_op_t ctx,int idx) {
    if(getenv("FCPE_METAL_DISABLE_GLU_PROJECT")||idx+3>=ctx->n_nodes()||
       !ggml_metal_device_get_props(ctx->dev)->has_simdgroup_mm)return 0;
    auto * mm=ctx->node(idx);auto * add=ctx->node(idx+1);auto * sigmoid=ctx->node(idx+2);auto * out=ctx->node(idx+3);
    if(mm->op!=GGML_OP_MUL_MAT||add->op!=GGML_OP_ADD||sigmoid->op!=GGML_OP_UNARY||out->op!=GGML_OP_MUL||
       ggml_get_unary_op(sigmoid)!=GGML_UNARY_OP_SIGMOID||add->src[0]!=mm||out->src[1]!=sigmoid||
       !fcpe_metal_matrix(mm)||!fcpe_metal_matrix(out)||!ggml_is_contiguous(out)||mm->ne[0]!=out->ne[0]*2||
       !fcpe_metal_bias(add->src[1],mm->ne[0])||!ctx->can_fuse_span(idx,4))return 0;
    auto * w=mm->src[0];auto * x=mm->src[1];auto * a=out->src[0];auto * b=sigmoid->src[0];
    if(!fcpe_metal_matrix(w)||!fcpe_metal_matrix(x)||!ggml_is_contiguous(w)||!ggml_is_contiguous(x)||
       w->ne[0]%4||w->ne[1]%64||x->ne[1]<9||(uintptr_t)w->data%16||(uintptr_t)x->data%16||a->view_src!=add||b->view_src!=add||
       a->view_offs!=0||b->view_offs!=size_t(out->ne[0])*4||a->nb[1]!=mm->nb[1]||b->nb[1]!=mm->nb[1]||
       !ggml_are_same_shape(a,out)||!ggml_are_same_shape(b,out)||
       fcpe_metal_overlap(out,x)||fcpe_metal_overlap(out,w)||fcpe_metal_overlap(out,add->src[1]))return 0;
    for(int j=1;j<4;++j)if(!ggml_metal_op_concurrency_check(ctx,ctx->node(idx+j)))ggml_metal_op_concurrency_reset(ctx);
    uint32_t args[]={uint32_t(w->ne[0]),uint32_t(w->ne[1]),uint32_t(x->ne[1]),0};
    auto pipeline=fcpe_metal_pipeline(ctx,"fcpe_linear_glu");
    ggml_metal_encoder_set_pipeline(ctx->enc,pipeline);
    ggml_metal_encoder_set_bytes(ctx->enc,args,sizeof(args),0);
    ggml_metal_encoder_set_buffer(ctx->enc,ggml_metal_get_buffer_id(w),1);
    ggml_metal_encoder_set_buffer(ctx->enc,ggml_metal_get_buffer_id(x),2);
    ggml_metal_encoder_set_buffer(ctx->enc,ggml_metal_get_buffer_id(out),3);
    ggml_metal_encoder_set_buffer(ctx->enc,ggml_metal_get_buffer_id(add->src[1]),4);
    ggml_metal_encoder_dispatch_threadgroups(ctx->enc,(args[2]+31)/32,(args[1]/2+31)/32,1,128,1,1);
    return 4;
}
static int fcpe_metal_fuse(ggml_metal_op_t ctx, int idx) {
    if (!ctx->use_fusion || getenv("FCPE_METAL_DISABLE_FUSION") || idx+1 >= ctx->n_nodes()) return 0;
    if (const int fused=fcpe_metal_glu_project(ctx,idx)) return fused;
    auto * first=ctx->node(idx); auto * last=ctx->node(idx+1);
    if (!fcpe_metal_matrix(first) || !fcpe_metal_matrix(last) || !ggml_is_contiguous(last)) return 0;
    const ggml_op ops[] = {first->op,last->op};
    if (!ctx->can_fuse(idx,ops,2)) return 0;
    auto * a=first->src[0]; auto * b=first->src[1];
    int mode=-1;
    if (first->op == GGML_OP_UNARY && ggml_get_unary_op(first) == GGML_UNARY_OP_SIGMOID &&
        last->op == GGML_OP_MUL && last->src[1] == first) {
        b=last->src[0];
        if (fcpe_metal_matrix(b) && ggml_are_same_shape(b,first)) mode=0;
    } else if (first->op == GGML_OP_ADD && last->src[0] == first && fcpe_metal_bias(b,first->ne[0])) {
        if (last->op == GGML_OP_UNARY) {
            if (ggml_get_unary_op(last) == GGML_UNARY_OP_SIGMOID) mode=1;
            if (ggml_get_unary_op(last) == GGML_UNARY_OP_SILU) mode=2;
        } else if (last->op == GGML_OP_LEAKY_RELU) mode=3;
    }
    if (mode<0 || !fcpe_metal_matrix(a) || fcpe_metal_overlap(a,last) || fcpe_metal_overlap(b,last)) return 0;
    struct { uint32_t count,cols,astride,bstride,mode; float alpha; } args {
        uint32_t(ggml_nelements(last)),uint32_t(last->ne[0]),uint32_t(a->nb[1]/4),
        uint32_t(b->nb[1]/4),uint32_t(mode),0.0f};
    if (mode==3) memcpy(&args.alpha,last->op_params,4);
    auto p=fcpe_metal_pipeline(ctx,"fcpe_elementwise_fused");
    // Include all fused consumers in the same upstream hazard checks.
    if (!ggml_metal_op_concurrency_check(ctx,last)) ggml_metal_op_concurrency_reset(ctx);
    ggml_metal_encoder_set_pipeline(ctx->enc,p);
    ggml_metal_encoder_set_bytes(ctx->enc,&args,sizeof(args),0);
    ggml_metal_encoder_set_buffer(ctx->enc,ggml_metal_get_buffer_id(a),1);
    ggml_metal_encoder_set_buffer(ctx->enc,ggml_metal_get_buffer_id(b),2);
    ggml_metal_encoder_set_buffer(ctx->enc,ggml_metal_get_buffer_id(last),3);
    ggml_metal_encoder_dispatch_threadgroups(ctx->enc,(args.count+255)/256,1,1,256,1,1);
    return 2;
}
