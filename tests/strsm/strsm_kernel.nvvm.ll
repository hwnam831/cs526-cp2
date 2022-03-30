; ModuleID = 'strsm_kernel.nvvm.bc'
source_filename = "strsm_kernel.cl"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-nvcl-nvidial"

; Function Attrs: nofree noinline norecurse nounwind
define dso_local spir_kernel void @strsm(float addrspace(1)* nocapture readonly %A, float addrspace(1)* nocapture readonly %B, float addrspace(1)* nocapture %C, i32 %width, i32 %input_width, i32 %i_val) local_unnamed_addr #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !8
  %1 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #3, !range !8
  %2 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3, !range !9
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #3, !range !10
  %narrow = mul nuw nsw i32 %3, %2
  %narrow58 = add nuw nsw i32 %narrow, %0
  %4 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #3, !range !9
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #3, !range !10
  %narrow59 = mul nuw nsw i32 %5, %4
  %narrow60 = add nuw nsw i32 %narrow59, %1
  %idx.ext = sext i32 %i_val to i64
  %add.ptr = getelementptr inbounds float, float addrspace(1)* %A, i64 %idx.ext
  %add13 = add nsw i32 %i_val, %width
  %mul14 = mul nsw i32 %i_val, %input_width
  %add15 = add nsw i32 %add13, %mul14
  %idx.ext16 = sext i32 %add15 to i64
  %add.ptr17 = getelementptr inbounds float, float addrspace(1)* %B, i64 %idx.ext16
  %cmp61 = icmp sgt i32 %width, 0
  %mul22 = shl nsw i32 %narrow60, 13
  br i1 %cmp61, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %xtraiter = and i32 %width, 1
  %6 = icmp eq i32 %width, 1
  br i1 %6, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body.preheader.new

for.body.preheader.new:                           ; preds = %for.body.preheader
  %unroll_iter = and i32 %width, -2
  br label %for.body

for.cond.cleanup.loopexit.unr-lcssa:              ; preds = %for.body, %for.body.preheader
  %.lcssa.ph = phi float [ undef, %for.body.preheader ], [ %16, %for.body ]
  %i.063.unr = phi i32 [ 0, %for.body.preheader ], [ %inc.1, %for.body ]
  %sum.062.unr = phi float [ 0.000000e+00, %for.body.preheader ], [ %16, %for.body ]
  %lcmp.mod.not = icmp eq i32 %xtraiter, 0
  br i1 %lcmp.mod.not, label %for.cond.cleanup, label %for.cond.cleanup.loopexit.epilog-lcssa

for.cond.cleanup.loopexit.epilog-lcssa:           ; preds = %for.cond.cleanup.loopexit.unr-lcssa
  %add23.epil = add nuw nsw i32 %i.063.unr, %mul22
  %idxprom.epil = zext i32 %add23.epil to i64
  %arrayidx.epil = getelementptr inbounds float, float addrspace(1)* %add.ptr, i64 %idxprom.epil
  %7 = load float, float addrspace(1)* %arrayidx.epil, align 4, !tbaa !11
  %mul24.epil = shl nsw i32 %i.063.unr, 13
  %add25.epil = add nuw nsw i32 %mul24.epil, %narrow58
  %idxprom26.epil = zext i32 %add25.epil to i64
  %arrayidx27.epil = getelementptr inbounds float, float addrspace(1)* %add.ptr17, i64 %idxprom26.epil
  %8 = load float, float addrspace(1)* %arrayidx27.epil, align 4, !tbaa !11
  %9 = tail call float @llvm.fmuladd.f32(float %7, float %8, float %sum.062.unr)
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit.epilog-lcssa, %for.cond.cleanup.loopexit.unr-lcssa, %entry
  %sum.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %.lcssa.ph, %for.cond.cleanup.loopexit.unr-lcssa ], [ %9, %for.cond.cleanup.loopexit.epilog-lcssa ]
  %idx.ext19 = sext i32 %add13 to i64
  %add.ptr20 = getelementptr inbounds float, float addrspace(1)* %C, i64 %idx.ext19
  %add30 = add nuw nsw i32 %mul22, %narrow58
  %idxprom31 = zext i32 %add30 to i64
  %arrayidx32 = getelementptr inbounds float, float addrspace(1)* %add.ptr20, i64 %idxprom31
  %10 = load float, float addrspace(1)* %arrayidx32, align 4, !tbaa !11
  %sub = fsub float %10, %sum.0.lcssa
  store float %sub, float addrspace(1)* %arrayidx32, align 4, !tbaa !11
  ret void

for.body:                                         ; preds = %for.body, %for.body.preheader.new
  %i.063 = phi i32 [ 0, %for.body.preheader.new ], [ %inc.1, %for.body ]
  %sum.062 = phi float [ 0.000000e+00, %for.body.preheader.new ], [ %16, %for.body ]
  %niter = phi i32 [ %unroll_iter, %for.body.preheader.new ], [ %niter.nsub.1, %for.body ]
  %add23 = add nuw nsw i32 %i.063, %mul22
  %idxprom = zext i32 %add23 to i64
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %add.ptr, i64 %idxprom
  %11 = load float, float addrspace(1)* %arrayidx, align 4, !tbaa !11
  %mul24 = shl nsw i32 %i.063, 13
  %add25 = add nuw nsw i32 %mul24, %narrow58
  %idxprom26 = zext i32 %add25 to i64
  %arrayidx27 = getelementptr inbounds float, float addrspace(1)* %add.ptr17, i64 %idxprom26
  %12 = load float, float addrspace(1)* %arrayidx27, align 4, !tbaa !11
  %13 = tail call float @llvm.fmuladd.f32(float %11, float %12, float %sum.062)
  %inc = or i32 %i.063, 1
  %add23.1 = add nuw nsw i32 %inc, %mul22
  %idxprom.1 = zext i32 %add23.1 to i64
  %arrayidx.1 = getelementptr inbounds float, float addrspace(1)* %add.ptr, i64 %idxprom.1
  %14 = load float, float addrspace(1)* %arrayidx.1, align 4, !tbaa !11
  %mul24.1 = shl nsw i32 %inc, 13
  %add25.1 = add nuw nsw i32 %mul24.1, %narrow58
  %idxprom26.1 = zext i32 %add25.1 to i64
  %arrayidx27.1 = getelementptr inbounds float, float addrspace(1)* %add.ptr17, i64 %idxprom26.1
  %15 = load float, float addrspace(1)* %arrayidx27.1, align 4, !tbaa !11
  %16 = tail call float @llvm.fmuladd.f32(float %14, float %15, float %13)
  %inc.1 = add nuw nsw i32 %i.063, 2
  %niter.nsub.1 = add i32 %niter, -2
  %niter.ncmp.1 = icmp eq i32 %niter.nsub.1, 0
  br i1 %niter.ncmp.1, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.fmuladd.f32(float, float, float) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #2

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y() #2

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #2

attributes #0 = { nofree noinline norecurse nounwind "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+ptx32,+sm_20" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!nvvm.annotations = !{!0}
!llvm.module.flags = !{!1}
!opencl.ocl.version = !{!2}
!llvm.ident = !{!3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3}

!0 = !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32, i32, i32)* @strsm, !"kernel", i32 1}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 2, i32 0}
!3 = !{!"clang version 12.0.0"}
!4 = !{i32 1, i32 1, i32 1, i32 0, i32 0, i32 0}
!5 = !{!"none", !"none", !"none", !"none", !"none", !"none"}
!6 = !{!"float*", !"float*", !"float*", !"int", !"int", !"int"}
!7 = !{!"", !"", !"", !"", !"", !""}
!8 = !{i32 0, i32 1024}
!9 = !{i32 0, i32 65535}
!10 = !{i32 1, i32 1025}
!11 = !{!12, !12, i64 0}
!12 = !{!"float", !13, i64 0}
!13 = !{!"omnipotent char", !14, i64 0}
!14 = !{!"Simple C/C++ TBAA"}
