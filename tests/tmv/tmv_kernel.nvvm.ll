; ModuleID = 'tmv_kernel.nvvm.bc'
source_filename = "tmv_kernel.cl"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-nvcl-nvidial"

; Function Attrs: nofree noinline norecurse nounwind
define dso_local spir_kernel void @tmv(float addrspace(1)* nocapture readonly %A, float addrspace(1)* nocapture readonly %B, float addrspace(1)* nocapture %C, i32 %width) local_unnamed_addr #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %cmp25 = icmp sgt i32 %width, 0
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3, !range !8
  %1 = shl nuw nsw i32 %0, 8
  %2 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !9
  br i1 %cmp25, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %xtraiter = and i32 %width, 1
  %3 = icmp eq i32 %width, 1
  br i1 %3, label %for.end.loopexit.unr-lcssa, label %for.body.preheader.new

for.body.preheader.new:                           ; preds = %for.body.preheader
  %unroll_iter = and i32 %width, -2
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader.new
  %i.027 = phi i32 [ 0, %for.body.preheader.new ], [ %add6.1, %for.body ]
  %sum.026 = phi float [ 0.000000e+00, %for.body.preheader.new ], [ %9, %for.body ]
  %niter = phi i32 [ %unroll_iter, %for.body.preheader.new ], [ %niter.nsub.1, %for.body ]
  %mul = shl nsw i32 %i.027, 11
  %narrow23 = or i32 %2, %mul
  %narrow24 = add nuw i32 %narrow23, %1
  %add3 = zext i32 %narrow24 to i64
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %A, i64 %add3
  %4 = load float, float addrspace(1)* %arrayidx, align 4, !tbaa !10
  %idxprom = zext i32 %i.027 to i64
  %arrayidx4 = getelementptr inbounds float, float addrspace(1)* %B, i64 %idxprom
  %5 = load float, float addrspace(1)* %arrayidx4, align 4, !tbaa !10
  %6 = tail call float @llvm.fmuladd.f32(float %4, float %5, float %sum.026)
  %add6 = or i32 %i.027, 1
  %mul.1 = shl nsw i32 %add6, 11
  %narrow23.1 = or i32 %2, %mul.1
  %narrow24.1 = add nuw i32 %narrow23.1, %1
  %add3.1 = zext i32 %narrow24.1 to i64
  %arrayidx.1 = getelementptr inbounds float, float addrspace(1)* %A, i64 %add3.1
  %7 = load float, float addrspace(1)* %arrayidx.1, align 4, !tbaa !10
  %idxprom.1 = zext i32 %add6 to i64
  %arrayidx4.1 = getelementptr inbounds float, float addrspace(1)* %B, i64 %idxprom.1
  %8 = load float, float addrspace(1)* %arrayidx4.1, align 4, !tbaa !10
  %9 = tail call float @llvm.fmuladd.f32(float %7, float %8, float %6)
  %add6.1 = add nuw nsw i32 %i.027, 2
  %niter.nsub.1 = add i32 %niter, -2
  %niter.ncmp.1 = icmp eq i32 %niter.nsub.1, 0
  br i1 %niter.ncmp.1, label %for.end.loopexit.unr-lcssa, label %for.body

for.end.loopexit.unr-lcssa:                       ; preds = %for.body, %for.body.preheader
  %.lcssa.ph = phi float [ undef, %for.body.preheader ], [ %9, %for.body ]
  %i.027.unr = phi i32 [ 0, %for.body.preheader ], [ %add6.1, %for.body ]
  %sum.026.unr = phi float [ 0.000000e+00, %for.body.preheader ], [ %9, %for.body ]
  %lcmp.mod.not = icmp eq i32 %xtraiter, 0
  br i1 %lcmp.mod.not, label %for.end, label %for.end.loopexit.epilog-lcssa

for.end.loopexit.epilog-lcssa:                    ; preds = %for.end.loopexit.unr-lcssa
  %mul.epil = shl nsw i32 %i.027.unr, 11
  %narrow23.epil = or i32 %2, %mul.epil
  %narrow24.epil = add nuw i32 %narrow23.epil, %1
  %add3.epil = zext i32 %narrow24.epil to i64
  %arrayidx.epil = getelementptr inbounds float, float addrspace(1)* %A, i64 %add3.epil
  %10 = load float, float addrspace(1)* %arrayidx.epil, align 4, !tbaa !10
  %idxprom.epil = zext i32 %i.027.unr to i64
  %arrayidx4.epil = getelementptr inbounds float, float addrspace(1)* %B, i64 %idxprom.epil
  %11 = load float, float addrspace(1)* %arrayidx4.epil, align 4, !tbaa !10
  %12 = tail call float @llvm.fmuladd.f32(float %10, float %11, float %sum.026.unr)
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit.epilog-lcssa, %for.end.loopexit.unr-lcssa, %entry
  %sum.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %.lcssa.ph, %for.end.loopexit.unr-lcssa ], [ %12, %for.end.loopexit.epilog-lcssa ]
  %narrow = add nuw nsw i32 %1, %2
  %add10 = zext i32 %narrow to i64
  %arrayidx11 = getelementptr inbounds float, float addrspace(1)* %C, i64 %add10
  store float %sum.0.lcssa, float addrspace(1)* %arrayidx11, align 4, !tbaa !10
  ret void
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.fmuladd.f32(float, float, float) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

attributes #0 = { nofree noinline norecurse nounwind "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+ptx32,+sm_20" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!nvvm.annotations = !{!0}
!llvm.module.flags = !{!1}
!opencl.ocl.version = !{!2}
!llvm.ident = !{!3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3}

!0 = !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @tmv, !"kernel", i32 1}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 2, i32 0}
!3 = !{!"clang version 12.0.0"}
!4 = !{i32 1, i32 1, i32 1, i32 0}
!5 = !{!"none", !"none", !"none", !"none"}
!6 = !{!"float*", !"float*", !"float*", !"int"}
!7 = !{!"", !"", !"", !""}
!8 = !{i32 0, i32 65535}
!9 = !{i32 0, i32 1024}
!10 = !{!11, !11, i64 0}
!11 = !{!"float", !12, i64 0}
!12 = !{!"omnipotent char", !13, i64 0}
!13 = !{!"Simple C/C++ TBAA"}
