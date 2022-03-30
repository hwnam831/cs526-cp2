; ModuleID = 'matrixMul_kernel.nvvm.bc'
source_filename = "matrixMul_kernel.cl"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-nvcl-nvidial"

; Function Attrs: nofree noinline norecurse nounwind
define dso_local spir_kernel void @matmul(float addrspace(1)* nocapture readonly %A, float addrspace(1)* nocapture readonly %B, float addrspace(1)* nocapture %C, i32 %width, i32 %height) local_unnamed_addr #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %cmp46 = icmp sgt i32 %width, 0
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #3, !range !8
  %1 = shl nuw nsw i32 %0, 4
  %2 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #3, !range !9
  %narrow42 = add nuw nsw i32 %1, %2
  %3 = shl nuw nsw i32 %narrow42, 10
  %4 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3, !range !8
  %5 = shl nuw nsw i32 %4, 4
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !9
  br i1 %cmp46, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %xtraiter = and i32 %width, 1
  %7 = icmp eq i32 %width, 1
  br i1 %7, label %for.end.loopexit.unr-lcssa, label %for.body.preheader.new

for.body.preheader.new:                           ; preds = %for.body.preheader
  %unroll_iter = and i32 %width, -2
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader.new
  %i.048 = phi i32 [ 0, %for.body.preheader.new ], [ %add13.1, %for.body ]
  %sum.047 = phi float [ 0.000000e+00, %for.body.preheader.new ], [ %13, %for.body ]
  %niter = phi i32 [ %unroll_iter, %for.body.preheader.new ], [ %niter.nsub.1, %for.body ]
  %narrow43 = add nuw i32 %3, %i.048
  %add3 = zext i32 %narrow43 to i64
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %A, i64 %add3
  %8 = load float, float addrspace(1)* %arrayidx, align 4, !tbaa !10
  %mul4 = shl nsw i32 %i.048, 10
  %narrow44 = or i32 %6, %mul4
  %narrow45 = add nuw i32 %narrow44, %5
  %add10 = zext i32 %narrow45 to i64
  %arrayidx11 = getelementptr inbounds float, float addrspace(1)* %B, i64 %add10
  %9 = load float, float addrspace(1)* %arrayidx11, align 4, !tbaa !10
  %10 = tail call float @llvm.fmuladd.f32(float %8, float %9, float %sum.047)
  %add13 = or i32 %i.048, 1
  %narrow43.1 = add nuw i32 %3, %add13
  %add3.1 = zext i32 %narrow43.1 to i64
  %arrayidx.1 = getelementptr inbounds float, float addrspace(1)* %A, i64 %add3.1
  %11 = load float, float addrspace(1)* %arrayidx.1, align 4, !tbaa !10
  %mul4.1 = shl nsw i32 %add13, 10
  %narrow44.1 = or i32 %6, %mul4.1
  %narrow45.1 = add nuw i32 %narrow44.1, %5
  %add10.1 = zext i32 %narrow45.1 to i64
  %arrayidx11.1 = getelementptr inbounds float, float addrspace(1)* %B, i64 %add10.1
  %12 = load float, float addrspace(1)* %arrayidx11.1, align 4, !tbaa !10
  %13 = tail call float @llvm.fmuladd.f32(float %11, float %12, float %10)
  %add13.1 = add nuw nsw i32 %i.048, 2
  %niter.nsub.1 = add i32 %niter, -2
  %niter.ncmp.1 = icmp eq i32 %niter.nsub.1, 0
  br i1 %niter.ncmp.1, label %for.end.loopexit.unr-lcssa, label %for.body

for.end.loopexit.unr-lcssa:                       ; preds = %for.body, %for.body.preheader
  %.lcssa.ph = phi float [ undef, %for.body.preheader ], [ %13, %for.body ]
  %i.048.unr = phi i32 [ 0, %for.body.preheader ], [ %add13.1, %for.body ]
  %sum.047.unr = phi float [ 0.000000e+00, %for.body.preheader ], [ %13, %for.body ]
  %lcmp.mod.not = icmp eq i32 %xtraiter, 0
  br i1 %lcmp.mod.not, label %for.end, label %for.end.loopexit.epilog-lcssa

for.end.loopexit.epilog-lcssa:                    ; preds = %for.end.loopexit.unr-lcssa
  %narrow43.epil = add nuw i32 %3, %i.048.unr
  %add3.epil = zext i32 %narrow43.epil to i64
  %arrayidx.epil = getelementptr inbounds float, float addrspace(1)* %A, i64 %add3.epil
  %14 = load float, float addrspace(1)* %arrayidx.epil, align 4, !tbaa !10
  %mul4.epil = shl nsw i32 %i.048.unr, 10
  %narrow44.epil = or i32 %6, %mul4.epil
  %narrow45.epil = add nuw i32 %narrow44.epil, %5
  %add10.epil = zext i32 %narrow45.epil to i64
  %arrayidx11.epil = getelementptr inbounds float, float addrspace(1)* %B, i64 %add10.epil
  %15 = load float, float addrspace(1)* %arrayidx11.epil, align 4, !tbaa !10
  %16 = tail call float @llvm.fmuladd.f32(float %14, float %15, float %sum.047.unr)
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit.epilog-lcssa, %for.end.loopexit.unr-lcssa, %entry
  %sum.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %.lcssa.ph, %for.end.loopexit.unr-lcssa ], [ %16, %for.end.loopexit.epilog-lcssa ]
  %narrow40 = add nuw nsw i32 %5, %6
  %narrow41 = add nuw i32 %narrow40, %3
  %add23 = zext i32 %narrow41 to i64
  %arrayidx24 = getelementptr inbounds float, float addrspace(1)* %C, i64 %add23
  store float %sum.0.lcssa, float addrspace(1)* %arrayidx24, align 4, !tbaa !10
  ret void
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

attributes #0 = { nofree noinline norecurse nounwind "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+ptx32,+sm_20" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!nvvm.annotations = !{!0}
!llvm.module.flags = !{!1}
!opencl.ocl.version = !{!2}
!llvm.ident = !{!3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3}

!0 = !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32, i32)* @matmul, !"kernel", i32 1}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 2, i32 0}
!3 = !{!"clang version 12.0.0"}
!4 = !{i32 1, i32 1, i32 1, i32 0, i32 0}
!5 = !{!"none", !"none", !"none", !"none", !"none"}
!6 = !{!"float*", !"float*", !"float*", !"int", !"int"}
!7 = !{!"", !"", !"", !"", !""}
!8 = !{i32 0, i32 65535}
!9 = !{i32 0, i32 1024}
!10 = !{!11, !11, i64 0}
!11 = !{!"float", !12, i64 0}
!12 = !{!"omnipotent char", !13, i64 0}
!13 = !{!"Simple C/C++ TBAA"}
