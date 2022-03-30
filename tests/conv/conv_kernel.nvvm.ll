; ModuleID = 'conv_kernel.nvvm.bc'
source_filename = "conv_kernel.cl"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-nvcl-nvidial"

; Function Attrs: nofree noinline norecurse nounwind
define dso_local spir_kernel void @conv(float addrspace(1)* nocapture readonly %A, float addrspace(1)* nocapture readonly %B, float addrspace(1)* nocapture %C, i32 %width, i32 %height, i32 %w, i32 %h) local_unnamed_addr #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %cmp63 = icmp sgt i32 %h, 0
  br i1 %cmp63, label %for.cond1.preheader.lr.ph, label %for.end24

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %cmp260 = icmp sgt i32 %w, 0
  %conv569 = zext i32 %h to i64
  %conv14 = sext i32 %w to i64
  %xtraiter = and i32 %w, 1
  %0 = icmp eq i32 %w, 1
  %unroll_iter = and i32 %w, -2
  %lcmp.mod.not = icmp eq i32 %xtraiter, 0
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond1.preheader.lr.ph, %for.inc22
  %j.065 = phi i32 [ 0, %for.cond1.preheader.lr.ph ], [ %add23, %for.inc22 ]
  %sum.064 = phi float [ 0.000000e+00, %for.cond1.preheader.lr.ph ], [ %sum.1.lcssa, %for.inc22 ]
  br i1 %cmp260, label %for.body3.lr.ph, label %for.inc22

for.body3.lr.ph:                                  ; preds = %for.cond1.preheader
  %1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #3, !range !8
  %2 = shl nuw nsw i32 %1, 4
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #3, !range !9
  %narrow58 = add nuw nsw i32 %2, %3
  %add = zext i32 %narrow58 to i64
  %conv = zext i32 %j.065 to i64
  %sub = sub nsw i64 %conv569, %conv
  %add6 = add nsw i64 %sub, %add
  %mul7 = mul nsw i64 %add6, 4128
  %4 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3, !range !8
  %5 = shl nuw nsw i32 %4, 4
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !9
  %narrow59 = add nuw nsw i32 %5, %6
  %add11 = zext i32 %narrow59 to i64
  %mul17 = shl nsw i32 %j.065, 5
  br i1 %0, label %for.inc22.loopexit.unr-lcssa, label %for.body3

for.body3:                                        ; preds = %for.body3.lr.ph, %for.body3
  %i.062 = phi i32 [ %add21.1, %for.body3 ], [ 0, %for.body3.lr.ph ]
  %sum.161 = phi float [ %12, %for.body3 ], [ %sum.064, %for.body3.lr.ph ]
  %niter = phi i32 [ %niter.nsub.1, %for.body3 ], [ %unroll_iter, %for.body3.lr.ph ]
  %conv12 = zext i32 %i.062 to i64
  %sub13 = sub nsw i64 %conv14, %conv12
  %add15 = add nsw i64 %sub13, %add11
  %add16 = add nsw i64 %add15, %mul7
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %A, i64 %add16
  %7 = load float, float addrspace(1)* %arrayidx, align 4, !tbaa !10
  %add18 = add nuw nsw i32 %i.062, %mul17
  %idxprom = zext i32 %add18 to i64
  %arrayidx19 = getelementptr inbounds float, float addrspace(1)* %B, i64 %idxprom
  %8 = load float, float addrspace(1)* %arrayidx19, align 4, !tbaa !10
  %9 = tail call float @llvm.fmuladd.f32(float %7, float %8, float %sum.161)
  %add21 = or i32 %i.062, 1
  %conv12.1 = zext i32 %add21 to i64
  %sub13.1 = sub nsw i64 %conv14, %conv12.1
  %add15.1 = add nsw i64 %sub13.1, %add11
  %add16.1 = add nsw i64 %add15.1, %mul7
  %arrayidx.1 = getelementptr inbounds float, float addrspace(1)* %A, i64 %add16.1
  %10 = load float, float addrspace(1)* %arrayidx.1, align 4, !tbaa !10
  %add18.1 = add nuw nsw i32 %add21, %mul17
  %idxprom.1 = zext i32 %add18.1 to i64
  %arrayidx19.1 = getelementptr inbounds float, float addrspace(1)* %B, i64 %idxprom.1
  %11 = load float, float addrspace(1)* %arrayidx19.1, align 4, !tbaa !10
  %12 = tail call float @llvm.fmuladd.f32(float %10, float %11, float %9)
  %add21.1 = add nuw nsw i32 %i.062, 2
  %niter.nsub.1 = add i32 %niter, -2
  %niter.ncmp.1 = icmp eq i32 %niter.nsub.1, 0
  br i1 %niter.ncmp.1, label %for.inc22.loopexit.unr-lcssa, label %for.body3

for.inc22.loopexit.unr-lcssa:                     ; preds = %for.body3, %for.body3.lr.ph
  %.lcssa.ph = phi float [ undef, %for.body3.lr.ph ], [ %12, %for.body3 ]
  %i.062.unr = phi i32 [ 0, %for.body3.lr.ph ], [ %add21.1, %for.body3 ]
  %sum.161.unr = phi float [ %sum.064, %for.body3.lr.ph ], [ %12, %for.body3 ]
  br i1 %lcmp.mod.not, label %for.inc22, label %for.inc22.loopexit.epilog-lcssa

for.inc22.loopexit.epilog-lcssa:                  ; preds = %for.inc22.loopexit.unr-lcssa
  %conv12.epil = zext i32 %i.062.unr to i64
  %sub13.epil = sub nsw i64 %conv14, %conv12.epil
  %add15.epil = add nsw i64 %sub13.epil, %add11
  %add16.epil = add nsw i64 %add15.epil, %mul7
  %arrayidx.epil = getelementptr inbounds float, float addrspace(1)* %A, i64 %add16.epil
  %13 = load float, float addrspace(1)* %arrayidx.epil, align 4, !tbaa !10
  %add18.epil = add nuw nsw i32 %i.062.unr, %mul17
  %idxprom.epil = zext i32 %add18.epil to i64
  %arrayidx19.epil = getelementptr inbounds float, float addrspace(1)* %B, i64 %idxprom.epil
  %14 = load float, float addrspace(1)* %arrayidx19.epil, align 4, !tbaa !10
  %15 = tail call float @llvm.fmuladd.f32(float %13, float %14, float %sum.161.unr)
  br label %for.inc22

for.inc22:                                        ; preds = %for.inc22.loopexit.epilog-lcssa, %for.inc22.loopexit.unr-lcssa, %for.cond1.preheader
  %sum.1.lcssa = phi float [ %sum.064, %for.cond1.preheader ], [ %.lcssa.ph, %for.inc22.loopexit.unr-lcssa ], [ %15, %for.inc22.loopexit.epilog-lcssa ]
  %add23 = add nuw nsw i32 %j.065, 1
  %exitcond68.not = icmp eq i32 %add23, %h
  br i1 %exitcond68.not, label %for.end24, label %for.cond1.preheader

for.end24:                                        ; preds = %for.inc22, %entry
  %sum.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %sum.1.lcssa, %for.inc22 ]
  %16 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #3, !range !8
  %17 = shl nuw nsw i32 %16, 4
  %18 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #3, !range !9
  %narrow = add nuw nsw i32 %17, %18
  %add28 = zext i32 %narrow to i64
  %mul29 = shl nuw nsw i64 %add28, 12
  %19 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3, !range !8
  %20 = shl nuw nsw i32 %19, 4
  %21 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !9
  %narrow57 = add nuw nsw i32 %20, %21
  %add33 = zext i32 %narrow57 to i64
  %add34 = add nuw nsw i64 %mul29, %add33
  %arrayidx35 = getelementptr inbounds float, float addrspace(1)* %C, i64 %add34
  store float %sum.0.lcssa, float addrspace(1)* %arrayidx35, align 4, !tbaa !10
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

!0 = !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32, i32, i32, i32)* @conv, !"kernel", i32 1}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 2, i32 0}
!3 = !{!"clang version 12.0.0"}
!4 = !{i32 1, i32 1, i32 1, i32 0, i32 0, i32 0, i32 0}
!5 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!6 = !{!"float*", !"float*", !"float*", !"int", !"int", !"int", !"int"}
!7 = !{!"", !"", !"", !"", !"", !"", !""}
!8 = !{i32 0, i32 65535}
!9 = !{i32 0, i32 1024}
!10 = !{!11, !11, i64 0}
!11 = !{!"float", !12, i64 0}
!12 = !{!"omnipotent char", !13, i64 0}
!13 = !{!"Simple C/C++ TBAA"}
