; ModuleID = 'conv_kernel.nvvm.bc'
source_filename = "conv_kernel.cl"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-nvcl-nvidial"

; Function Attrs: nofree noinline norecurse nounwind
define dso_local spir_kernel void @conv(float addrspace(1)* nocapture readonly %A, float addrspace(1)* nocapture readonly %B, float addrspace(1)* nocapture %C, i32 %width, i32 %height, i32 %w, i32 %h) local_unnamed_addr #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #3, !range !8
  %1 = shl nuw nsw i32 %0, 4
  %2 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #3, !range !9
  %narrow56 = add nuw nsw i32 %1, %2
  %add = zext i32 %narrow56 to i64
  %conv5 = sext i32 %h to i64
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3, !range !8
  %4 = shl nuw nsw i32 %3, 4
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !9
  %narrow57 = add nuw nsw i32 %4, %5
  %add11 = zext i32 %narrow57 to i64
  %conv14 = sext i32 %w to i64
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %entry, %for.inc22
  %j.061 = phi i32 [ 0, %entry ], [ %add23, %for.inc22 ]
  %sum.060 = phi float [ 0.000000e+00, %entry ], [ %11, %for.inc22 ]
  %conv = zext i32 %j.061 to i64
  %sub = sub nsw i64 %conv5, %conv
  %add6 = add nsw i64 %sub, %add
  %mul7 = mul nsw i64 %add6, 4112
  %mul17 = shl nsw i32 %j.061, 4
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.cond1.preheader
  %i.059 = phi i32 [ 0, %for.cond1.preheader ], [ %add21.1, %for.body3 ]
  %sum.158 = phi float [ %sum.060, %for.cond1.preheader ], [ %11, %for.body3 ]
  %conv12 = zext i32 %i.059 to i64
  %sub13 = sub nsw i64 %conv14, %conv12
  %add15 = add nsw i64 %sub13, %add11
  %add16 = add nsw i64 %add15, %mul7
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %A, i64 %add16
  %6 = load float, float addrspace(1)* %arrayidx, align 4, !tbaa !10
  %add18 = add nuw nsw i32 %i.059, %mul17
  %idxprom = zext i32 %add18 to i64
  %arrayidx19 = getelementptr inbounds float, float addrspace(1)* %B, i64 %idxprom
  %7 = load float, float addrspace(1)* %arrayidx19, align 4, !tbaa !10
  %8 = tail call float @llvm.fmuladd.f32(float %6, float %7, float %sum.158)
  %add21 = or i32 %i.059, 1
  %conv12.1 = zext i32 %add21 to i64
  %sub13.1 = sub nsw i64 %conv14, %conv12.1
  %add15.1 = add nsw i64 %sub13.1, %add11
  %add16.1 = add nsw i64 %add15.1, %mul7
  %arrayidx.1 = getelementptr inbounds float, float addrspace(1)* %A, i64 %add16.1
  %9 = load float, float addrspace(1)* %arrayidx.1, align 4, !tbaa !10
  %add18.1 = add nuw nsw i32 %add21, %mul17
  %idxprom.1 = zext i32 %add18.1 to i64
  %arrayidx19.1 = getelementptr inbounds float, float addrspace(1)* %B, i64 %idxprom.1
  %10 = load float, float addrspace(1)* %arrayidx19.1, align 4, !tbaa !10
  %11 = tail call float @llvm.fmuladd.f32(float %9, float %10, float %8)
  %add21.1 = add nuw nsw i32 %i.059, 2
  %exitcond.not.1 = icmp eq i32 %add21.1, 16
  br i1 %exitcond.not.1, label %for.inc22, label %for.body3

for.inc22:                                        ; preds = %for.body3
  %add23 = add nuw nsw i32 %j.061, 1
  %exitcond62.not = icmp eq i32 %add23, 16
  br i1 %exitcond62.not, label %for.end24, label %for.cond1.preheader

for.end24:                                        ; preds = %for.inc22
  %mul29 = shl nuw nsw i64 %add, 12
  %add34 = add nuw nsw i64 %mul29, %add11
  %arrayidx35 = getelementptr inbounds float, float addrspace(1)* %C, i64 %add34
  store float %11, float addrspace(1)* %arrayidx35, align 4, !tbaa !10
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
