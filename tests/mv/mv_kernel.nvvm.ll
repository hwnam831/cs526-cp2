; ModuleID = 'mv_kernel.nvvm.bc'
source_filename = "mv_kernel.cl"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-nvcl-nvidial"

; Function Attrs: nofree noinline norecurse nounwind
define dso_local spir_kernel void @mv(float addrspace(1)* nocapture readonly %A, float addrspace(1)* nocapture readonly %B, float addrspace(1)* nocapture %C, i32 %width) local_unnamed_addr #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3, !range !8
  %1 = shl nuw nsw i32 %0, 8
  %2 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !9
  %narrow23 = add nuw nsw i32 %1, %2
  %add = zext i32 %narrow23 to i64
  %mul2 = shl nuw nsw i64 %add, 11
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.025 = phi i32 [ 0, %entry ], [ %add6.1, %for.body ]
  %sum.024 = phi float [ 0.000000e+00, %entry ], [ %8, %for.body ]
  %conv = zext i32 %i.025 to i64
  %add3 = add nuw nsw i64 %mul2, %conv
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %A, i64 %add3
  %3 = load float, float addrspace(1)* %arrayidx, align 4, !tbaa !10
  %arrayidx4 = getelementptr inbounds float, float addrspace(1)* %B, i64 %conv
  %4 = load float, float addrspace(1)* %arrayidx4, align 4, !tbaa !10
  %5 = tail call float @llvm.fmuladd.f32(float %3, float %4, float %sum.024)
  %add6 = or i32 %i.025, 1
  %conv.1 = zext i32 %add6 to i64
  %add3.1 = add nuw nsw i64 %mul2, %conv.1
  %arrayidx.1 = getelementptr inbounds float, float addrspace(1)* %A, i64 %add3.1
  %6 = load float, float addrspace(1)* %arrayidx.1, align 4, !tbaa !10
  %arrayidx4.1 = getelementptr inbounds float, float addrspace(1)* %B, i64 %conv.1
  %7 = load float, float addrspace(1)* %arrayidx4.1, align 4, !tbaa !10
  %8 = tail call float @llvm.fmuladd.f32(float %6, float %7, float %5)
  %add6.1 = add nuw nsw i32 %i.025, 2
  %exitcond.not.1 = icmp eq i32 %add6.1, 2048
  br i1 %exitcond.not.1, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %arrayidx11 = getelementptr inbounds float, float addrspace(1)* %C, i64 %add
  store float %8, float addrspace(1)* %arrayidx11, align 4, !tbaa !10
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

!0 = !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @mv, !"kernel", i32 1}
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
