; ModuleID = 'tr_kernel.nvvm.bc'
source_filename = "tr_kernel.cl"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-nvcl-nvidial"

; Function Attrs: nofree noinline norecurse nounwind willreturn
define dso_local spir_kernel void @tr(float addrspace(1)* nocapture readonly %A, float addrspace(1)* nocapture %C, i32 %width) local_unnamed_addr #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2, !range !8
  %1 = shl nuw nsw i32 %0, 4
  %2 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2, !range !9
  %narrow = add nuw nsw i32 %1, %2
  %add = zext i32 %narrow to i64
  %mul2 = shl nuw nsw i64 %add, 11
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #2, !range !8
  %4 = shl nuw nsw i32 %3, 4
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #2, !range !9
  %narrow24 = add nuw nsw i32 %4, %5
  %add6 = zext i32 %narrow24 to i64
  %add7 = add nuw nsw i64 %mul2, %add6
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %A, i64 %add7
  %6 = load float, float addrspace(1)* %arrayidx, align 4, !tbaa !10
  %mul12 = shl nuw nsw i64 %add6, 11
  %add17 = add nuw nsw i64 %mul12, %add
  %arrayidx18 = getelementptr inbounds float, float addrspace(1)* %C, i64 %add17
  store float %6, float addrspace(1)* %arrayidx18, align 4, !tbaa !10
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y() #1

attributes #0 = { nofree noinline norecurse nounwind willreturn "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+ptx32,+sm_20" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!nvvm.annotations = !{!0}
!llvm.module.flags = !{!1}
!opencl.ocl.version = !{!2}
!llvm.ident = !{!3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3}

!0 = !{void (float addrspace(1)*, float addrspace(1)*, i32)* @tr, !"kernel", i32 1}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 2, i32 0}
!3 = !{!"clang version 12.0.0"}
!4 = !{i32 1, i32 1, i32 0}
!5 = !{!"none", !"none", !"none"}
!6 = !{!"float*", !"float*", !"int"}
!7 = !{!"", !"", !""}
!8 = !{i32 0, i32 65535}
!9 = !{i32 0, i32 1024}
!10 = !{!11, !11, i64 0}
!11 = !{!"float", !12, i64 0}
!12 = !{!"omnipotent char", !13, i64 0}
!13 = !{!"Simple C/C++ TBAA"}
