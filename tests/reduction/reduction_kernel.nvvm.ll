; ModuleID = 'reduction_kernel.nvvm.bc'
source_filename = "reduction_kernel.cl"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-nvcl-nvidial"

; Function Attrs: nofree noinline norecurse nounwind willreturn
define dso_local spir_kernel void @reduction(float addrspace(1)* nocapture %d_odata, float addrspace(1)* nocapture readonly %d_idata, i32 %num_elements) local_unnamed_addr #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #2, !range !8
  %1 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #2, !range !9
  %narrow = mul nuw i32 %1, %0
  %mul = zext i32 %narrow to i64
  %2 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2, !range !8
  %retval.0.i32 = zext i32 %2 to i64
  %add = add nuw nsw i64 %mul, %retval.0.i32
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2, !range !10
  %retval.0.i31 = zext i32 %3 to i64
  %mul4 = mul nuw nsw i64 %add, %retval.0.i31
  %4 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2, !range !11
  %retval.0.i30 = zext i32 %4 to i64
  %add6 = add nuw nsw i64 %mul4, %retval.0.i30
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %d_idata, i64 %add6
  %5 = load float, float addrspace(1)* %arrayidx, align 4, !tbaa !12
  %div = sdiv i32 %num_elements, 2
  %conv = sext i32 %div to i64
  %add16 = add nsw i64 %add6, %conv
  %arrayidx17 = getelementptr inbounds float, float addrspace(1)* %d_idata, i64 %add16
  %6 = load float, float addrspace(1)* %arrayidx17, align 4, !tbaa !12
  %add18 = fadd float %5, %6
  %arrayidx28 = getelementptr inbounds float, float addrspace(1)* %d_odata, i64 %add6
  store float %add18, float addrspace(1)* %arrayidx28, align 4, !tbaa !12
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #1

attributes #0 = { nofree noinline norecurse nounwind willreturn "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+ptx32,+sm_20" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!nvvm.annotations = !{!0}
!llvm.module.flags = !{!1}
!opencl.ocl.version = !{!2}
!llvm.ident = !{!3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3}

!0 = !{void (float addrspace(1)*, float addrspace(1)*, i32)* @reduction, !"kernel", i32 1}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 2, i32 0}
!3 = !{!"clang version 12.0.0"}
!4 = !{i32 1, i32 1, i32 0}
!5 = !{!"none", !"none", !"none"}
!6 = !{!"float*", !"float*", !"int"}
!7 = !{!"", !"", !""}
!8 = !{i32 0, i32 65535}
!9 = !{i32 1, i32 65536}
!10 = !{i32 1, i32 1025}
!11 = !{i32 0, i32 1024}
!12 = !{!13, !13, i64 0}
!13 = !{!"float", !14, i64 0}
!14 = !{!"omnipotent char", !15, i64 0}
!15 = !{!"Simple C/C++ TBAA"}
