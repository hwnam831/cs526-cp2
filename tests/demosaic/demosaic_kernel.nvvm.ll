; ModuleID = 'demosaic_kernel.nvvm.bc'
source_filename = "demosaic_kernel.cl"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-nvcl-nvidial"

; Function Attrs: norecurse nounwind readonly willreturn
define dso_local float @cal(float* nocapture readonly %temp) local_unnamed_addr #0 {
entry:
  %arrayidx = getelementptr inbounds float, float* %temp, i64 4
  %0 = load float, float* %arrayidx, align 4, !tbaa !4
  %conv = fpext float %0 to double
  %arrayidx1 = getelementptr inbounds float, float* %temp, i64 1
  %1 = load float, float* %arrayidx1, align 4, !tbaa !4
  %arrayidx2 = getelementptr inbounds float, float* %temp, i64 3
  %2 = load float, float* %arrayidx2, align 4, !tbaa !4
  %add = fadd float %1, %2
  %arrayidx3 = getelementptr inbounds float, float* %temp, i64 5
  %3 = load float, float* %arrayidx3, align 4, !tbaa !4
  %add4 = fadd float %add, %3
  %arrayidx5 = getelementptr inbounds float, float* %temp, i64 7
  %4 = load float, float* %arrayidx5, align 4, !tbaa !4
  %add6 = fadd float %add4, %4
  %conv7 = fpext float %add6 to double
  %5 = tail call double @llvm.fmuladd.f64(double %conv7, double 2.500000e-01, double %conv)
  %conv8 = fptrunc double %5 to float
  ret float %conv8
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare double @llvm.fmuladd.f64(double, double, double) #1

; Function Attrs: nofree noinline norecurse nounwind
define dso_local spir_kernel void @demosaic(float addrspace(1)* nocapture readonly %A, float addrspace(1)* nocapture %C, i32 %width) local_unnamed_addr #2 !kernel_arg_addr_space !8 !kernel_arg_access_qual !9 !kernel_arg_type !10 !kernel_arg_base_type !10 !kernel_arg_type_qual !11 {
entry:
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #4, !range !12
  %1 = shl nuw nsw i32 %0, 4
  %2 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #4, !range !13
  %narrow50 = add nuw nsw i32 %2, 16
  %narrow51 = add nuw nsw i32 %narrow50, %1
  %add5 = zext i32 %narrow51 to i64
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !range !12
  %4 = shl nuw nsw i32 %3, 4
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !range !13
  %narrow52 = add nuw nsw i32 %5, 16
  %narrow53 = add nuw nsw i32 %narrow52, %4
  %add11 = zext i32 %narrow53 to i64
  %mul6 = mul nuw nsw i64 %add5, 2064
  %sub13.1 = add nsw i64 %add11, -1
  %add14.1 = add nuw nsw i64 %sub13.1, %mul6
  %arrayidx.1 = getelementptr inbounds float, float addrspace(1)* %A, i64 %add14.1
  %6 = load float, float addrspace(1)* %arrayidx.1, align 4, !tbaa !4
  %sub13.2 = add nsw i64 %add11, -2
  %7 = mul nuw nsw i64 %add5, 2064
  %mul6.1 = add nsw i64 %7, -2064
  %add14.158 = add nuw nsw i64 %mul6.1, %add11
  %arrayidx.159 = getelementptr inbounds float, float addrspace(1)* %A, i64 %add14.158
  %8 = load float, float addrspace(1)* %arrayidx.159, align 4, !tbaa !4
  %add14.1.1 = add nuw nsw i64 %sub13.1, %mul6.1
  %arrayidx.1.1 = getelementptr inbounds float, float addrspace(1)* %A, i64 %add14.1.1
  %9 = load float, float addrspace(1)* %arrayidx.1.1, align 4, !tbaa !4
  %add14.2.1 = add nuw nsw i64 %sub13.2, %mul6.1
  %arrayidx.2.1 = getelementptr inbounds float, float addrspace(1)* %A, i64 %add14.2.1
  %10 = load float, float addrspace(1)* %arrayidx.2.1, align 4, !tbaa !4
  %11 = mul nuw nsw i64 %add5, 2064
  %mul6.2 = add nsw i64 %11, -4128
  %add14.1.2 = add nuw nsw i64 %sub13.1, %mul6.2
  %arrayidx.1.2 = getelementptr inbounds float, float addrspace(1)* %A, i64 %add14.1.2
  %12 = load float, float addrspace(1)* %arrayidx.1.2, align 4, !tbaa !4
  %conv.i = fpext float %9 to double
  %add.i = fadd float %6, %8
  %add4.i = fadd float %add.i, %10
  %add6.i = fadd float %add4.i, %12
  %conv7.i = fpext float %add6.i to double
  %13 = tail call double @llvm.fmuladd.f64(double %conv7.i, double 2.500000e-01, double %conv.i) #4
  %conv8.i = fptrunc double %13 to float
  %narrow = add nuw nsw i32 %1, %2
  %14 = shl nuw i32 %narrow, 11
  %mul26 = zext i32 %14 to i64
  %narrow49 = add nuw nsw i32 %4, %5
  %add30 = zext i32 %narrow49 to i64
  %add31 = add nuw nsw i64 %add30, %mul26
  %arrayidx32 = getelementptr inbounds float, float addrspace(1)* %C, i64 %add31
  store float %conv8.i, float addrspace(1)* %arrayidx32, align 4, !tbaa !4
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #3

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y() #3

attributes #0 = { norecurse nounwind readonly willreturn "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+ptx32,+sm_20" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { nofree noinline norecurse nounwind "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+ptx32,+sm_20" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone }
attributes #4 = { nounwind }

!nvvm.annotations = !{!0}
!llvm.module.flags = !{!1}
!opencl.ocl.version = !{!2}
!llvm.ident = !{!3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3}

!0 = !{void (float addrspace(1)*, float addrspace(1)*, i32)* @demosaic, !"kernel", i32 1}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 2, i32 0}
!3 = !{!"clang version 12.0.0"}
!4 = !{!5, !5, i64 0}
!5 = !{!"float", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = !{i32 1, i32 1, i32 0}
!9 = !{!"none", !"none", !"none"}
!10 = !{!"float*", !"float*", !"int"}
!11 = !{!"", !"", !""}
!12 = !{i32 0, i32 65535}
!13 = !{i32 0, i32 1024}
