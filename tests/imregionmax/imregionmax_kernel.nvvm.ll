; ModuleID = 'imregionmax_kernel.nvvm.bc'
source_filename = "imregionmax_kernel.cl"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-nvcl-nvidial"

; Function Attrs: norecurse nounwind readonly willreturn
define dso_local float @cal(float* nocapture readonly %temp) local_unnamed_addr #0 {
entry:
  %0 = load float, float* %temp, align 4, !tbaa !4
  %arrayidx1 = getelementptr inbounds float, float* %temp, i64 1
  %1 = load float, float* %arrayidx1, align 4, !tbaa !4
  %cmp = fcmp ogt float %0, %1
  %cond = select i1 %cmp, float %0, float %1
  %arrayidx4 = getelementptr inbounds float, float* %temp, i64 2
  %2 = load float, float* %arrayidx4, align 4, !tbaa !4
  %cmp5 = fcmp ogt float %cond, %2
  %cond10 = select i1 %cmp5, float %cond, float %2
  %arrayidx11 = getelementptr inbounds float, float* %temp, i64 3
  %3 = load float, float* %arrayidx11, align 4, !tbaa !4
  %cmp12 = fcmp ogt float %cond10, %3
  %cond17 = select i1 %cmp12, float %cond10, float %3
  %arrayidx18 = getelementptr inbounds float, float* %temp, i64 5
  %4 = load float, float* %arrayidx18, align 4, !tbaa !4
  %cmp19 = fcmp ogt float %cond17, %4
  %cond24 = select i1 %cmp19, float %cond17, float %4
  %arrayidx25 = getelementptr inbounds float, float* %temp, i64 6
  %5 = load float, float* %arrayidx25, align 4, !tbaa !4
  %cmp26 = fcmp ogt float %cond24, %5
  %cond31 = select i1 %cmp26, float %cond24, float %5
  %arrayidx32 = getelementptr inbounds float, float* %temp, i64 7
  %6 = load float, float* %arrayidx32, align 4, !tbaa !4
  %cmp33 = fcmp ogt float %cond31, %6
  %cond38 = select i1 %cmp33, float %cond31, float %6
  %arrayidx39 = getelementptr inbounds float, float* %temp, i64 8
  %7 = load float, float* %arrayidx39, align 4, !tbaa !4
  %cmp40 = fcmp ogt float %cond38, %7
  %cond45 = select i1 %cmp40, float %cond38, float %7
  %arrayidx46 = getelementptr inbounds float, float* %temp, i64 4
  %8 = load float, float* %arrayidx46, align 4, !tbaa !4
  %cmp47 = fcmp ule float %cond45, %8
  %conv = uitofp i1 %cmp47 to float
  ret float %conv
}

; Function Attrs: nofree noinline norecurse nounwind
define dso_local spir_kernel void @imregionmax(float addrspace(1)* nocapture readonly %A, float addrspace(1)* nocapture %C, i32 %width) local_unnamed_addr #1 !kernel_arg_addr_space !8 !kernel_arg_access_qual !9 !kernel_arg_type !10 !kernel_arg_base_type !10 !kernel_arg_type_qual !11 {
entry:
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #3, !range !12
  %1 = shl nuw nsw i32 %0, 4
  %2 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #3, !range !13
  %narrow50 = add nuw nsw i32 %2, 16
  %narrow51 = add nuw nsw i32 %narrow50, %1
  %add5 = zext i32 %narrow51 to i64
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3, !range !12
  %4 = shl nuw nsw i32 %3, 4
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3, !range !13
  %narrow52 = add nuw nsw i32 %5, 16
  %narrow53 = add nuw nsw i32 %narrow52, %4
  %add11 = zext i32 %narrow53 to i64
  %mul6 = mul nuw nsw i64 %add5, 2064
  %add14 = add nuw nsw i64 %mul6, %add11
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %A, i64 %add14
  %6 = load float, float addrspace(1)* %arrayidx, align 4, !tbaa !4
  %sub13.1 = add nsw i64 %add11, -1
  %add14.1 = add nuw nsw i64 %sub13.1, %mul6
  %arrayidx.1 = getelementptr inbounds float, float addrspace(1)* %A, i64 %add14.1
  %7 = load float, float addrspace(1)* %arrayidx.1, align 4, !tbaa !4
  %sub13.2 = add nsw i64 %add11, -2
  %add14.2 = add nuw nsw i64 %sub13.2, %mul6
  %arrayidx.2 = getelementptr inbounds float, float addrspace(1)* %A, i64 %add14.2
  %8 = load float, float addrspace(1)* %arrayidx.2, align 4, !tbaa !4
  %9 = mul nuw nsw i64 %add5, 2064
  %mul6.1 = add nsw i64 %9, -2064
  %add14.158 = add nuw nsw i64 %mul6.1, %add11
  %arrayidx.159 = getelementptr inbounds float, float addrspace(1)* %A, i64 %add14.158
  %10 = load float, float addrspace(1)* %arrayidx.159, align 4, !tbaa !4
  %add14.1.1 = add nuw nsw i64 %sub13.1, %mul6.1
  %arrayidx.1.1 = getelementptr inbounds float, float addrspace(1)* %A, i64 %add14.1.1
  %11 = load float, float addrspace(1)* %arrayidx.1.1, align 4, !tbaa !4
  %add14.2.1 = add nuw nsw i64 %sub13.2, %mul6.1
  %arrayidx.2.1 = getelementptr inbounds float, float addrspace(1)* %A, i64 %add14.2.1
  %12 = load float, float addrspace(1)* %arrayidx.2.1, align 4, !tbaa !4
  %13 = mul nuw nsw i64 %add5, 2064
  %mul6.2 = add nsw i64 %13, -4128
  %add14.263 = add nuw nsw i64 %mul6.2, %add11
  %arrayidx.264 = getelementptr inbounds float, float addrspace(1)* %A, i64 %add14.263
  %14 = load float, float addrspace(1)* %arrayidx.264, align 4, !tbaa !4
  %add14.1.2 = add nuw nsw i64 %sub13.1, %mul6.2
  %arrayidx.1.2 = getelementptr inbounds float, float addrspace(1)* %A, i64 %add14.1.2
  %15 = load float, float addrspace(1)* %arrayidx.1.2, align 4, !tbaa !4
  %add14.2.2 = add nuw nsw i64 %sub13.2, %mul6.2
  %arrayidx.2.2 = getelementptr inbounds float, float addrspace(1)* %A, i64 %add14.2.2
  %16 = load float, float addrspace(1)* %arrayidx.2.2, align 4, !tbaa !4
  %cmp.i = fcmp ogt float %6, %7
  %cond.i = select i1 %cmp.i, float %6, float %7
  %cmp5.i = fcmp ogt float %cond.i, %8
  %cond10.i = select i1 %cmp5.i, float %cond.i, float %8
  %cmp12.i = fcmp ogt float %cond10.i, %10
  %cond17.i = select i1 %cmp12.i, float %cond10.i, float %10
  %cmp19.i = fcmp ogt float %cond17.i, %12
  %cond24.i = select i1 %cmp19.i, float %cond17.i, float %12
  %cmp26.i = fcmp ogt float %cond24.i, %14
  %cond31.i = select i1 %cmp26.i, float %cond24.i, float %14
  %cmp33.i = fcmp ogt float %cond31.i, %15
  %cond38.i = select i1 %cmp33.i, float %cond31.i, float %15
  %cmp40.i = fcmp ogt float %cond38.i, %16
  %cond45.i = select i1 %cmp40.i, float %cond38.i, float %16
  %cmp47.i = fcmp ule float %cond45.i, %11
  %conv.i = uitofp i1 %cmp47.i to float
  %narrow = add nuw nsw i32 %1, %2
  %17 = shl nuw i32 %narrow, 11
  %mul26 = zext i32 %17 to i64
  %narrow49 = add nuw nsw i32 %4, %5
  %add30 = zext i32 %narrow49 to i64
  %add31 = add nuw nsw i64 %add30, %mul26
  %arrayidx32 = getelementptr inbounds float, float addrspace(1)* %C, i64 %add31
  store float %conv.i, float addrspace(1)* %arrayidx32, align 4, !tbaa !4
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #2

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y() #2

attributes #0 = { norecurse nounwind readonly willreturn "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+ptx32,+sm_20" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nofree noinline norecurse nounwind "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+ptx32,+sm_20" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!nvvm.annotations = !{!0}
!llvm.module.flags = !{!1}
!opencl.ocl.version = !{!2}
!llvm.ident = !{!3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3}

!0 = !{void (float addrspace(1)*, float addrspace(1)*, i32)* @imregionmax, !"kernel", i32 1}
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
