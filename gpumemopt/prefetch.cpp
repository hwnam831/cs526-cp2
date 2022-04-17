//===- ScalarReplAggregates.cpp - Scalar Replacement of Aggregates --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This transformation implements the well known scalar replacement of
// aggregates transformation.  This xform breaks up alloca instructions of
// structure type into individual alloca instructions for
// each member (if possible).  Then, if possible, it transforms the individual
// alloca instructions into nice clean scalar SSA form.
//
// This combines an SRoA algorithm with Mem2Reg because they
// often interact, especially for C++ programs.  As such, this code
// iterates between SRoA and Mem2Reg until we run out of things to promote.
//
//===----------------------------------------------------------------------===//

#define CAST(T,N,U) T *N = dyn_cast<T>(U)
#include <iostream>
#include "llvm/Transforms/Scalar.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
using namespace llvm;

namespace {
  struct GPUMemPrefetching : public FunctionPass {
    static char ID; // Pass identification
    GPUMemPrefetching() : FunctionPass(ID) { }

    // Entry point for the overall scalar-replacement pass
    bool runOnFunction(Function &F);
    bool runOnLoop(Loop *L);
    const SCEVAddRecExpr *findAddRecExpr(const SCEVNAryExpr * expr);
    void findInductionVariables(Loop *L);
    // getAnalysisUsage - List passes required by this pass.  We also know it
    // will not alter the CFG, so say so.
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<LoopInfoWrapperPass>();
      AU.addRequired<ScalarEvolutionWrapperPass>();
      AU.addPreserved<ScalarEvolutionWrapperPass>();
    }

  private:
    // Add fields and helper functions for this pass here.
    ScalarEvolution *SE;
  };
}

char GPUMemPrefetching::ID = 0;
static RegisterPass<GPUMemPrefetching> X("gpumempref",
			    "GPU Memory prefetching pass",
			    false /* does not modify the CFG */,
			    false /* transformation, not just analysis */);


//===----------------------------------------------------------------------===//
//                      SKELETON FUNCTION TO BE IMPLEMENTED
//===----------------------------------------------------------------------===//
//
// Function runOnFunction:
// Entry point for the overall GPUMemPrefetching function pass.
// This function is provided to you.

const SCEVAddRecExpr *GPUMemPrefetching::findAddRecExpr(const SCEVNAryExpr * expr){
  // expr->dump();
  if(!isa<SCEVAddRecExpr>(expr)){
    for (unsigned i = 0; i < expr->getNumOperands(); ++i){
      if(!isa<SCEVNAryExpr>(expr->getOperand(i)))
        continue;

      const SCEVNAryExpr * SCEVNAry_expr = dyn_cast<SCEVNAryExpr>(expr->getOperand(i));
      const SCEVAddRecExpr * sub_expr = findAddRecExpr(SCEVNAry_expr);
      if(sub_expr != nullptr)
        return sub_expr;
    }
    return nullptr;
  } else {
    const SCEVAddRecExpr *LSCEAddRec_expr = dyn_cast<SCEVAddRecExpr>(expr);
    return LSCEAddRec_expr;
  }
}

void GPUMemPrefetching::findInductionVariables(Loop *L){
  BasicBlock *H = L->getHeader();
  BasicBlock *Incoming = nullptr, *Backedge = nullptr;
  if (!L->getIncomingAndBackEdge(Incoming, Backedge))
    return;  

  // Loop over all of the PHI nodes, looking for induction variables.
  for (BasicBlock::iterator I = H->begin(); I != H->end(); ++I) {
    if(!isa<PHINode>(I))
      continue;
    PHINode *PN = cast<PHINode>(I);
    Value * back = PN->getIncomingValueForBlock(Backedge);
    if(!SE->isSCEVable(back->getType()))
      continue;

    const SCEV *LSCEV_back = SE->getSCEV(PN);

    if(!isa<SCEVAddRecExpr>(LSCEV_back))
      errs() << "not supported stride type\n";
    else {
      const SCEVAddRecExpr *LSCEAddRec_back = dyn_cast<SCEVAddRecExpr>(LSCEV_back);
      LSCEAddRec_back->dump();
      for (unsigned i = 0, e = LSCEAddRec_back->getNumOperands(); i != e; ++i)
        outs() << *LSCEAddRec_back->getOperand(i);

      PN->dump();
      back->dump();
    }
    //ConstantInt *CI = dyn_cast<ConstantInt>(PN->getIncomingValueForBlock(Incoming))
  }
}

bool GPUMemPrefetching::runOnLoop(Loop *L) {
outs() << "start-------------------------------------\n";

  bool Changed = false;
  findInductionVariables(L); // not used for now...

outs() << "-------------------------------------\n";

  for (const auto BB : L->blocks()) {
    for (auto &I : *BB) {
      Value *PtrOp;
      Value *LPtrOp;
      Value *ValOp;
      Value *LValOp;
      Instruction *MemI;

      // only prefetch for stores from global memory to shared memory
      if (StoreInst *SMemI = dyn_cast<StoreInst>(&I)) {
        MemI = SMemI;
        PtrOp = SMemI->getPointerOperand();
        ValOp = SMemI->getValueOperand();
        SMemI->dump();
        unsigned PtrAddrSpace = PtrOp->getType()->getPointerAddressSpace();
        PtrOp->dump();
        outs() << "store ptr addr space: " << PtrAddrSpace << "\n";
        // shared memory access address space 3
        if (PtrAddrSpace != 3)
          continue;
      } else continue;
      
      if (LoadInst *LMemI = dyn_cast<LoadInst>(ValOp)) {
        LMemI->dump();
        LPtrOp = LMemI->getPointerOperand();
        LPtrOp->dump();
        unsigned ValAddrSpace = LPtrOp->getType()->getPointerAddressSpace();
        outs() << "value ptr addr space: " << ValAddrSpace << "\n";
        // global memory access address space 1
        if (ValAddrSpace != 1)
          continue;
      } else {
        errs() << "store value not from a load\n";
        continue;
      }

      outs() << "adding prefetch\n";

      const SCEV *LSCEV = SE->getSCEV(LPtrOp);
      LSCEV->dump();
      const SCEVNAryExpr  *LSCEVAddRec = dyn_cast<SCEVNAryExpr>(LSCEV);
      if(LSCEVAddRec != nullptr){
        outs() << "finding addrecs\n";
        const SCEVAddRecExpr *expr = findAddRecExpr(LSCEVAddRec);
        if(expr != nullptr){
          expr->dump();
          IRBuilder Builder(MemI);
          Value *prefAddr = Builder.CreateAdd(LPtrOp, ConstantInt::get(expr->getStepRecurrence(*SE)->getType(), expr->getOperand(1), true));
          const SCEV *NextLSCEV = SE->getAddExpr(LSCEVAddRec, expr->getStepRecurrence(*SE));
          Type *I8Ptr = Type::getInt8PtrTy(BB->getContext(), 0/*PtrAddrSpace*/);
          SCEVExpander SCEVE(*SE, BB->getModule()->getDataLayout(), "prefaddr");
          Value *PrefPtrValue = SCEVE.expandCodeFor(NextLSCEV, I8Ptr, MemI);
        } else {
          outs() << ("finding nullptr\n");
        }
      } else {
        continue;
      }

      // outs() << LSCEVAddRec;
      // LSCEVAddRec->dump();
      // outs() << LSCEVAddRec->getNumOperands() << "---\n";
      // LSCEVAddRec->getOperand(0)->dump();
      // const SCEVNAryExpr  *LSCEVAddRec2 = dyn_cast<SCEVNAryExpr>(LSCEVAddRec->getOperand(0));
      // LSCEVAddRec2->dump();
      // LSCEVAddRec2->getOperand(1)->dump();
      // outs() << "---\n";
      // const SCEVNAryExpr  *LSCEVAddRec3 = dyn_cast<SCEVNAryExpr>(LSCEVAddRec2->getOperand(1));
      // LSCEVAddRec3->dump();
      // LSCEVAddRec3->getOperand(2)->dump();

      // Type *I8Ptr = Type::getInt8PtrTy(BB->getContext(), 0/*PtrAddrSpace*/);
      // SCEVExpander SCEVE(*SE, BB->getModule()->getDataLayout(), "prefaddr");
      // outs() << "NULL? " << I8Ptr << "\n";
      // Value *PrefPtrValue = SCEVE.expandCodeFor(LSCEVAddRec3->getOperand(2), I8Ptr, MemI);

      // const SCEVAddExpr *LSCEVAddRec4 = dyn_cast<SCEVAddExpr>(LSCEVAddRec3->getOperand(2));
      // LSCEVAddRec4->dump();
      // const SCEVAddExpr *LSCEVAddRec = dyn_cast<SCEVAddExpr>(LSCEV);
      // if (!LSCEVAddRec){
      //   outs() << "not strided\n";
      //   continue;
      // } else {
      //   outs() << "strided\n";
      // }

      // BasicBlock *Latch = L->getLoopLatch();
      // if (BranchInst *BI = dyn_cast<BranchInst>(Latch->getTerminator())){
      //   if (!BI->isConditional()){
      //     for (Instruction& inst : *Latch){
      //       inst.dump();
      //       Value* op = inst.getOperand(0);
      //       op->dump();
      //     }
      //   }
      // }



      //BranchI->dump();

    }
  }
  outs() << "end-------------------------------------\n";

  return Changed;
}
bool GPUMemPrefetching::runOnFunction(Function &F) {
  outs() << "Start Prefetching.\n";

  LoopInfo *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();

  bool Changed = false;
  for (Loop *I : *LI)
    for (Loop *L : depth_first(I))
      Changed |= runOnLoop(L);

  return Changed;

}

