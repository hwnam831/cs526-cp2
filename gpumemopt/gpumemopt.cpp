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

#define DEBUG_TYPE "scalarrepl"
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
using namespace llvm;

STATISTIC(NumReplaced,  "Number of aggregate allocas broken up");
STATISTIC(NumPromoted,  "Number of scalar allocas promoted to register");

namespace {
  struct GPUMemCoalescing : public FunctionPass {
    static char ID; // Pass identification
    GPUMemCoalescing() : FunctionPass(ID) { }

    // Entry point for the overall scalar-replacement pass
    bool runOnFunction(Function &F);

    // getAnalysisUsage - List passes required by this pass.  We also know it
    // will not alter the CFG, so say so.
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<LoopInfoWrapperPass>();
    }

  private:
    // Add fields and helper functions for this pass here.
  };
}

char GPUMemCoalescing::ID = 0;
static RegisterPass<GPUMemCoalescing> X("gpumemcoal",
			    "GPU Memory coalescing pass",
			    false /* does not modify the CFG */,
			    false /* transformation, not just analysis */);

bool AnalyzeIncrement(Value* base, ValueMap<Value*, int>& map, int incr){
  bool isChanged = false;
  base->dump();
  errs() << "Analyzed increment : " << incr << "\n\n";
  for (auto U: base->users()){
    if (map.count(U)){
      continue;
    }
    if (CAST(BinaryOperator,BO,U)) {
      switch (BO->getOpcode()){
        case Instruction::BinaryOps::Add : {
          map.insert(std::make_pair(U, incr));
          isChanged = true;
          AnalyzeIncrement(U, map, incr);
          break;}
        case Instruction::BinaryOps::Sub : {
          //TODO: What if it is a subtraction base?
          if (BO->getOperand(0) == base) {
            map.insert(std::make_pair(U, incr));
            isChanged = true;
            AnalyzeIncrement(U, map, incr);
          } else {
            map.insert(std::make_pair(U, -incr));
            isChanged = true;
            AnalyzeIncrement(U, map, -incr);
          }
          break;}
        case Instruction::BinaryOps::Mul: {
          Value *op1 = BO->getOperand(0);
          Value *op2 = BO->getOperand(1);
          int multiplier;
          if(isa<ConstantInt>(op1)){
            multiplier = dyn_cast<ConstantInt>(op1)->getLimitedValue();
            map.insert(std::make_pair(U, incr*multiplier));
            isChanged = true;
            AnalyzeIncrement(U, map, incr*multiplier);
          } else if(isa<ConstantInt>(op2)){
            multiplier = dyn_cast<ConstantInt>(op2)->getSExtValue();
            map.insert(std::make_pair(U, incr*multiplier));
            isChanged = true;
            AnalyzeIncrement(U, map, incr*multiplier);
          }
          break;}
        default: {
          break;}
      }
    } else if(isa<SExtInst>(U)){
      map.insert(std::make_pair(U, incr));
      isChanged = true;
      AnalyzeIncrement(U, map, incr);
    } else if(isa<ZExtInst>(U)){
      map.insert(std::make_pair(U, incr));
      isChanged = true;
      AnalyzeIncrement(U, map, incr);
    }
    
  }
  return isChanged;
}

//===----------------------------------------------------------------------===//
//                      SKELETON FUNCTION TO BE IMPLEMENTED
//===----------------------------------------------------------------------===//
//
// Function runOnFunction:
// Entry point for the overall ScalarReplAggregates function pass.
// This function is provided to you.
bool GPUMemCoalescing::runOnFunction(Function &F) {
  ValueMap<Value*, int> IVincr;
  ValueMap<Value*, int> TIDincr;
  for (BasicBlock& block : F){
    LoopInfo& loops = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    //loops.print(errs());
    for (auto& loop : loops){
      auto IV = loop->getCanonicalInductionVariable();
      if (IVincr.count(IV) == 0){
        IVincr.insert(std::make_pair(IV, 1));
        errs() << "Found induction variable\n";
        AnalyzeIncrement(IV, IVincr, 1);
      }
    }
    for (Instruction& inst : block){
      if (CallInst *CI = dyn_cast<CallInst>(&inst)){
        //CI->dump();
        auto funcname = CI->getCalledFunction()->getName();
        //errs() << funcname << "\n";
        if(funcname == "llvm.nvvm.read.ptx.sreg.tid.x"){
          
          if (TIDincr.count(CI) == 0){
            TIDincr.insert(std::make_pair(CI, 1));
            errs() << "Found tid.x\n";
            AnalyzeIncrement(CI, TIDincr, 1);
          }
        }
      } 
    }
  }
  bool Changed = false;
  return Changed;

}

