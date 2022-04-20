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
#include "utils.h"
#include <iostream>
#include "llvm/Transforms/Scalar.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
using namespace llvm;

STATISTIC(NumReplaced,  "Number of aggregate allocas broken up");
STATISTIC(NumPromoted,  "Number of scalar allocas promoted to register");

namespace {
  struct GPUMemCoalescing : public FunctionPass {
    static char ID; // Pass identification
    GPUMemCoalescing() : FunctionPass(ID) { }

    bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<LoopInfoWrapperPass>();
      AU.addRequired<DominatorTreeWrapperPass>();
    }

  private:
    // Add fields and helper functions for this pass here.
  };
}

char GPUMemCoalescing::ID = 0;
static RegisterPass<GPUMemCoalescing> X("gpumemcoal",
			    "GPU Memory coalescing pass",
			    true /* does not modify the CFG */,
			    false /* transformation, not just analysis */);

/**
 * Only works when within a for-loop
 * 1) Identify IV/TID and analyze increment coefficients -- done
 * 2) Identify their base address when IV/TID -- done
 * 3) Create tiled inner loop. Put everything inside.
 * 4) Outer IV -> IV*16, Replace all IV uses with IV*16+NewIV
 * 5) Global->shared load + syncthreads on outer loop. If segment size is 16 --> No loop. If 16x16, create loop
 * 
 */
bool GPUMemCoalescing::runOnFunction(Function &F) {
  ValueMap<Value*, int> TIDincr;

  for (BasicBlock& block : F){
    for (Instruction& inst : block){
      if (CAST(CallInst, CI, &inst)){
        //CI->dump();
        auto funcname = CI->getCalledFunction()->getName();
        //errs() << funcname << "\n";
        if(funcname == "llvm.nvvm.read.ptx.sreg.tid.x"){
          if (TIDincr.count(CI) == 0){
            // Zero marks that this is the exact TID.x
            TIDincr.insert(std::make_pair(CI, 0));
            errs() << "Found tid.x\n";
            AnalyzeIncrement(CI, TIDincr, 1);
          }
        }
      }
    }
  }

  LoopInfo& loops = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

  for (auto loop : loops){
    ValueMap<Value*, int> IVincr;
    std::vector<GetElementPtrInst*> Noncoalesced;
    ValueMap<GetElementPtrInst*, int> Segdims; //0,1, or 2
    ValueMap<GetElementPtrInst*, Value*> Baseaddr; //0,1, or 2
    auto IV = loop->getCanonicalInductionVariable();
    //IV->dump();
    
    //loop->addChildLoop
    
    //Zero marks that this is the exact IV
    IVincr.insert(std::make_pair(IV, 0));
    errs() << "Found induction variable\n";
    AnalyzeIncrement(IV, IVincr, 1);

    for(auto& B: loop->getBlocks()){
      PHINode* lastphi;
      for (Instruction& inst : *B){
        if (CAST(GetElementPtrInst, GEPI, &inst)){
          if(GEPI->getAddressSpace() == 1 &&
            GEPI->getNumIndices() == 1){
            GEPI->dump();
            auto IDX = GEPI->idx_begin()->get();

            int ivinc = IVincr.count(IDX) ? IVincr[IDX] : 0;
            int tidinc = TIDincr.count(IDX) ? TIDincr[IDX] : 0;
            
            if(ivinc % 64 == 0 && tidinc == 1){
              errs() << "This GEPI is coalesced\n";
              //errs() << "IV increment coefficient: " << ivinc << '\n';
              //errs() << "Thread ID.x increment coefficient: " << tidinc << '\n';
              //FindBaseAddress(IDX, IVincr, TIDincr);
            } else {
              errs() << "This GEPI is not coalesced\n";
              errs() << "IV increment coefficient: " << ivinc << '\n';
              errs() << "Thread ID.x increment coefficient: " << tidinc << '\n';
              int segdim = (ivinc>0)+(tidinc>0); //Cannot be 0 due to LICM
              Segdims.insert(std::make_pair(GEPI, (ivinc>0)+(tidinc>0)));
              Noncoalesced.push_back(GEPI);
              Instruction *insertionpt = &*(B->getFirstInsertionPt());
              IRBuilder<> builder(insertionpt);
              Value* baddr = FindBaseAddress(builder, IDX, IVincr, TIDincr);
              
              //Null means zero. Assuming 64-bit address
              if (baddr == NULL) {
                Type* i64type = IntegerType::getInt64Ty(F.getContext());
                baddr = ConstantInt::get(i64type, 0);
              }
              errs() << "Base address calculated: ";
              baddr->dump();
              Baseaddr.insert(std::make_pair(GEPI, baddr));
            }
          }
          
        } else if (CAST(PHINode, PN, &inst)){
          lastphi = PN;
        }
      }

      //Start loop transformation.
      //Create outer loops, move necessary instructions
      //Make i++ -> i+=16
      //Introduce internal IV, replacealluseswith
      DominatorTree& DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
      llvm::StringRef basename = B->getName();
      BasicBlock* outercond = SplitBlock(B, dyn_cast<Instruction>(lastphi), &DT, 
                                &loops,NULL, basename + ".outer.cond", false);
      BasicBlock* outerhead = IV->getParent();
      auto IVop = IV->getIncomingValue(1);

      //Want this to be increment operation
      assert(isa<Instruction>(IVop));
        
      errs() << "Identified IV increment op";
      IVop->dump();
      DominatorTree& DT2 = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
      
      BasicBlock* innerloop = SplitBlock(outercond, dyn_cast<Instruction>(IVop), &DT, 
                                &loops,NULL, basename + ".innerloop", true);
                                
      if(CAST(BinaryOperator, BO, IVop)){
        assert (BO->getOpcode() == Instruction::BinaryOps::Add);
        assert (isa<ConstantInt>(BO->getOperand(1)));
        int val = dyn_cast<ConstantInt>(BO->getOperand(1))->getSExtValue();
        
        BO->setOperand(1, ConstantInt::get(BO->getType(), val*16));
        
      } else {
        errs() << "Not well formed IV\n";
      }

      
      //Create internal IV
      IRBuilder<> builder_front(&*(innerloop->getFirstInsertionPt()));

      Instruction* terminator = innerloop->getTerminator();
      IRBuilder<> builder_end(terminator);
      Type* IVType = IV->getType();
      PHINode *newIV = builder_front.CreatePHI(IVType, 2); // assert that numpredecessors=2
      
      auto increment = builder_end.CreateAdd(newIV, ConstantInt::get(IVType, 1));
      auto compare = builder_end.CreateICmpEQ(increment, ConstantInt::get(IVType, 16));
      auto newbranch = builder_end.CreateCondBr(compare, outercond, innerloop);
      newbranch->setSuccessor(0, outercond);
      newbranch->setSuccessor(1, innerloop);
      terminator->eraseFromParent();
      newIV->addIncoming(ConstantInt::get(IVType, 0), outerhead);
      newIV->addIncoming(increment, innerloop);
      auto IV2 = builder_front.CreateAdd(newIV, IV);
      std::vector<Use*> toReplace;
      for (auto& U: IV->uses()){
        User* usr = U.getUser();
        if(CAST(Instruction, Inst, usr)){
          if(Inst==IV2 || Inst==IVop)
            continue;
          toReplace.push_back(&U);
        }
      }
      for (auto U: toReplace){
        U->set(IV2);
      }
      
      

    }
  }
  bool Changed = false;
  return Changed;
}
