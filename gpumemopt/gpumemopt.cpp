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
static RegisterPass<GPUMemCoalescing> GMC("gpumemcoal",
			    "GPU Memory coalescing pass",
			    true /* does modify the CFG */,
			    false /* transformation, not just analysis */);

namespace{
  struct GPUMemOpt : public ModulePass{
    static char ID; // Pass identification
    GPUMemOpt() : ModulePass(ID) { }
    bool runOnModule(Module &M){
      M.getOrInsertFunction("llvm.nvvm.barrier0", FunctionType::get(Type::getVoidTy(M.getContext()),false));
      bool Changed = false;
      for (auto &F : M){
        if (!(F.isDeclaration()))
          getAnalysis<GPUMemCoalescing>(F, &Changed);
      }
      return Changed;
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<GPUMemCoalescing>();
    }

  };

}

char GPUMemOpt::ID = 0;
static RegisterPass<GPUMemOpt> GMO("gpumemopt",
			    "GPU Memory optimization pass",
			    true /* does modify the CFG */,
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
    ValueMap<GetElementPtrInst*, Value*> Baseaddr;
    auto IV = loop->getCanonicalInductionVariable();
    //IV->dump();
    
    //loop->addChildLoop

    //Zero marks that this is the exact IV
    IVincr.insert(std::make_pair(IV, 0));
    errs() << "Found induction variable\n";
    AnalyzeIncrement(IV, IVincr, 1);
    BasicBlock *B = IV->getParent();
    Instruction* splitpoint = &*(B->getFirstInsertionPt());

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
            IRBuilder<> builder(splitpoint);
            Value* baddr = FindBaseAddress(builder, IDX, IVincr, TIDincr);
            
            //Null means zero. Assuming 64-bit address
            if (baddr == NULL) {
              Type* i64type = IntegerType::getInt64Ty(F.getContext());
              baddr = ConstantInt::get(i64type, 0);
            } else if (CAST(Instruction, newpoint, baddr)){
              splitpoint = newpoint;
            }
            errs() << "Base address calculated: ";
            baddr->dump();
            Baseaddr.insert(std::make_pair(GEPI, baddr));
          }
        }
        
      }
    }

    //Start loop transformation.
    //Create outer loops, move necessary instructions
    //Make i++ -> i+=16
    //Introduce internal IV, replacealluseswith
    if(!Noncoalesced.empty()){

      DominatorTree& DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
      llvm::StringRef basename = B->getName();
      BasicBlock* outercond = SplitBlock(B, splitpoint->getNextNonDebugInstruction(), &DT, 
                                &loops,NULL, basename + ".outer.cond", false);
      BasicBlock* outerhead = IV->getParent();
      auto IVop = IV->getIncomingValue(1);

      //Want this to be increment operation
      assert(isa<Instruction>(IVop));
        
      errs() << "Identified IV increment op";
      IVop->dump();
      
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
      PHINode *newIV;
      for (auto &OPN: outerhead->phis()){
        PHINode *NPN = builder_front.CreatePHI(OPN.getType(), 2);// assert that numpredecessors=2
        if (&OPN == IV){
          newIV = NPN;
        } else {
          OPN.replaceAllUsesWith(NPN);
          NPN->addIncoming(&OPN, outerhead);
          NPN->addIncoming(OPN.getIncomingValue(1), innerloop);
        } 
      }
      Type* IVType = IV->getType();
      
      auto increment = BinaryOperator::CreateAdd(newIV, ConstantInt::get(IVType, 1),"new.iv", terminator);
      auto compare = ICmpInst::Create(Instruction::ICmp, CmpInst::ICMP_EQ,
                                      increment, ConstantInt::get(IVType, 16), "", terminator);
      auto newbranch = BranchInst::Create(outercond, innerloop, compare);
      newbranch->setSuccessor(0, outercond);
      newbranch->setSuccessor(1, innerloop);
      ReplaceInstWithInst(terminator, newbranch);
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

    for (auto GEPI: Noncoalesced){
      // SKIP segdims == 0 because it will hit the l1 cache
      auto IDX = GEPI->idx_begin()->get();
      int ivinc = IVincr.count(IDX) ? IVincr[IDX] : 0;
      int tidinc = TIDincr.count(IDX) ? TIDincr[IDX] : 0;
      Value* sharedAddr;
      Value* sharedIdx;
      if (ivinc == 1 && tidinc == 0){
        
      } else if (ivinc==1 && tidinc > 0){
        //create loop and reverse iv/tid
      } else if (ivinc > 0 && ivinc <= 8 && tidinc == 0){
        //should we count this case?
      } else if (ivinc > 0 && ivinc <= 8 && tidinc > 0){
        
      }
    }

  }
  bool Changed = false;
  return Changed;
}
