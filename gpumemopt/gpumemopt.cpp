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
#define TILEDIM 32
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

bool FindNoncoalesced(BasicBlock* B, ValueMap<Value*, int>& TIDincr,
  ValueMap<Value*, int>& IVincr, std::vector<GetElementPtrInst*>& Noncoalesced,
  ValueMap<GetElementPtrInst*, Value*>& Baseaddr, Instruction* splitpoint){
  
  for (Instruction& inst : *B){
    if (CAST(GetElementPtrInst, GEPI, &inst)){
      if(GEPI->getAddressSpace() == 1 &&
        GEPI->getNumIndices() == 1){
        auto IDX = GEPI->idx_begin()->get();

        int ivinc = IVincr.count(IDX) ? IVincr[IDX] : 0;
        int tidinc = TIDincr.count(IDX) ? TIDincr[IDX] : 0;
        
        if(ivinc % 64 == 0 && tidinc == 1){
          //errs() << "This GEPI is coalesced\n";
          //errs() << "IV increment coefficient: " << ivinc << '\n';
          //errs() << "Thread ID.x increment coefficient: " << tidinc << '\n';
          //FindBaseAddress(IDX, IVincr, TIDincr);
        } else {
          //errs() << "This GEPI is not coalesced\n";
          //errs() << "IV increment coefficient: " << ivinc << '\n';
          //errs() << "Thread ID.x increment coefficient: " << tidinc << '\n';
          Noncoalesced.push_back(GEPI);
          IRBuilder<> builder(splitpoint);
          Value* baddr = FindBaseAddress(builder, IDX, IVincr, TIDincr);
          
          //Null means zero. Assuming 64-bit address
          if (baddr == NULL) {
            Type* i64type = IntegerType::getInt64Ty(B->getParent()->getContext());
            baddr = ConstantInt::get(i64type, 0);
          } else if (CAST(Instruction, newpoint, baddr)){
            splitpoint = newpoint;
          }
          //errs() << "Base address calculated: ";
          //baddr->dump();
          Baseaddr.insert(std::make_pair(GEPI, baddr));
        }
      }
    }
  }
  return !Noncoalesced.empty();
}

Instruction* TileLoop(BasicBlock* B,PHINode* IV, Instruction* splitpoint, DominatorTree& DT,
    BasicBlock* innerloop, BasicBlock* outercond){
  llvm::StringRef basename = B->getName();
  outercond = SplitBlock(B, splitpoint, &DT, 
                            NULL,NULL, basename + ".outer.cond", false);
  BasicBlock* outerhead = IV->getParent();
  auto IVop = IV->getIncomingValue(1);

  //Want this to be increment operation
  assert(isa<Instruction>(IVop));
  
  innerloop = SplitBlock(outercond, dyn_cast<Instruction>(IVop), &DT, 
                            NULL,NULL, basename + ".innerloop", true);
                            
  if(CAST(BinaryOperator, BO, IVop)){
    assert (BO->getOpcode() == Instruction::BinaryOps::Add);
    assert (isa<ConstantInt>(BO->getOperand(1)));
    int val = dyn_cast<ConstantInt>(BO->getOperand(1))->getSExtValue();
    
    BO->setOperand(1, ConstantInt::get(BO->getType(), val*TILEDIM));
    
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
                                  increment, ConstantInt::get(IVType, TILEDIM), "", terminator);
  auto newbranch = BranchInst::Create(outercond, innerloop, compare);
  newbranch->setSuccessor(0, outercond);
  newbranch->setSuccessor(1, innerloop);
  ReplaceInstWithInst(terminator, newbranch);
  newIV->addIncoming(ConstantInt::get(IVType, 0), outerhead);
  newIV->addIncoming(increment, innerloop);
  auto IV2 = builder_front.CreateAdd(newIV, IV, IV->getName()+".new");
  std::vector<Use*> toReplace;
  for (auto& U: IV->uses()){
    User* usr = U.getUser();
    if(CAST(Instruction, Inst, usr)){
      if(Inst==IV2 || Inst==IVop)
        continue;
      if(Inst->getParent() != innerloop)
        continue;
      toReplace.push_back(&U);
    }
  }
  for (auto U: toReplace){
    U->set(IV2);
  }

  IRBuilder<> outercond_end(outercond);
  outercond_end.SetInsertPoint(outercond->getTerminator());
  auto barrierFunc = B->getParent()->getParent()->getOrInsertFunction("llvm.nvvm.barrier0", 
  FunctionType::get(Type::getVoidTy(B->getParent()->getContext()),false));
  outercond_end.CreateCall(barrierFunc);
  IRBuilder<> headerbuilder(outerhead);
  headerbuilder.SetInsertPoint(outerhead->getTerminator());
  headerbuilder.CreateCall(barrierFunc);
  return newIV;
}

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
  Value* TIDX = nullptr;
  Module* M = F.getParent();
  auto barrierFunc = M->getOrInsertFunction("llvm.nvvm.barrier0", 
    FunctionType::get(Type::getVoidTy(F.getContext()),false));

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
          if (TIDX == nullptr){
            TIDX = CI;
          }
        }
      }
    }
  }

  LoopInfo& loops = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  for (auto I : loops){
  for (auto loop : depth_first(I)){
    ValueMap<Value*, int> IVincr;
    std::vector<GetElementPtrInst*> Noncoalesced;
    ValueMap<GetElementPtrInst*, Value*> Baseaddr;
    auto IV = loop->getCanonicalInductionVariable();

    //Zero marks that this is the exact IV
    IVincr.insert(std::make_pair(IV, 0));

    //errs() << "Found induction variable\n";
    AnalyzeIncrement(IV, IVincr, 1);
    BasicBlock *B = IV->getParent();
    auto basename = B->getName();
    Instruction* splitpoint = &*(B->getFirstInsertionPt());
    //Start loop transformation.
    //Create outer loops, move necessary instructions
    //Make i++ -> i+=16
    //Introduce internal IV, replacealluseswith
    if(FindNoncoalesced(B, TIDincr, IVincr, Noncoalesced, Baseaddr, splitpoint)){

      DominatorTree& DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
      BasicBlock* outercond;
      BasicBlock* innerloop;
      Instruction* newIV = TileLoop(B, IV, splitpoint, DT, innerloop, outercond);
      BasicBlock* outerhead = IV->getParent();
      outerhead->setName(basename+".outerhead");
      for (auto GEPI: Noncoalesced){
      // SKIP segdims == 0 because it will hit the l1 cache
        auto IDX = GEPI->idx_begin()->get();
        int ivinc = IVincr.count(IDX) ? IVincr[IDX] : 0;
        int tidinc = TIDincr.count(IDX) ? TIDincr[IDX] : 0;
        Value* sharedAddr;
        Value* sharedIdx;
        GEPI->dump();
        errs() << "IVIncr : " << ivinc << "\tTIDincr: " << tidinc <<"\n";
        if (ivinc == 1 && tidinc == 0){
          auto shrtype = ArrayType::get(GEPI->getResultElementType(), TILEDIM);
          GlobalVariable* shr = new GlobalVariable(*F.getParent(),shrtype, false,
            GlobalValue::LinkageTypes::InternalLinkage, UndefValue::get(shrtype),
            "shared_" + GEPI->getPointerOperand()->getName().str(), nullptr, 
            GlobalValue::NotThreadLocal, 3);
          shr->setAlignment(MaybeAlign(4));
          IRBuilder<> headerbuilder(outerhead);
          auto baddr = Baseaddr[GEPI];
          //Insert before the barrier
          headerbuilder.SetInsertPoint(outerhead->getTerminator()->getPrevNode());
          
          Value* offset = headerbuilder.CreateAdd(TIDX, IV);
          if (baddr->getType() != offset->getType()){
            offset = headerbuilder.CreateZExt(offset,baddr->getType());
          }
          Value* nidx = headerbuilder.CreateAdd(baddr, offset);
          
          auto globalptr = headerbuilder.CreateGEP(GEPI->getPointerOperand(), nidx);
          auto globalval = headerbuilder.CreateLoad(globalptr);
          Value* Ntidx = TIDX;
          if (baddr->getType() != Ntidx->getType()){
            Ntidx = headerbuilder.CreateZExt(TIDX,baddr->getType());
          }
          auto shrptr = headerbuilder.CreateGEP(shr, {ConstantInt::get(baddr->getType(),0), Ntidx});
          headerbuilder.CreateStore(globalval, shrptr);

          auto newIV_addr = ZExtInst::Create(Instruction::CastOps::ZExt,
            newIV, baddr->getType(), "",GEPI);
          GetElementPtrInst* Ngepi = GetElementPtrInst::Create(nullptr, shr,
            {ConstantInt::get(baddr->getType(),0), newIV_addr},GEPI->getName()+"_shared", GEPI);
          
          bool gepialive = false;
          std::vector<Instruction*> to_erase;
          for (auto U : GEPI->users()){
            if (CAST(LoadInst, LI, U)){
              auto NLI = new LoadInst(LI->getType(),Ngepi,"", LI);
              LI->replaceAllUsesWith(NLI);
              to_erase.push_back(LI);
            } else {
              gepialive = true;
              errs() << "why is this alive?\n";
            }
          }
          for(auto LI: to_erase){
            LI->eraseFromParent();
          }
          if (!gepialive)
            GEPI->eraseFromParent();
          
        } else if (ivinc==1 && tidinc > 8){
          //create loop and reverse iv/tid
          auto shrtype = ArrayType::get(GEPI->getResultElementType(), TILEDIM*TILEDIM);
          GlobalVariable* shr = new GlobalVariable(*F.getParent(),shrtype, false,
            GlobalValue::LinkageTypes::InternalLinkage, UndefValue::get(shrtype),
            "shared_" + GEPI->getPointerOperand()->getName().str(), nullptr, 
            GlobalValue::NotThreadLocal, 3);
          shr->setAlignment(MaybeAlign(4));

          Instruction* barrier = outerhead->getTerminator()->getPrevNode();
          
          BasicBlock* nouterhead = SplitBlock(outerhead, barrier);
          BasicBlock* loadloop = splitBlockBefore(nouterhead, barrier, 
            nullptr, nullptr, nullptr, basename+".loadloop."+GEPI->getPointerOperand()->getName());
          outerhead = barrier->getParent();

          IRBuilder<> loopbuilder(loadloop);
          loopbuilder.SetInsertPoint(loadloop->getTerminator());
          PHINode *LIV = loopbuilder.CreatePHI(IV->getType(), 2);
          auto baddr = Baseaddr[GEPI];
          auto iiv = loopbuilder.CreateAdd(IV, TIDX);
          auto tidoffset = loopbuilder.CreateMul(LIV, ConstantInt::get(LIV->getType(), tidinc));
          auto glboffset = loopbuilder.CreateAdd(iiv, tidoffset);
          Value* gidx;
          if(baddr->getType() != glboffset->getType()){
            gidx = loopbuilder.CreateAdd(baddr, loopbuilder.CreateZExt(glboffset, baddr->getType()));
          } else {
            gidx = loopbuilder.CreateAdd(baddr, glboffset);
          }
          auto GGEP = loopbuilder.CreateGEP(GEPI->getPointerOperand(), gidx);
          auto GLD = loopbuilder.CreateLoad(GGEP);
          
          auto livTile = loopbuilder.CreateMul(LIV, ConstantInt::get(LIV->getType(), TILEDIM));
          auto sidx = loopbuilder.CreateAdd(livTile, TIDX);
          if (sidx->getType() != baddr->getType()){
            sidx = loopbuilder.CreateZExt(sidx, baddr->getType());
          }
          auto SGEP = loopbuilder.CreateGEP(shr, {ConstantInt::get(baddr->getType(), 0), sidx});
          auto SST = loopbuilder.CreateStore(GLD, SGEP);

          IRBuilder<> innerbuilder(GEPI);
          auto tidxTile = innerbuilder.CreateMul(TIDX, ConstantInt::get(TIDX->getType(), TILEDIM));
          auto nsidx = innerbuilder.CreateAdd(tidxTile, newIV);
          if (nsidx->getType() != baddr->getType()){
            nsidx = innerbuilder.CreateZExt(nsidx, baddr->getType());
          }
          auto Ngepi = innerbuilder.CreateGEP(shr, {ConstantInt::get(baddr->getType(), 0), nsidx});

          bool gepialive = false;
          std::vector<Instruction*> to_erase;
          for (auto U : GEPI->users()){
            if (CAST(LoadInst, LI, U)){
              auto NLI = new LoadInst(LI->getType(),Ngepi,"", LI);
              LI->replaceAllUsesWith(NLI);
              to_erase.push_back(LI);
            } else {
              gepialive = true;
              errs() << "why is this alive?\n";
            }
          }
          for(auto LI: to_erase){
            LI->eraseFromParent();
          }
          if (!gepialive)
            GEPI->eraseFromParent();

          auto increment = loopbuilder.CreateAdd(LIV, ConstantInt::get(IV->getType(), 1));
          auto compare = loopbuilder.CreateICmpEQ(increment, ConstantInt::get(IV->getType(), TILEDIM));
          auto newbranch = BranchInst::Create(nouterhead, loadloop, compare);
          newbranch->setSuccessor(0, nouterhead);
          newbranch->setSuccessor(1, loadloop);
          ReplaceInstWithInst(loadloop->getTerminator(), newbranch);
          LIV->addIncoming(ConstantInt::get(IV->getType(), 0), loadloop->getPrevNode());
          LIV->addIncoming(increment, loadloop);
        } else if (ivinc ==1 && tidinc <= 8 && tidinc > 0){
          auto shrtype = ArrayType::get(GEPI->getResultElementType(), (ivinc+tidinc)*TILEDIM);
          GlobalVariable* shr = new GlobalVariable(*F.getParent(),shrtype, false,
            GlobalValue::LinkageTypes::InternalLinkage, UndefValue::get(shrtype),
            "shared_" + GEPI->getPointerOperand()->getName().str(), nullptr, 
            GlobalValue::NotThreadLocal, 3);
          shr->setAlignment(MaybeAlign(4));
          IRBuilder<> headerbuilder(outerhead);
          auto baddr = Baseaddr[GEPI];
          //Insert before the barrier
          headerbuilder.SetInsertPoint(outerhead->getTerminator()->getPrevNode());

          //Unrolled loads
          Value* offset = headerbuilder.CreateAdd(TIDX, IV);
          if (baddr->getType() != offset->getType()){
            offset = headerbuilder.CreateZExt(offset,baddr->getType());
          }
          Value* nidx = headerbuilder.CreateAdd(baddr, offset);
          
          auto globalptr = headerbuilder.CreateGEP(GEPI->getPointerOperand(), nidx);
          auto globalval = headerbuilder.CreateLoad(globalptr);
          Value* Ntidx = TIDX;
          if (baddr->getType() != Ntidx->getType()){
            Ntidx = headerbuilder.CreateZExt(TIDX,baddr->getType());
          }
          auto shrptr = headerbuilder.CreateGEP(shr, {ConstantInt::get(baddr->getType(),0), Ntidx});
          headerbuilder.CreateStore(globalval, shrptr);
          for (int k = 1; k<(ivinc+tidinc); k++){
            nidx = headerbuilder.CreateAdd(nidx, ConstantInt::get(baddr->getType(),TILEDIM));
            auto gptr = headerbuilder.CreateGEP(GEPI->getPointerOperand(), nidx);
            auto gval = headerbuilder.CreateLoad(gptr);
            Ntidx = headerbuilder.CreateAdd(Ntidx, ConstantInt::get(baddr->getType(),TILEDIM));
            auto sptr = headerbuilder.CreateGEP(shr, {ConstantInt::get(baddr->getType(),0), Ntidx});
            headerbuilder.CreateStore(gval, sptr);
          }

          IRBuilder<> innerbuilder(GEPI);
          auto tidxTile = innerbuilder.CreateMul(TIDX, ConstantInt::get(TIDX->getType(), tidinc));
          auto nsidx = innerbuilder.CreateAdd(tidxTile, newIV);
          if (nsidx->getType() != baddr->getType()){
            nsidx = innerbuilder.CreateZExt(nsidx, baddr->getType());
          }
          auto Ngepi = innerbuilder.CreateGEP(shr, {ConstantInt::get(baddr->getType(), 0), nsidx});
          
          bool gepialive = false;
          std::vector<Instruction*> to_erase;
          for (auto U : GEPI->users()){
            if (CAST(LoadInst, LI, U)){
              auto NLI = new LoadInst(LI->getType(),Ngepi,"", LI);
              LI->replaceAllUsesWith(NLI);
              to_erase.push_back(LI);
            } else {
              gepialive = true;
              errs() << "why is this alive?\n";
            }
          }
          for(auto LI: to_erase){
            LI->eraseFromParent();
          }
          if (!gepialive)
            GEPI->eraseFromParent();
        } else if (ivinc > 0 && ivinc <= 8 && tidinc > 0){
          
        }
      }
    }

    

  }}
  bool Changed = false;
  return Changed;
}