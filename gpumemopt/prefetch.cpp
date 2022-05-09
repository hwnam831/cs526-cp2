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
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/InstrTypes.h"
using namespace llvm;

namespace {
  struct GPUMemPrefetching : public FunctionPass {
    static char ID; // Pass identification
    GPUMemPrefetching() : FunctionPass(ID) { }

    // Entry point for the overall scalar-replacement pass
    bool runOnFunction(Function &F);
    bool runOnLoop(Loop *L);
    const SCEVAddRecExpr *findAddRecExpr(const SCEV * expr);
    const SCEV *createInitialPrefAddr(const SCEV * expr);
    // getAnalysisUsage - List passes required by this pass.  We also know it
    // will not alter the CFG, so say so.
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addRequired<LoopInfoWrapperPass>();
      AU.addRequired<PostDominatorTreeWrapperPass>();
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<ScalarEvolutionWrapperPass>();
      AU.addPreserved<ScalarEvolutionWrapperPass>();
    }

  private:
    // Add fields and helper functions for this pass here.
    LoopInfo *LI;
    ScalarEvolution *SE;
    PostDominatorTree *PDT;
    DominatorTree *DT;
    Loop *prefLoop; // The loop containing the post-dominating barrier instruction
    BasicBlock *prefBlock;
    BasicBlock *prefBackBlock;
    Instruction *prefIfInsertPt; // The else branch where this is not the last iteration
    ICmpInst *prefLoopGuardCmp;
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

// This function finds the AddRecExpr (start, +, step) in expr. This information 
// can be used to analyze the linear relationship between the address and the induction 
// variable in order to calculate the prefetch address.
const SCEVAddRecExpr *GPUMemPrefetching::findAddRecExpr(const SCEV * expr_o){
  if (const SCEVNAryExpr *expr = dyn_cast<SCEVNAryExpr>(expr_o)){
   if(!isa<SCEVAddRecExpr>(expr)){
      for (unsigned i = 0; i < expr->getNumOperands(); ++i){
        if(!SE->containsAddRecurrence(expr->getOperand(i))){
          continue;
        }

        const SCEVAddRecExpr * sub_expr = findAddRecExpr(expr->getOperand(i));
        if(sub_expr != nullptr)
          return sub_expr;
      }
      return nullptr;
    } else {
      const SCEVAddRecExpr *LSCEAddRec_expr = dyn_cast<SCEVAddRecExpr>(expr);
      return LSCEAddRec_expr;
    }
  } else if(const SCEVCastExpr *expr= dyn_cast<SCEVCastExpr>(expr_o)){
      for (unsigned i = 0; i < expr->getNumOperands(); ++i){
        if(!SE->containsAddRecurrence(expr->getOperand(i))){
          continue;
        }

        const SCEVAddRecExpr * sub_expr = findAddRecExpr(expr->getOperand(i));
        if(sub_expr != nullptr)
          return sub_expr;
      }
      return nullptr;
  } else if(const SCEVUDivExpr *expr = dyn_cast<SCEVUDivExpr>(expr_o)){
      for (unsigned i = 0; i < expr->getNumOperands(); ++i){
        if(!SE->containsAddRecurrence(expr->getOperand(i))){
          continue;
        }

        const SCEVAddRecExpr * sub_expr = findAddRecExpr(expr->getOperand(i));
        if(sub_expr != nullptr)
          return sub_expr;

      }
      return nullptr;
  } else {
    return nullptr;
  }
}

// Since SCEV object is immutable, this function creates a new SCEV by replacing all AddRecExpr that are within
// subloops of prefLoop with its initial value in the loop. This new SCEV is used for computing the prefetch address
// at the first iteration before the loop starts.
const SCEV *GPUMemPrefetching::createInitialPrefAddr(const SCEV * expr){
  if(!isa<SCEVAddRecExpr>(expr)){
    switch (expr->getSCEVType()) {
      case scConstant:
        return expr;
      case scPtrToInt: 
        if(SE->containsAddRecurrence(expr)){
          const SCEVCastExpr *SCEVCast_expr = dyn_cast<SCEVCastExpr>(expr);
          return SE->getPtrToIntExpr(createInitialPrefAddr(SCEVCast_expr->getOperand(0)), expr->getType());
        } else
          return SE->getPtrToIntExpr(expr, expr->getType());
      case scTruncate:
        if(SE->containsAddRecurrence(expr)){
          const SCEVCastExpr *SCEVCast_expr = dyn_cast<SCEVCastExpr>(expr);
          return SE->getTruncateExpr(createInitialPrefAddr(SCEVCast_expr->getOperand(0)), expr->getType());
        } else
          return SE->getTruncateExpr(expr, expr->getType());
      case scZeroExtend:
        if(SE->containsAddRecurrence(expr)){
          const SCEVCastExpr *SCEVCast_expr = dyn_cast<SCEVCastExpr>(expr);
          return SE->getZeroExtendExpr(createInitialPrefAddr(SCEVCast_expr->getOperand(0)), expr->getType());
        } else
          return SE->getZeroExtendExpr(expr, expr->getType());
      case scSignExtend:
        if(SE->containsAddRecurrence(expr)){
          const SCEVCastExpr *SCEVCast_expr = dyn_cast<SCEVCastExpr>(expr);
          return SE->getSignExtendExpr(createInitialPrefAddr(SCEVCast_expr->getOperand(0)), expr->getType());
        } else
          return SE->getSignExtendExpr(expr, expr->getType());
      // case scAddRecExpr:
      case scMulExpr: {
        const SCEVMulExpr * SCEVMul_expr = dyn_cast<SCEVMulExpr>(expr);
        SmallVector<const SCEV *, 5> operands;

        for (unsigned i = 0; i < SCEVMul_expr->getNumOperands(); ++i){
          if(SE->containsAddRecurrence(SCEVMul_expr->getOperand(i))){
            operands.push_back(createInitialPrefAddr(SCEVMul_expr->getOperand(i)));
          } else {
            operands.push_back(SCEVMul_expr->getOperand(i));
          }
        }
        return SE->getMulExpr(operands);
      }
      case scUMaxExpr:
      case scSMaxExpr:
      case scUMinExpr:
      case scSMinExpr: {
        const SCEVMinMaxExpr * SCEVMinMax_expr = dyn_cast<SCEVMinMaxExpr>(expr);
        SmallVector<const SCEV *, 5> operands;
        for (unsigned i = 0; i < SCEVMinMax_expr->getNumOperands(); ++i){
          if(SE->containsAddRecurrence(SCEVMinMax_expr->getOperand(i))){
            operands.push_back(createInitialPrefAddr(SCEVMinMax_expr->getOperand(i)));
          } else {
            operands.push_back(SCEVMinMax_expr->getOperand(i));
          }
        }
        return SE->getMinMaxExpr(expr->getSCEVType(), operands);
      }
      case scAddExpr: {
        const SCEVAddExpr * SCEVAdd_expr = dyn_cast<SCEVAddExpr>(expr);
        SmallVector<const SCEV *, 5> operands;
        for (unsigned i = 0; i < SCEVAdd_expr->getNumOperands(); ++i){
          if(SE->containsAddRecurrence(SCEVAdd_expr->getOperand(i))){
            operands.push_back(createInitialPrefAddr(SCEVAdd_expr->getOperand(i)));
          } else {
            operands.push_back(SCEVAdd_expr->getOperand(i));
          }
        }
        return SE->getAddExpr(operands);
      }
      case scUDivExpr: {
        const SCEVUDivExpr * SCEVUDiv_expr = dyn_cast<SCEVUDivExpr>(expr);
        SmallVector<const SCEV *, 2> operands;
        for (unsigned i = 0; i < SCEVUDiv_expr->getNumOperands(); ++i){
          if(SE->containsAddRecurrence(SCEVUDiv_expr->getOperand(i))){
            operands.push_back(createInitialPrefAddr(SCEVUDiv_expr->getOperand(i)));
          } else {
            operands.push_back(SCEVUDiv_expr->getOperand(i));
          }
        }
        return SE->getUDivExpr(operands[0], operands[1]);
      }
      case scUnknown:
        return expr;
      case scCouldNotCompute:
      default:{
        // llvm_unreachable("Attempt to use a SCEVCouldNotCompute object!");
        return nullptr;
      }
    }
  } else {
    const SCEVAddRecExpr *LSCEAddRec_expr = dyn_cast<SCEVAddRecExpr>(expr);
    const Loop * expr_loop = LSCEAddRec_expr->getLoop();
    if(prefLoop == expr_loop){ // TODO: check others
      return LSCEAddRec_expr->getStart();
    } else {
      for (Loop *subLoop : prefLoop->getSubLoops()) {
        if(subLoop == expr_loop){
          return LSCEAddRec_expr->getStart();
        }
      }
      return LSCEAddRec_expr;
    }
  }
}

bool GPUMemPrefetching::runOnLoop(Loop *L) {
  bool Changed = false;

  BasicBlock *Incoming = nullptr, *Backedge = nullptr;
  if (!L->getIncomingAndBackEdge(Incoming, Backedge)){
    // errs() << "unsupported loop form.\n";
    return Changed;
  }

  // Find memory load from global memory that is stored into shared memory
  for (auto BB : L->blocks()) {
    for (auto &I : *BB) {
      // only prefetch for loads from global memory that stores to shared memory
      StoreInst *SMemI;
      Value *ValOp;
      if (SMemI = dyn_cast<StoreInst>(&I)) {
        Value *PtrOp = SMemI->getPointerOperand();
        ValOp = SMemI->getValueOperand();
        unsigned PtrAddrSpace = PtrOp->getType()->getPointerAddressSpace();
        // shared memory access address space 3
        if (PtrAddrSpace != 3)
          continue;
      } else continue;

      LoadInst *LMemI;
      Value *LPtrOp;
      if (LMemI = dyn_cast<LoadInst>(ValOp)) {
        LPtrOp = LMemI->getPointerOperand();
        unsigned ValAddrSpace = LPtrOp->getType()->getPointerAddressSpace();
        // global memory access address space 1
        if (ValAddrSpace != 1)
          continue;
      } else {
        // errs() << "store value not from a load\n";
        continue;
      }

      // find the next post-dominating barrier
      Instruction *Inst = dyn_cast<Instruction>(&I);
      BasicBlock *BBlock = BB;
      bool found_barrier = false;
      while(!found_barrier && BBlock->getTerminator()->getNumSuccessors() > 0){
        while(Inst->getNextNode() != nullptr) {
          if (CallInst *CI = dyn_cast<CallInst>(Inst)){
            auto funcname = CI->getCalledFunction()->getName();
            if(funcname == "llvm.nvvm.barrier0"){
              found_barrier = true;
              break;
            }
          }
          Inst = Inst->getNextNode();
        }
        if(found_barrier)
          break;
        bool found_child = false;
        for (succ_iterator sit = succ_begin(BBlock), set = succ_end(BBlock); sit != set; ++sit){
          BasicBlock *succ = *sit;
          Inst = dyn_cast<Instruction>(succ->begin());
          if(PDT->dominates(succ, BBlock)){
            BBlock = succ;
            found_child = true;
            break;
          }
        }
        if(!found_child){
          break;
        }
      }
      // if such barrier is not found, then prefetch cannot be safely carried out, so we do not 
      // support prefetch in this condition.
      if(!found_barrier){
        // errs() << "not supported for prefetch.\n";
        continue;
      }
      
      // to check if 2-dimensional prefetch is necessary here
      bool to_tile = false;
      if(L->contains(Inst)){
        prefLoop = L;
        prefBlock = Incoming;
        prefBackBlock = Backedge;

        bool is_tiled = false;
        for (Loop *subLoop : L->getSubLoops()) {
            // needs a temp array to prefetch (2d-prefetch)
            if (subLoop->contains(&I)){
              is_tiled = true;
              break;
            }
        }
        // process later in inner loop
        if(is_tiled){
          if(GetElementPtrInst *gepi = dyn_cast<GetElementPtrInst>(LPtrOp)){
            const SCEV *LSCEV = SE->getSCEVAtScope(gepi->getOperand(1), prefLoop);
          }
          // errs() << "process later\n";
          continue;
        }
      } else if(prefLoop != nullptr && prefLoop->contains(Inst)){ // outerloop has prefetched access
        to_tile = true;
      } else if(prefLoop == nullptr){ // find which loop contains the post-dominating barrier instruction
        for (Loop *I_loop : *LI){
          for (Loop *L_loop : depth_first(I_loop)){
            if(L_loop->contains(Inst)){
              prefLoop = L_loop;
              break;
            }
          }
          if(prefLoop != nullptr)
            break;
        }
        if(prefLoop == nullptr)
          continue;
        to_tile = true;
      } else{
        // errs() << "not supported for prefetch.\n";
        continue;
      }

      // need this information for 2-d prefetch
      PHINode *IV = L->getCanonicalInductionVariable();
      if(IV == nullptr && to_tile) continue;

      if(GetElementPtrInst *gepi = dyn_cast<GetElementPtrInst>(LPtrOp)){
        const SCEV *LSCEV = SE->getSCEVAtScope(gepi->getOperand(1), prefLoop);

        const SCEVNAryExpr  *LSCEVAddRec = dyn_cast<SCEVNAryExpr>(LSCEV);
        if(LSCEV != nullptr){
          const SCEVAddRecExpr *expr = findAddRecExpr(LSCEV);
          if(expr != nullptr){
            if(to_tile) {
              // analyze to get the prefetch address of the first iteration
              const SCEV *LSCEV_inner = SE->getSCEVAtScope(gepi->getOperand(1), L);
              const SCEV *initLSCEV = LSCEV_inner;
              initLSCEV = createInitialPrefAddr(initLSCEV);

              const SCEV *LSCEV_inner_test = SE->getSCEVAtScope(gepi->getOperand(1), prefLoop);
              const SCEVNAryExpr  *LSCEVAddRec_inner = dyn_cast<SCEVNAryExpr>(LSCEV_inner);
              if(LSCEVAddRec != nullptr){
                const SCEVAddRecExpr *expr_inner = findAddRecExpr(LSCEVAddRec_inner);
                if(expr_inner != nullptr){
                  const SCEVConstant *C, *C_inner;
                  ConstantInt *CI, *CI_inner;
                  // get step increment at each iteration for the inner loop containing SI 
                  // and the prefLoop containing BI
                  if (C = dyn_cast<SCEVConstant>(expr->getStepRecurrence(*SE))) {
                    if (C_inner = dyn_cast<SCEVConstant>(expr_inner->getStepRecurrence(*SE))) {            
                      C->getValue()->getSExtValue();          
                      CI = C->getValue();
                      CI_inner = C_inner->getValue();
                    } else continue;
                  } else continue;
                  Changed = true;
                  
                  BranchInst *loopGuardBr;
                  IRBuilder<> Builder(Inst->getNextNode()); // insert after the barrier instruction
                  if(CI->getType() != gepi->getOperand(1)->getType()){
                    CI = dyn_cast<ConstantInt>(Builder.CreateZExt(CI, gepi->getOperand(1)->getType())); 
                  }
                  if(CI_inner->getType() != gepi->getOperand(1)->getType()){
                    CI_inner = dyn_cast<ConstantInt>(Builder.CreateZExt(CI_inner, gepi->getOperand(1)->getType())); 
                  }

                  // add boundary check for whether this is the last iteration if this is not added before
                  if(prefIfInsertPt == nullptr){
                    if(loopGuardBr = dyn_cast<BranchInst>(prefBackBlock->getTerminator())){ 
                      if(loopGuardBr->isConditional()){
                        if(ICmpInst *loopGuardCmp = dyn_cast<ICmpInst>(loopGuardBr->getOperand(0))){
                          prefLoopGuardCmp = loopGuardCmp;
                          
                          Value *iter_next;
                          if(isa<PHINode>(loopGuardCmp->getOperand(0))){
                            iter_next = Builder.CreateAdd(loopGuardCmp->getOperand(0), ConstantInt::get(loopGuardCmp->getOperand(0)->getType(), C->getValue()->getSExtValue()));
                          } else {
                            iter_next = Builder.CreateAdd(dyn_cast<Instruction>(loopGuardCmp->getOperand(0))->getOperand(0), ConstantInt::get(loopGuardCmp->getOperand(0)->getType(), C->getValue()->getSExtValue()));
                          }
                          Value *check_res = Builder.CreateICmp(loopGuardCmp->getPredicate(), iter_next, loopGuardCmp->getOperand(1));
                          Instruction *ThenTerm , *ElseTerm;
                          SplitBlockAndInsertIfThenElse(check_res, (dyn_cast<ICmpInst>(check_res))->getNextNode(), &ThenTerm, &ElseTerm);

                          prefIfInsertPt = ElseTerm;
                        } else {
                          // errs() << "loop backedge block terminator's first operand is not icmp.\n";
                          break;
                        }
                      } else {
                        // errs() << "loop backedge block terminator inst is not a conditional br.\n";
                        break;
                      }
                    } else {
                      // errs() << "loop backedge block terminator inst is not br.\n";
                    }
                  } else {
                    // errs() << "prefIfInsertPt is not null\n";
                    if(!isa<BranchInst>(prefIfInsertPt)){ //TODO: shouldn't need this check here.
                      // errs() << "prefIfInsertPt is not a branch instruction.\n";
                      continue;
                    }
                  }

                  // essentially duplicate the loop to prefetch for the next iteration
                  BasicBlock *prefIfInsertB = prefIfInsertPt->getParent();
                  BasicBlock *loopBody = SplitBlock(prefIfInsertB, prefIfInsertPt, DT, NULL, NULL, "", false);
                  Builder.SetInsertPoint(prefIfInsertPt);
                  PHINode *NPN = Builder.CreatePHI(CI->getType(), 2);
                  NPN->addIncoming(ConstantInt::get(CI->getType(), dyn_cast<ConstantInt>(IV->getOperand(0))->getSExtValue()), prefIfInsertB);
                  ConstantInt *CI_one = ConstantInt::get(CI->getType(), 1);
                  Value *prefAddrInc = Builder.CreateMul(NPN, CI_inner);
                  PHINode *prefIV = prefLoop->getInductionVariable(*SE);
                  Value *prefAddrInc2 = Builder.CreateAdd(prefAddrInc, CI); // get next iteration by adding 1 step
                  if(prefIV == nullptr) {
                    Instruction *prefIV_icmp = dyn_cast<Instruction>(prefLoopGuardCmp->getOperand(0));
                    for (User *U: prefIV_icmp->users()){
                      if(Instruction *I = dyn_cast<Instruction>(U)){
                        if(I->getOpcode()==Instruction::PHI){
                          prefIV = dyn_cast<PHINode>(I);
                        }
                      }
                    }
                  }
                  Builder.SetInsertPoint(prefIV->getNextNode());
                  ConstantInt *CI_zero = ConstantInt::get(CI->getType(), 0);
                  PHINode *NewIV = Builder.CreatePHI(CI->getType(), 2);
                  NewIV->addIncoming(CI_zero, prefIV->getIncomingBlock(0));
                  Builder.SetInsertPoint(prefIV->getParent()->getTerminator());
                  Value *NewIV_next = Builder.CreateAdd(NewIV, CI_one);
                  NewIV->addIncoming(NewIV_next, prefIV->getIncomingBlock(1));

                  Builder.SetInsertPoint(prefIfInsertPt);
                  Value *prefAddrInc3 = Builder.CreateMul(NewIV, CI); // newIV * prefLoop step
                  Value *prefAddrInc4 = Builder.CreateAdd(prefAddrInc3, prefAddrInc2);

                  Instruction *brTerminator = dyn_cast<Instruction>(prefBlock->getTerminator());
                  SCEVExpander SCEVE(*SE, BB->getModule()->getDataLayout(), "prefaddr");
                  Value *PrefPtrValue = SCEVE.expandCodeFor(initLSCEV, gepi->getOperand(1)->getType(), brTerminator);

                  Value *prefAddr = Builder.CreateAdd(PrefPtrValue, prefAddrInc4);

                  gepi->moveBefore(prefIfInsertPt);
                  gepi->setOperand(1, prefAddr);
                  LMemI->moveBefore(prefIfInsertPt);
                  Builder.SetInsertPoint(brTerminator);

                  // this is allocating the temporary array used to prefetch
                  Value *tempAllocaPtr = Builder.CreateAlloca(ArrayType::get(gepi->getOperand(0)->getType()->getPointerElementType(), CI->getSExtValue()));

                  // essentialy also duplicating the loop before loop starts to prefetch for first iteration
                  BasicBlock *initLoopBody = SplitBlock(prefBlock, brTerminator, DT, NULL, NULL, "", false);
                  Builder.SetInsertPoint(brTerminator);

                  PHINode *NPN_init = Builder.CreatePHI(CI->getType(), 2);
                  NPN_init->addIncoming(ConstantInt::get(CI->getType(), dyn_cast<ConstantInt>(IV->getOperand(0))->getSExtValue()), prefBlock);

                  // perform the initial prefetch before the loop starts
                  Value *prefAddrInc_init = Builder.CreateMul(NPN_init, CI_inner);
                  Value *prefAddr_init = Builder.CreateAdd(PrefPtrValue, prefAddrInc_init);
                  Value *initPrefAddr = Builder.CreateGEP(gepi->getOperand(0)->getType()->getPointerElementType(), gepi->getOperand(0), prefAddr_init);
                  Value *initPrefVal = Builder.CreateLoad(gepi->getOperand(0)->getType()->getPointerElementType(), initPrefAddr);
                  Value *tempGEP_init = Builder.CreateGEP(tempAllocaPtr, {ConstantInt::get(NPN_init->getType(),0), NPN_init});
                  Builder.CreateStore(initPrefVal, tempGEP_init);

                  Value *NPN_it_init = Builder.CreateAdd(NPN_init, CI_one);
                  Value *icmp_init = Builder.CreateICmpEQ(NPN_it_init, CI);
                  Builder.CreateCondBr(icmp_init, dyn_cast<BasicBlock>(brTerminator->getOperand(0)), initLoopBody);
                  NPN_init->addIncoming(NPN_it_init, initLoopBody);
                  brTerminator->eraseFromParent();

                  Builder.SetInsertPoint(prefIfInsertPt);
                  Value *tempGEP = Builder.CreateGEP(tempAllocaPtr, {ConstantInt::get(NPN->getType(),0), NPN});
                  Builder.CreateStore(LMemI, tempGEP);
                  Value *NPN_it = Builder.CreateAdd(NPN, CI_one);
                  Value *icmp = Builder.CreateICmpEQ(NPN_it, CI);
                  
                  BasicBlock *newblk = SplitBlock(prefIfInsertPt->getParent(), prefIfInsertPt, DT, NULL, NULL, "", false);
                  Builder.SetInsertPoint(dyn_cast<Instruction>(icmp)->getParent()->getTerminator());
                  Builder.CreateCondBr(icmp, newblk, loopBody);
                  dyn_cast<Instruction>(icmp)->getParent()->getTerminator()->eraseFromParent();

                  NPN->addIncoming(NPN_it, loopBody);

                  Builder.SetInsertPoint(SMemI);
                  Value *tempGEP2 = Builder.CreateGEP(tempAllocaPtr, {ConstantInt::get(NPN->getType(),0), IV});
                  Value *tempVal = Builder.CreateLoad(gepi->getOperand(0)->getType()->getPointerElementType(), tempGEP2);
                  SMemI->setOperand(0, tempVal);

                } else continue;
              } else continue;
            // this is 1-d prefetch case
            } else if(BranchInst *loopGuardBr = dyn_cast<BranchInst>(Backedge->getTerminator())){ 
              if(loopGuardBr->isConditional()){
                if(ICmpInst *loopGuardCmp = dyn_cast<ICmpInst>(loopGuardBr->getOperand(0))){
                  prefLoopGuardCmp = loopGuardCmp;
                  const SCEVConstant *C;
                  ConstantInt *CI;
                  if (C = dyn_cast<SCEVConstant>(expr->getStepRecurrence(*SE))) {
                      CI = ConstantInt::get(SE->getContext(), C->getAPInt());
                      IRBuilder<> Builder(gepi);
                      Value *CI_exd = Builder.CreateZExt(CI, gepi->getOperand(1)->getType());
                      Value *prefAddr = Builder.CreateAdd(gepi->getOperand(1), CI_exd);
                      gepi->setOperand(1, prefAddr);
                  } else continue;
                  Changed = true;
                  const SCEV *initLSCEV = createInitialPrefAddr(LSCEV);
                  SCEVExpander SCEVE(*SE, BB->getModule()->getDataLayout(), "prefaddr");
                  // expand code before loop starts to compute the prefetch address for the first iteration
                  Value *PrefPtrValue = SCEVE.expandCodeFor(initLSCEV, gepi->getOperand(1)->getType(), Incoming->getTerminator());

                  IRBuilder<> Builder(Incoming->getTerminator());

                  // this is allocating the temporary scalar variable
                  Value *tempAllocaPtr = Builder.CreateAlloca(gepi->getOperand(0)->getType()->getPointerElementType());
                  Value *initPrefAddr = Builder.CreateGEP(gepi->getOperand(0)->getType()->getPointerElementType(), gepi->getOperand(0), PrefPtrValue);
                  Value *initPrefVal = Builder.CreateLoad(gepi->getOperand(0)->getType()->getPointerElementType(), initPrefAddr);
                  Builder.CreateStore(initPrefVal, tempAllocaPtr);
                  
                  Builder.SetInsertPoint(SMemI);
                  Value *tempVal = Builder.CreateLoad(gepi->getOperand(0)->getType()->getPointerElementType(), tempAllocaPtr);
                  SMemI->setOperand(0, tempVal);
                  
                  Builder.SetInsertPoint(Inst->getNextNode()); // insert after the barrier instruction

                  Value *iter_next;
                  if(isa<PHINode>(loopGuardCmp->getOperand(0))){
                    iter_next = Builder.CreateAdd(loopGuardCmp->getOperand(0), ConstantInt::get(loopGuardCmp->getOperand(0)->getType(), C->getValue()->getSExtValue()));
                  } else {
                    iter_next = Builder.CreateAdd(dyn_cast<Instruction>(loopGuardCmp->getOperand(0))->getOperand(0), ConstantInt::get(loopGuardCmp->getOperand(0)->getType(), C->getValue()->getSExtValue()));
                  }

                  // insert boundary check to see if this is the last iteration
                  Value *check_res = Builder.CreateICmp(loopGuardCmp->getPredicate(), iter_next, loopGuardCmp->getOperand(1));
                  Instruction *ThenTerm , *ElseTerm;
                  SplitBlockAndInsertIfThenElse(check_res, (dyn_cast<ICmpInst>(check_res))->getNextNode(), &ThenTerm, &ElseTerm);

                  prefIfInsertPt = ElseTerm; // TODO: add check for if this already existed
                  Builder.SetInsertPoint(ElseTerm);
                  LMemI->moveBefore(ElseTerm);
                  Builder.CreateStore(LMemI, tempAllocaPtr);

                } else {
                  // errs() << "loop header terminator's previous instruction is not icmp.\n";
                }
              } else {
                // errs() << "loop header terminator inst is not a conditional br.\n";
              }
            } else {
              // errs() << "loop header terminator inst is not br.\n";
            }
            
          } else {
            // errs() << "unsupported stride type.\n";
          }
        } else {
          continue;
        }
      } else continue;
    }
  }

  return Changed;
}

bool GPUMemPrefetching::runOnFunction(Function &F) {
  // errs() << "Start Prefetching.\n";

  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  PDT = &getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  bool Changed = false;
  prefIfInsertPt = nullptr;
  prefLoop = nullptr;
  prefBlock = nullptr;
  prefBackBlock = nullptr;
  prefLoopGuardCmp = nullptr; 
  for (Loop *I : *LI)
    for (Loop *L : depth_first(I))
      Changed |= runOnLoop(L);

  return Changed;

}
