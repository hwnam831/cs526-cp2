#define CAST(T,N,U) T *N = dyn_cast<T>(U)
#include <iostream>
#include "llvm/Transforms/Scalar.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Debug.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/IR/IRBuilder.h"
using namespace llvm;

bool AnalyzeIncrement(Value* base, ValueMap<Value*, int>& map, int incr){
  bool isChanged = false;
  //base->dump();
  //errs() << "Analyzed increment : " << incr << "\n\n";
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
        case Instruction::BinaryOps::Or : {
          map.insert(std::make_pair(U, incr));
          isChanged = true;
          AnalyzeIncrement(U, map, incr);
          break;}
        case Instruction::BinaryOps::Mul: {
          Value *op1 = BO->getOperand(0);
          Value *op2 = BO->getOperand(1);
          int multiplier;
          if(isa<ConstantInt>(op1)){
            multiplier = dyn_cast<ConstantInt>(op1)->getSExtValue();
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
        case Instruction::BinaryOps::Shl: {
          Value *op1 = BO->getOperand(0);
          Value *op2 = BO->getOperand(1);
          int exponent;
          if(isa<ConstantInt>(op2)){
            exponent = dyn_cast<ConstantInt>(op2)->getSExtValue();
            map.insert(std::make_pair(U, incr<<exponent));
            isChanged = true;
            AnalyzeIncrement(U, map, incr<<exponent);
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

bool isZero(Value* ADDR, ValueMap<Value*, int>& IVincr, ValueMap<Value*, int>& TIDincr){
  if(CAST(ConstantInt, CINT, ADDR)){
    return !(CINT->getSExtValue());
  } else if(IVincr.count(ADDR) == 0 && TIDincr.count(ADDR) == 0){
    errs() << ADDR->getName();
    return false;
  } else if (IVincr.count(ADDR) && IVincr[ADDR] == 0){
    return true;
  } else if (TIDincr.count(ADDR) && TIDincr[ADDR] == 0){
    return true;
  }
  if (CAST(BinaryOperator,BO,ADDR)) {
    switch (BO->getOpcode()){
      case Instruction::BinaryOps::Add : {
        return  isZero(BO->getOperand(0), IVincr, TIDincr) &&
                isZero(BO->getOperand(1), IVincr, TIDincr);
        break;}
      case Instruction::BinaryOps::Sub : {
        return  isZero(BO->getOperand(0), IVincr, TIDincr) &&
                isZero(BO->getOperand(1), IVincr, TIDincr);
        break;}
      case Instruction::BinaryOps::Mul: {
        return  isZero(BO->getOperand(0), IVincr, TIDincr) ||
                isZero(BO->getOperand(1), IVincr, TIDincr);
        break;}
      case Instruction::BinaryOps::Or: {
        return  isZero(BO->getOperand(0), IVincr, TIDincr) &&
                isZero(BO->getOperand(1), IVincr, TIDincr);
        break;}
      case Instruction::BinaryOps::Shl: {
        return  isZero(BO->getOperand(0), IVincr, TIDincr);
        break;}
      default: {
        errs() << "ISzero: BO exception? ";
        ADDR->dump();
        break;}
    }
  } else if(isa<SExtInst>(ADDR)){
    return isZero(dyn_cast<SExtInst>(ADDR)->getOperand(0), IVincr, TIDincr);
  } else if(isa<ZExtInst>(ADDR)){
    return isZero(dyn_cast<ZExtInst>(ADDR)->getOperand(0), IVincr, TIDincr);
  } else {
    errs() << "Iszero: What is this? ";
    ADDR->dump();
  }
  return false;
}

//This may introduce a lot of dead code. May need ADCE after this
//NULL means this is const zero
//TODO: use IRbuilder to add instructions at the beginning of the loop block
Value* FindBaseAddress(IRBuilder<>& builder, Value* ADDR, ValueMap<Value*, int>& IVincr, ValueMap<Value*, int>& TIDincr){
  if (isZero(ADDR, IVincr, TIDincr)){
    return NULL;
  }
  if(CAST(ConstantInt, CINT, ADDR)){
    errs() << CINT->getSExtValue();
    return ADDR;
  } else if(IVincr.count(ADDR) == 0 && TIDincr.count(ADDR) == 0){
    errs() << ADDR->getName();
    //ADDR->dump();
    return ADDR;
  } else if (IVincr.count(ADDR) && IVincr[ADDR] == 0){
    errs() << "0";
    return NULL;
  } else if (TIDincr.count(ADDR) && TIDincr[ADDR] == 0){
    errs() << "0";
    return NULL;
  }
  if(CAST(SExtInst, SE, ADDR)){
    errs() << "(SEXT ";
    Value *op1 = FindBaseAddress(builder, SE->getOperand(0), IVincr, TIDincr);
    errs() << ")";
    return builder.CreateSExt(op1, SE->getDestTy());
  } else if(CAST(ZExtInst, ZE, ADDR)){
    errs() << "(ZEXT ";
    Value *op1 = FindBaseAddress(builder, ZE->getOperand(0), IVincr, TIDincr);
    errs() << ")";
    return builder.CreateZExt(op1, ZE->getDestTy());
  }else if (CAST(BinaryOperator,BO,ADDR)) {
    Value* op1 = BO->getOperand(0);
    Value* op2 = BO->getOperand(1);
    switch (BO->getOpcode()){
      case Instruction::BinaryOps::Add : {
        if (isZero(op1, IVincr, TIDincr) &&
            isZero(op2, IVincr, TIDincr)){
            return NULL;
        } else if (isZero(op1, IVincr, TIDincr)){
          return FindBaseAddress(builder, op2, IVincr, TIDincr);
        } else if (isZero(op2, IVincr, TIDincr)){
          return FindBaseAddress(builder, op1, IVincr, TIDincr);
        }
        errs() << "(Add ";
        Value* lhs = FindBaseAddress(builder, op1, IVincr, TIDincr);
        errs() << ", ";
        Value* rhs = FindBaseAddress(builder, op2, IVincr, TIDincr);
        errs() << ")";
        return builder.CreateAdd(lhs, rhs);
        break;}
      case Instruction::BinaryOps::Sub : {
        errs() << "(Sub ";
        Value* lhs = FindBaseAddress(builder, op1, IVincr, TIDincr);
        errs() << ", ";
        Value* rhs = FindBaseAddress(builder, op2, IVincr, TIDincr);
        errs() << ")";
        return builder.CreateSub(lhs, rhs);
        break;}
      case Instruction::BinaryOps::Or : {
        if (isZero(op1, IVincr, TIDincr) &&
            isZero(op2, IVincr, TIDincr)){
            return NULL;
        } else if (isZero(op1, IVincr, TIDincr)){
          return FindBaseAddress(builder, op2, IVincr, TIDincr);
        } else if (isZero(op2, IVincr, TIDincr)){
          return FindBaseAddress(builder, op1, IVincr, TIDincr);
        }
        errs() << "(Or ";
        Value* lhs = FindBaseAddress(builder, op1, IVincr, TIDincr);
        errs() << ", ";
        Value* rhs = FindBaseAddress(builder, op2, IVincr, TIDincr);
        errs() << ")";
        return builder.CreateOr(lhs, rhs);
        break;}
      case Instruction::BinaryOps::Mul: {
        if (isZero(op1, IVincr, TIDincr) ||
            isZero(op2, IVincr, TIDincr)){
            return NULL;
        }
        errs() << "(Mul ";
        Value* lhs = FindBaseAddress(builder, op1, IVincr, TIDincr);
        errs() << ", ";
        Value* rhs = FindBaseAddress(builder, op2, IVincr, TIDincr);
        errs() << ")";
        return builder.CreateMul(lhs, rhs);
        break;}
      case Instruction::BinaryOps::Shl: {
        if (isZero(op1, IVincr, TIDincr)){
          return NULL;
        } else if (isZero(op2, IVincr, TIDincr)){
          return FindBaseAddress(builder, op1, IVincr, TIDincr);
        }
        errs() << "(Shl ";
        Value* lhs = FindBaseAddress(builder, op1, IVincr, TIDincr);
        errs() << ", ";
        Value* rhs = FindBaseAddress(builder, op2, IVincr, TIDincr);
        errs() << ")";
        return builder.CreateShl(lhs, rhs);
        break;}
      default: {
        errs() << "BO exception? ";
        ADDR->dump();
        break;}
    }
  } else {
    errs() << "What is this? ";
    ADDR->dump();
  }
  return NULL;
}