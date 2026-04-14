/* 
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 *
 * KuDNN is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *        http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef CG_JIT_GENERATOR_HPP
#define CG_JIT_GENERATOR_HPP

#include <mutex>
#include <string>
#include "tools/cg_cpu_isa_traits.hpp"
#include "cg_codegen.hpp"
#include "src/cg_reg.hpp"
#include "../../include/service/kudnn_service.hpp"

#define MAX_CODENAME_LEN \
    (MAX_FNAME_LEN - sizeof(DUMP_BASE_FNAME) - sizeof(DUMP_EXT_FNAME))

namespace ARMCG {

struct CCompatible {
    static void *operator new(size_t sz)
    {
        return ::KuDNN::Service::AlignedAlloc(sz);
    }
    static void *operator new[](size_t sz)
    {
        return ::KuDNN::Service::AlignedAlloc(sz);
    }
    static void operator delete(void *p)
    {
        ::KuDNN::Service::Deallocate(p);
    }
    static void operator delete[](void *p)
    {
        ::KuDNN::Service::Deallocate(p);
    }

protected:
    // This member is intended as a check for whether all necessary members in
    // derived classes have been allocated. Consequently status::out_of_memory
    // should be returned if a fail occurs. This member should be modified only
    // in a constructor.
    bool isInitialized = true;
};

// Callee-saved registers
constexpr Operand::Code ABI_SAVE_GPR_REGS[]
        = {Operand::Code::X16, Operand::Code::X17,
           Operand::Code::X19, Operand::Code::X20,
           Operand::Code::X21, Operand::Code::X22,
           Operand::Code::X23, Operand::Code::X24,
           Operand::Code::X25, Operand::Code::X26,
           Operand::Code::X27, Operand::Code::X28};

// See "Procedure Call Standsard for the ARM 64-bit Architecture (AArch64)"
const XReg ABI_PARAM1(static_cast<uint32_t>(Operand::Code::X0));
const XReg ABI_PARAM2(static_cast<uint32_t>(Operand::Code::X1));
const XReg ABI_PARAM3(static_cast<uint32_t>(Operand::Code::X2));
const XReg ABI_PARAM4(static_cast<uint32_t>(Operand::Code::X3));
const XReg ABI_PARAM5(static_cast<uint32_t>(Operand::Code::X4));
const XReg ABI_PARAM6(static_cast<uint32_t>(Operand::Code::X5));
const XReg ABI_PARAM7(static_cast<uint32_t>(Operand::Code::X6));
const XReg ABI_PARAM8(static_cast<uint32_t>(Operand::Code::X7));
const XReg ABI_NOT_PARAM1(static_cast<uint32_t>(Operand::Code::X15));

class JitGenerator : public CodeGenerator, public CCompatible {
public:
    using CCompatible::operator new;
    using CCompatible::operator new[];
    using CCompatible::operator delete;
    using CCompatible::operator delete[];

    const uint64_t cpuSveLen = static_cast<uint64_t>(GetSveLength());

    const WReg W_TMP_0 = w23;
    const WReg W_TMP_1 = w24;
    const WReg W_TMP_2 = w25;
    const WReg W_TMP_3 = w26;
    const WReg W_TMP_4 = w27;
    const XReg X_TMP_0 = x23;
    const XReg X_TMP_1 = x24;
    const XReg X_TMP_2 = x25;
    const XReg X_TMP_3 = x26;
    const XReg X_TMP_4 = x27;
    const XReg xDefaultAddr = x28;
    const XReg xSP = x21;
    const XReg xTranslatorStack = x22;
    const PReg pAllZero = p10;
    const PReg P_NOT_256 = p13;
    const PReg P_NOT_128 = p14;
    const PReg pAllOne = p0;

    const std::vector<XReg> xTmpVec
            = {X_TMP_0, X_TMP_1, X_TMP_2, X_TMP_3, X_TMP_4};
    const int xTmpVecSize = xTmpVec.size();

    const XReg param1 = ABI_PARAM1;
    constexpr static size_t TRANSLATOR_STACK_OFFSET = VAL_1024 * VAL_128;
    constexpr static uint32_t DUMMY_IDX = VAL_99;

    inline size_t GetSizeOfAbiSaveRegs()
    {
        return sizeOfAbiSaveRegs;
    }

    void Preamble()
    {
        using namespace Util;
        SveLen sveLen = GetSveLength();

        Stp(x29, x30, PrePtr(sp, SIZE_NEG_16));
        /* x29 is a frame pointer. */
        Mov(x29, sp);
        Sub(sp, sp, static_cast<int32_t>(preservedStackSize) - VAL_16);

        /* x9 can be used as a temporal register. */
        Mov(x9, sp);

        if (vregToPreserve) {
            St4((v8.d - v11.d)[0], PostPtr(x9, vregLenPreserve * VAL_4));
            St4((v12.d - v15.d)[0], PostPtr(x9, vregLenPreserve * VAL_4));
        }
        for (size_t i = 0; i < numAbiSaveGprRegs; i += VAL_2) {
            Stp(XReg(static_cast<uint32_t>(ABI_SAVE_GPR_REGS[i])),
                XReg(static_cast<uint32_t>(ABI_SAVE_GPR_REGS[i + VAL_1])),
                PostPtr(x9, xregLen * VAL_2));
        }

        if (sveLen != SveLen::VectorComputation_NONE) { /* VectorComputation is available. */
            Ptrue(pAllOne.b);
            Pfalse(pAllZero.b);
        }
        if (sveLen >= SveLen::VectorComputation_256) {
            Ptrue(P_NOT_128.b, Pattern::VL16);
            Not(P_NOT_128.b, pAllOne / PredType::T_Z, P_NOT_128.b);
        }
        if (sveLen >= SveLen::VectorComputation_512) {
            Ptrue(P_NOT_256.b, Pattern::VL32);
            Not(P_NOT_256.b, pAllOne / PredType::T_Z, P_NOT_256.b);
        }

        Mov(xSP, sp);
        SubImm(xTranslatorStack, xSP, TRANSLATOR_STACK_OFFSET, X_TMP_0);
    }

    void Postamble()
    {
        using namespace Util;

        Mov(x9, sp);

        if (vregToPreserve) {
            Ld4((v8.d - v11.d)[0], PostPtr(x9, vregLenPreserve * VAL_4));
            Ld4((v12.d - v15.d)[0], PostPtr(x9, vregLenPreserve * VAL_4));
        }

        for (size_t i = 0; i < numAbiSaveGprRegs; i += VAL_2) {
            Ldp(XReg(static_cast<uint32_t>(ABI_SAVE_GPR_REGS[i])),
                XReg(static_cast<uint32_t>(ABI_SAVE_GPR_REGS[i + 1])),
                PostPtr(x9, xregLen * VAL_2));
        }

        Add(sp, sp, static_cast<int64_t>(preservedStackSize) - VAL_16);
        Ldp(x29, x30, PostPtr(sp, VAL_16));
        Ret();
    }

    // Disallow char-based labels completely
    void L(const char *label) = delete;
    void L(Label &label)
    {
        CodeGenerator::L(label);
    }

    void LAligned(Label &label, int alignment = VAL_16)
    {
        Align(alignment);
        L(label);
    }

    template <typename T>
     XReg AddrOff(const  XReg &base, const T off,
            const  XReg &addr, const  XReg &xTmp)
    {
        if (off == 0) {
            return base;
        }

        AddImm(addr, base, off, xTmp);
        return addr;
    }

    template <typename PRegBHSD, typename T>
    void SetPreg(const PRegBHSD &p, T tailSize,
            const XReg xTmp0 = XReg(DUMMY_IDX),
            const XReg xTmp1 = XReg(DUMMY_IDX))
    {
        assert(tailSize <= VAL_64); // Implemented only for "VectorComputation size <=  512"

        switch (tailSize) {
            case VAL_0: Pfalse(PRegB(p.getIdx())); return;
            case VAL_1: Ptrue(p, Pattern::VL1); return;
            case VAL_2: Ptrue(p, Pattern::VL2); return;
            case VAL_3: Ptrue(p, Pattern::VL3); return;
            case VAL_4: Ptrue(p, Pattern::VL4); return;
            case VAL_5: Ptrue(p, Pattern::VL5); return;
            case VAL_6: Ptrue(p, Pattern::VL6); return;
            case VAL_7: Ptrue(p, Pattern::VL7); return;
            case VAL_8: Ptrue(p, Pattern::VL8); return;
            case VAL_16: Ptrue(p, Pattern::VL16); return;
            case VAL_32: Ptrue(p, Pattern::VL32); return;
            case VAL_64: Ptrue(p, Pattern::VL64); return;
        }

        assert(xTmp0.GetIdx() != DUMMY_IDX && xTmp1.GetIdx() != DUMMY_IDX);
        MovImm(xTmp0, 0);
        MovImm(xTmp1, tailSize);
        Whilelt(p, xTmp0, xTmp1);
    }

// convenient wrapper instructions
#include "src/cg_mnemonic_tpl.hpp"
#include "src/cg_mnemonic_uni.hpp"

    /* A utility function to process f32 tail (load, store or other) depending
     * on tail size, stored in Reg64. Tail size must be value from 0 to 3/7
     * (Xmm/Ymm). Tail process functions require integer as argument to specify
     * behavior for each tail size.
     *
     * Only supported for Xmm and Ymm.
     */
    template <CPUISAT isa>
    void RuntimeTailProcess(const XReg &regTail,
            const XReg &regTmp,
            const std::function<void(int)> &tailProcess)
    {
        constexpr int simdWYmm = BIT_8;
        constexpr int f32Bits = sizeof(float) * BIT_8;
        const auto simdW = CpuIsaTraits<isa>::vlen * BIT_8 / f32Bits;
        assert(simdW != CpuIsaTraits<isa>::vlen * BIT_8 / f32Bits);

        Label labelTbl;
        Label labelTblEnd;
        Label lCase[simdWYmm];

        Adr(regTmp, labelTbl);
        MovImm(X_TMP_0, sizeof(void *));
        Madd(xDefaultAddr, regTail, X_TMP_0, regTmp);
        Br(xDefaultAddr);

        // create jump table
        L(labelTbl);
        for (size_t i = 0; i < simdW; i++) {
            PutL(lCase[i]);
        }

        // cases for each tail size - from 0 to 3/7
        L(lCase[0]);
        B(labelTblEnd);
        for (size_t i = 1; i < simdW; i++) {
            L(lCase[i]);
            tailProcess(i);
            B(labelTblEnd);
        }
        L(labelTblEnd);
    }

    JitGenerator(const JitGenerator &) = delete;
    JitGenerator &operator=(const JitGenerator &) = delete;

    explicit JitGenerator(void *codePtr = nullptr, size_t codeSize = CODE_SIZE_256_1024, bool useAutogrow = true)
        : CodeGenerator(codeSize, (codePtr == nullptr && useAutogrow) ? AUTO_GROW : codePtr) {}
    ~JitGenerator() override {}

    virtual const char *Name() const = 0;
    virtual const char *SourceFile() const = 0;

    const uint8_t *JitKer() const
    {
        return jitKer;
    }

    template <typename... kernel_args_t>
    void operator()(kernel_args_t... args) const
    {
        using JitKernelFuncT = void (*)(const kernel_args_t... args);
        auto *fptr = reinterpret_cast<JitKernelFuncT>(jitKer);
        (*fptr)(std::forward<kernel_args_t>(args)...);
    }

    virtual bool CreateKernel()
    {
        Generate();
        jitKer = GetCode();
        return (jitKer) ? true : false;
    }

    template <class F>
    const F GetAsmCode() const
    {
        return CodeGenerator::GetCode<F>();
    }

    void DumpJitCode(const void *code, size_t codeSize, const char *codeName)
    {
        const char *value = ::getenv("ONEDNN_JIT_DUMP");
        if (code != nullptr && value != nullptr) {
            FILE *fp = fopen(codeName, "wb+");
            // Failure to dump code is not fatal
            if (fp) {
                size_t unused = fwrite(code, codeSize, 1, fp);
                Util::Unused(unused);
                fclose(fp);
            }
        }
    }

    void RegisterJitCode(const void *code, size_t codeSize,
        const char *codeName, const char *sourceFileName)
    {
        // The #ifdef guards are required to avoid generating a function that only
        // consists of lock and unlock code
        static std::mutex m;
        static int uniqueId = 0;
        std::lock_guard<std::mutex> guard(m);
        std::string uniqueCodeName = codeName;
        uniqueCodeName += ".";
        uniqueCodeName += std::to_string(uniqueId++);
        std::string name = DUMP_BASE_FNAME;
        name += uniqueCodeName.c_str();
        name += DUMP_EXT_FNAME;
        DumpJitCode(code, codeSize, name.c_str());
        Util::Unused(sourceFileName);
    }

protected:
    virtual void Generate() = 0;
    const uint8_t *jitKer = nullptr;

private:
    const uint8_t *GetCode()
    {
        this->Ready();
        if (!IsInitialized()) {
            return nullptr;
        }
        const uint8_t *code = CodeGenerator::GetCode();
        RegisterJitCode(code, GetSize() * sizeof(uint32_t), Name(), SourceFile());
        return code;
    }

    static inline bool IsInitialized()
    {
        // At the moment, armcg does not have GetError() so that return dummy result.
        return true;
    }

    const size_t xregLen = 8;
    const size_t vregLenPreserve = 8; // Only bottom 8byte must be preserved.
    const size_t vregToPreserve = 8; // VREG8 - VREG15

    const size_t numAbiSaveGprRegs
            = sizeof(ABI_SAVE_GPR_REGS) / sizeof(ABI_SAVE_GPR_REGS[0]);

    const size_t preservedStackSize = xregLen * (2 + numAbiSaveGprRegs)
            + vregLenPreserve * vregToPreserve;

    const size_t sizeOfAbiSaveRegs = numAbiSaveGprRegs * x0.GetBit() / 8
            + vregToPreserve * vregLenPreserve;
};

} // ARMCG

#endif // CG_JIT_GENERATOR_HPP
