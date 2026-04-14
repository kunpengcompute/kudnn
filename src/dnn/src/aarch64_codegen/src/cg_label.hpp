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

#ifndef CG_LABEL_HPP
#define CG_LABEL_HPP

#include "cg_memory.hpp"
#include "cg_inner.hpp"

namespace ARMCG {

struct JmpLabel {
    // type of partially applied function for encoding
    using EncFunc = std::function<uint32_t(int64_t)>;
    explicit JmpLabel(const EncFunc &encFuncIn, size_t endOfJmpIn = 0)
        : endOfJmp(endOfJmpIn), encFunc(encFuncIn) {}
    
    size_t endOfJmp; // offset from top to the end address of jmp
    EncFunc encFunc;
};

class LabelManager;

class Label {
    mutable LabelManager *mgr;
    mutable int id;
    friend class LabelManager;

public:
    Label() : mgr(nullptr), id(0) {}
    Label(const Label &rhs);
    Label &operator=(const Label &rhs);
    ~Label();

    void Clear()
    {
        mgr = nullptr;
        id = 0;
    }
    int GetId() const
    {
        return id;
    }
    const uint8_t *GetAddress() const;
};

class LabelManager {
    // for Label class
    struct ClabelVal {
        explicit ClabelVal(size_t offsetIn = 0) : offset(offsetIn), refCount(1) {}
        size_t offset;
        int refCount;
    };
    using ClabelDefList = std::unordered_map<int, ClabelVal>;
    using ClabelUndefList = std::unordered_multimap<int, const JmpLabel>;
    using LabelPtrList = std::unordered_set<Label *>;
    CodeArray *base_;
    // global : stateList_.front(), local : stateList_.back()
    mutable int labelId_;
    ClabelDefList clabelDefList_;
    ClabelUndefList clabelUndefList_;
    LabelPtrList labelPtrList_;

    int GetId(const Label &label) const
    {
        if (label.id == 0) {
            label.id = labelId_++;
        }
        return label.id;
    }
    template <class DefList, class UndefList, class T>
    void DefineInner(DefList &defList, UndefList &undefList, const T &labelId, size_t addrOffset)
    {
        // add label
        typename DefList::value_type item(labelId, addrOffset);
        std::pair<typename DefList::iterator, bool> ret = defList.insert(item);
        if (!ret.second) {
            throw Error(Err::LABEL_IS_REDEFINED);
        }
        // search undefined label
        for (;;) {
            typename UndefList::iterator itr = undefList.find(labelId);
            if (itr == undefList.end()) {
                break;
            }
            const JmpLabel *jmp = &itr->second;
            const size_t offset = jmp->endOfJmp;
            int64_t labelOffset = static_cast<int64_t>(addrOffset - offset) * CSIZE;
            uint32_t disp = jmp->encFunc(labelOffset);
            if (base_->IsAutoGrow()) {
                base_->Save(offset, addrOffset, jmp->encFunc);
            } else {
                base_->Rewrite(offset, disp);
            }
            undefList.erase(itr);
        }
    }
    template <class DefList, class T>
    bool GetOffsetInner(const DefList &defList, size_t &offset, const T &label) const
    {
        typename DefList::const_iterator i = defList.find(label);
        if (i == defList.end()) {
            return false;
        }
        offset = i->second.offset;
        return true;
    }
    friend class Label;
    void IncRefCount(int id, Label *label)
    {
        clabelDefList_[id].refCount++;
        labelPtrList_.insert(label);
    }
    void DecRefCount(int id, Label *label)
    {
        labelPtrList_.erase(label);
        ClabelDefList::iterator i = clabelDefList_.find(id);
        if (i == clabelDefList_.end()) {
            return;
        }
        if (i->second.refCount == 1) {
            clabelDefList_.erase(id);
        } else {
            --i->second.refCount;
        }
    }

    template <class T>
    bool HasUndefinedLabelInner(const T &list) const
    {
        return !list.empty();
    }
    // detach all labels linked to LabelManager
    void ResetLabelPtrList()
    {
        for (LabelPtrList::iterator i = labelPtrList_.begin(), ie = labelPtrList_.end(); i != ie; ++i) {
            (*i)->Clear();
        }
        labelPtrList_.clear();
    }

public:
    LabelManager()
    {
        Reset();
    }
    ~LabelManager()
    {
        ResetLabelPtrList();
    }
    void Reset()
    {
        base_ = nullptr;
        labelId_ = 1;
        clabelDefList_.clear();
        clabelUndefList_.clear();
        ResetLabelPtrList();
    }
    void Set(CodeArray *base)
    {
        base_ = base;
    }
    void DefineClabel(Label &label)
    {
        DefineInner(clabelDefList_, clabelUndefList_, GetId(label), base_->size_);
        label.mgr = this;
        labelPtrList_.insert(&label);
    }
    void Assign(Label &dst, const Label &src)
    {
        ClabelDefList::const_iterator i = clabelDefList_.find(src.id);
        if (i == clabelDefList_.end()) {
            throw Error(Err::LABEL_ISNOT_SET_BY_L);
        }
        DefineInner(clabelDefList_, clabelUndefList_, dst.id, i->second.offset);
        dst.mgr = this;
        labelPtrList_.insert(&dst);
    }
    bool GetOffset(size_t &offset, const Label &label) const
    {
        return GetOffsetInner(clabelDefList_, offset, GetId(label));
    }
    void AddUndefinedLabel(const Label &label, const JmpLabel &jmp)
    {
        clabelUndefList_.insert(ClabelUndefList::value_type(label.id, jmp));
    }

    bool HasUndefClabel() const
    {
        return HasUndefinedLabelInner(clabelUndefList_);
    }
    uint8_t *GetCode() const
    {
        return base_->GetCode();
    }
    bool IsReady() const
    {
        return !base_->IsAutoGrow() || base_->IsCalledCalcJmpAddress();
    }
};

inline Label::Label(const Label &rhs)
{
    id = rhs.id;
    mgr = rhs.mgr;
    if (mgr) {
        mgr->IncRefCount(id, this);
    }
}

inline Label &Label::operator=(const Label &rhs)
{
    if (id) {
        throw Error(Err::LABEL_IS_ALREADY_SET_BY_L);
    }
    id = rhs.id;
    mgr = rhs.mgr;
    if (mgr) {
        mgr->IncRefCount(id, this);
    }
    return *this;
}

inline Label::~Label()
{
    if (id && mgr) {
        mgr->DecRefCount(id, this);
    }
}

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
#endif

inline const uint8_t *Label::GetAddress() const
{
    if (mgr == nullptr || !mgr->IsReady()) {
        return nullptr;
    }
    size_t offset;
    if (!mgr->GetOffset(offset, *this)) {
        return nullptr;
    }
    // GetCode() is always a multiple of 4
    return mgr->GetCode() + offset;
}

} // namespace ARMCG

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#endif // CG_LABEL_HPP
