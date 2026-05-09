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

#include "gtest/gtest.h"
#include "kudnn.hpp"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>

static std::unordered_map<std::string, KuDNN::Layout> layoutMap = {
    {"KuDNN::Layout::UNDEFINED", KuDNN::Layout::UNDEFINED},
    {"KuDNN::Layout::A", KuDNN::Layout::A},
    {"KuDNN::Layout::AB", KuDNN::Layout::AB},
    {"KuDNN::Layout::BA", KuDNN::Layout::BA},
    {"KuDNN::Layout::ABC", KuDNN::Layout::ABC},
    {"KuDNN::Layout::ACB", KuDNN::Layout::ACB},
    {"KuDNN::Layout::BAC", KuDNN::Layout::BAC},
    {"KuDNN::Layout::BCA", KuDNN::Layout::BCA},
    {"KuDNN::Layout::CAB", KuDNN::Layout::CAB},
    {"KuDNN::Layout::CBA", KuDNN::Layout::CBA},
    {"KuDNN::Layout::ABCD", KuDNN::Layout::ABCD},
    {"KuDNN::Layout::ABDC", KuDNN::Layout::ABDC},
    {"KuDNN::Layout::ACBD", KuDNN::Layout::ACBD},
    {"KuDNN::Layout::ACDB", KuDNN::Layout::ACDB},
    {"KuDNN::Layout::ADBC", KuDNN::Layout::ADBC},
    {"KuDNN::Layout::ADCB", KuDNN::Layout::ADCB},
    {"KuDNN::Layout::BACD", KuDNN::Layout::BACD},
    {"KuDNN::Layout::BCDA", KuDNN::Layout::BCDA},
    {"KuDNN::Layout::CDAB", KuDNN::Layout::CDAB},
    {"KuDNN::Layout::CDBA", KuDNN::Layout::CDBA},
    {"KuDNN::Layout::DCAB", KuDNN::Layout::DCAB},
    {"KuDNN::Layout::ABCDE", KuDNN::Layout::ABCDE},
    {"KuDNN::Layout::ABCED", KuDNN::Layout::ABCED},
    {"KuDNN::Layout::ABDEC", KuDNN::Layout::ABDEC},
    {"KuDNN::Layout::ACBDE", KuDNN::Layout::ACBDE},
    {"KuDNN::Layout::ACDEB", KuDNN::Layout::ACDEB},
    {"KuDNN::Layout::ADECB", KuDNN::Layout::ADECB},
    {"KuDNN::Layout::BACDE", KuDNN::Layout::BACDE},
    {"KuDNN::Layout::BCDEA", KuDNN::Layout::BCDEA},
    {"KuDNN::Layout::CDEAB", KuDNN::Layout::CDEAB},
    {"KuDNN::Layout::CDEBA", KuDNN::Layout::CDEBA},
    {"KuDNN::Layout::DECAB", KuDNN::Layout::DECAB},
    {"KuDNN::Layout::ROW_MAJOR", KuDNN::Layout::ROW_MAJOR},
    {"KuDNN::Layout::COL_MAJOR", KuDNN::Layout::COL_MAJOR},
    {"KuDNN::Layout::NCHW", KuDNN::Layout::NCHW},
    {"KuDNN::Layout::NHWC", KuDNN::Layout::NHWC},
    {"KuDNN::Layout::NCDHW", KuDNN::Layout::NCDHW},
    {"KuDNN::Layout::NDHWC", KuDNN::Layout::NDHWC},
    {"KuDNN::Layout::OIHW", KuDNN::Layout::OIHW},
    {"KuDNN::Layout::HWIO", KuDNN::Layout::HWIO},
    {"KuDNN::Layout::HWOI", KuDNN::Layout::HWOI},
    {"KuDNN::Layout::OHWI", KuDNN::Layout::OHWI},
    {"KuDNN::Layout::OHWO", KuDNN::Layout::OHWO},
    {"KuDNN::Layout::IOHW", KuDNN::Layout::IOHW},
};

static std::unordered_map<std::string, KuDNN::Element::TypeT> typeTMap = {
    {"UNDEFINED", KuDNN::Element::TypeT::UNDEFINED},
    {"KuDNN::Element::TypeT::F32", KuDNN::Element::TypeT::F32},
    {"KuDNN::Element::TypeT::F16", KuDNN::Element::TypeT::F16},
    {"KuDNN::Element::TypeT::BF16", KuDNN::Element::TypeT::BF16},
    {"KuDNN::Element::TypeT::S32", KuDNN::Element::TypeT::S32},
    {"KuDNN::Element::TypeT::S8", KuDNN::Element::TypeT::S8},
    {"KuDNN::Element::TypeT::U8", KuDNN::Element::TypeT::U8},
};

static std::vector<KuDNN::TensorInfo> g_Tensor_data;

class CustomEnvironment : public ::testing::Environment {
public:
    explicit CustomEnvironment(std::string& filename)
        : filename_(filename) {}

    void SetUp() override {
        std::cout << "filename: " << filename_ << std::endl;
        ReadTensorData(filename_);
    }

    void TearDown() override {
    }

private:
    int *ReadShapeData(const std::string& shapedata) {
        matrix.clear();
        size_t pos = 0;
        size_t next;
        while ((next = shapedata.find(',', pos)) != std::string::npos) {
            matrix.push_back(std::stoi(shapedata.substr(pos, next - pos)));
            pos = next + 1;
        }
        return matrix.data();
    }

    KuDNN::Element::TypeT ReadType(const std::string& typedata) {
        auto it = typeTMap.find(typedata);
        if (it != typeTMap.end()) {
            return it->second;
        }

        return KuDNN::Element::TypeT::UNDEFINED;
    }

    KuDNN::Layout ReadLayout(const std::string& layoutdata) {
        auto it = layoutMap.find(layoutdata);
        if (it != layoutMap.end()) {
            return it->second;
        }

        return KuDNN::Layout::UNDEFINED;
    }

    void ReadTensorData(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        std::string delimiter = "}, ";
        std::string token;
        std::string line;
        size_t pos = 0;
        std::getline(file, line);
        while (std::getline(file, line)) {
            std::cout << "line:" << line << std::endl;
            pos = line.find(delimiter);
            token = line.substr(2, pos);
            KuDNN::Shape shape(ReadShapeData(token), matrix.size());
            line.erase(0, pos + delimiter.length());

            pos = line.find(", ");
            token = line.substr(0, pos);
            KuDNN::Element::TypeT typeT = ReadType(token);
            line.erase(0, pos + 2);

            KuDNN::Layout layout = ReadLayout(line.substr(0, line.find("}")));

            g_Tensor_data.push_back({shape, typeT, layout});
        }
    }

    std::string filename_;
    std::vector<int> matrix;
};
