// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <filesystem>
#include <random>
#include <vector>
#include <iostream>
#include <fstream>

#include "filemanager/FileManager.h"
#include "filemanager/impl/LocalFileManager.h"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "knowhere/feder/OdinANN.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node_data_mock_wrapper.h"
#include "utils.h"
#include "index/odinann/odinann_config.h"

#ifdef KNOWHERE_WITH_CARDINAL
constexpr char kOdinAnn[] = "ODINANN_DEPRECATED";
#else
constexpr char kOdinAnn[] = "ODINANN";
#endif

// Helper function to generate random float data
void
GenerateRandomFloatData(std::vector<float>& data, size_t num_elements) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    data.resize(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        data[i] = dis(gen);
    }
}

// Helper function to write binary data to file
void
WriteBinaryFile(const std::string& filename, const std::vector<float>& data) {
    std::ofstream file(filename, std::ios::binary);
    REQUIRE(file.is_open());

    uint32_t num_vectors = data.size() / 128;  // Assuming 128-dimensional vectors
    uint32_t dim = 128;

    file.write(reinterpret_cast<const char*>(&num_vectors), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&dim), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));

    file.close();
}

TEST_CASE("OdinANN basic test", "[odinann]") {
    // Generate test data
    const size_t num_vectors = 1000;
    const size_t dim = 128;
    const size_t num_queries = 10;
    
    std::vector<float> base_data;
    GenerateRandomFloatData(base_data, num_vectors * dim);
    
    std::vector<float> query_data;
    GenerateRandomFloatData(query_data, num_queries * dim);
    
    // Write base data to file
    const std::string data_file = "/tmp/odinann_test_data.bin";
    WriteBinaryFile(data_file, base_data);
    
    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
    auto odinann_index_pack = knowhere::Pack(file_manager);
    // Create index
    auto index = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(kOdinAnn, version, odinann_index_pack).value();

    // Configure index
    knowhere::Json json_config;
    json_config["metric_type"] = "L2";
    json_config["max_degree"] = 32;
    json_config["search_list_size"] = 64;
    json_config["pq_code_budget_gb"] = 0.001;
    json_config["build_dram_budget_gb"] = 1.0;
    json_config["disk_pq_dims"] = 0;
    json_config["accelerate_build"] = false;
    json_config["data_path"] = data_file;
    json_config["index_prefix"] = "/tmp/odinann_test_index";
    
    auto cfg = std::make_unique<knowhere::OdinANNConfig>();
    // auto cfg = index->CreateConfig();
    knowhere::Status status = knowhere::Config::Load(*cfg, json_config, knowhere::PARAM_TYPE::TRAIN);
    REQUIRE(status == knowhere::Status::success);
    
    // Create mock dataset
    auto base_dataset = knowhere::GenDataSet(num_vectors, dim, base_data.data());
    
    // Build index
    status = index.Build(base_dataset, json_config);
    REQUIRE(status == knowhere::Status::success);
    
    // Test serialization
    knowhere::BinarySet binset;
    status = index.Serialize(binset);
    REQUIRE(status == knowhere::Status::success);
    
    // Test deserialization
    status = index.Deserialize(binset, json_config);
    REQUIRE(status == knowhere::Status::success);
    
    // Test search
    auto query_dataset = knowhere::GenDataSet(num_queries, dim, query_data.data());
    json_config["k"] = 10;
    json_config["search_list_size"] = 64;
    
    status = knowhere::Config::Load(*cfg, json_config, knowhere::PARAM_TYPE::SEARCH);
    REQUIRE(status == knowhere::Status::success);
    
    auto result = index.Search(query_dataset, json_config, nullptr);
    REQUIRE(result.has_value());
    
    auto ids = result.value()->GetIds();
    auto distances = result.value()->GetDistance();
    REQUIRE(ids != nullptr);
    REQUIRE(distances != nullptr);
    
    // Clean up test files
    std::filesystem::remove(data_file);
    std::filesystem::remove("/tmp/odinann_test_index_pq_pivots.bin");
    std::filesystem::remove("/tmp/odinann_test_index_pq_compressed.bin");
    std::filesystem::remove("/tmp/odinann_test_index_disk.index");
    std::filesystem::remove("/tmp/odinann_test_index_disk.index_medoids.bin");
    std::filesystem::remove("/tmp/odinann_test_index_disk.index_centroids.bin");
}

TEST_CASE("OdinANN config test", "[odinann]") {
    // Test config validation
    auto cfg = knowhere::OdinANNConfig();
    
    // Test default values
    knowhere::Json json_config;
    json_config["metric_type"] = "L2";
    json_config["data_path"] = "/tmp/test_data.bin";
    json_config["index_prefix"] = "/tmp/test_index";
    
    knowhere::Status status = knowhere::Config::Load(cfg, json_config, knowhere::PARAM_TYPE::TRAIN);
    REQUIRE(status == knowhere::Status::success);
    
    // Check default values
    REQUIRE(cfg.max_degree.value() == 48);
    REQUIRE(cfg.search_list_size.has_value() == false);  // Should be set during CheckAndAdjust
    
    // Test CheckAndAdjust for TRAIN
    status = cfg.CheckAndAdjust(knowhere::PARAM_TYPE::TRAIN, nullptr);
    REQUIRE(status == knowhere::Status::success);
    REQUIRE(cfg.search_list_size.value() == 128);  // Default for build
    
    // Test CheckAndAdjust for SEARCH
    json_config["k"] = 10;
    status = knowhere::Config::Load(cfg, json_config, knowhere::PARAM_TYPE::SEARCH);
    REQUIRE(status == knowhere::Status::success);
    
    status = cfg.CheckAndAdjust(knowhere::PARAM_TYPE::SEARCH, nullptr);
    REQUIRE(status == knowhere::Status::success);
    REQUIRE(cfg.search_list_size.value() == 16);  // max(k, min_value)
}