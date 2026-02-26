#pragma once

#include "utils.h"
#if defined(__x86_64__)
#include <immintrin.h>
#elif defined(__aarch64__) && defined(USE_SVE2)
#include <arm_sve.h>  
#endif
#include "query_buf.h"

#include <sstream>
#include <string_view>

namespace pipeann {

#define NUM_PQ_CENTROIDS_ODIN 256
#define NUM_PQ_OFFSETS_ODIN 5
  template<typename T>
  class FixedChunkPQTable {
    // data_dim = n_chunks * chunk_size;
    //    _u64   n_chunks;    // n_chunks = # of chunks ndims is split into
    //    _u64   chunk_size;  // chunk_size = chunk size of each dimension chunk
    float *tables = nullptr;  // pq_tables = float* [[2^8 * [chunk_size]] * n_chunks]
    float *centroid = nullptr;
    _u64 ndims;  // ndims = chunk_size * n_chunks
    _u64 n_chunks;
    _u32 *chunk_offsets = nullptr;
    _u32 *rearrangement = nullptr;
    float *tables_T = nullptr;  // same as pq_tables, but col-major
    float *all_to_all_dists = nullptr;

   public:
    uint64_t all_to_all_dist_size() {
      return sizeof(float) * n_chunks * NUM_PQ_CENTROIDS_ODIN * NUM_PQ_CENTROIDS_ODIN;
    }

    FixedChunkPQTable() {
    }

    virtual ~FixedChunkPQTable() {
      if (tables != nullptr)
        delete[] tables;
      if (tables_T != nullptr)
        delete[] tables_T;
      if (rearrangement != nullptr)
        delete[] rearrangement;
      if (chunk_offsets != nullptr)
        delete[] chunk_offsets;
      if (centroid != nullptr)
        delete[] centroid;
      if (all_to_all_dists != nullptr)
        delete[] all_to_all_dists;
    }

    _u64 get_dim() {
      return ndims;
    }

    void load_pq_pivots_new(std::basic_istream<char> &reader, size_t num_chunks, size_t offset) {
      _u64 nr, nc;
      std::unique_ptr<_u64[]> file_offset_data;
      _u64 *file_offset_data_raw;
      pipeann::load_bin_impl<_u64>(reader, file_offset_data_raw, nr, nc, offset);
      file_offset_data.reset(file_offset_data_raw);

      if (nr != NUM_PQ_OFFSETS_ODIN) {
        LOG(ERROR) << "Pivot offset incorrect, # offsets = " << nr << ", but expecting " << NUM_PQ_OFFSETS_ODIN;
        crash();
      }

      pipeann::load_bin_impl<float>(reader, tables, nr, nc, file_offset_data[0] + offset);

      if ((nr != NUM_PQ_CENTROIDS_ODIN)) {
        LOG(ERROR) << "Num centers incorrect, centers = " << nr << " but expecting " << NUM_PQ_CENTROIDS_ODIN;
        crash();
      }

      this->ndims = nc;
      pipeann::load_bin_impl<float>(reader, centroid, nr, nc, file_offset_data[1] + offset);

      if ((nr != this->ndims) || (nc != 1)) {
        LOG(ERROR) << "Centroid file dim incorrect: row " << nr << ", col " << nc << " expecting " << this->ndims;
        crash();
      }

      pipeann::load_bin_impl<uint32_t>(reader, rearrangement, nr, nc, file_offset_data[2] + offset);
      if ((nr != this->ndims) || (nc != 1)) {
        LOG(ERROR) << "Rearrangement incorrect: row " << nr << ", col " << nc << " expecting " << this->ndims;
        crash();
      }

      pipeann::load_bin_impl<uint32_t>(reader, chunk_offsets, nr, nc, file_offset_data[3] + offset);

      if (nr != (uint64_t) num_chunks + 1 || nc != 1) {
        LOG(ERROR) << "Chunk offsets: nr=" << nr << ", nc=" << nc << ", expecting nr=" << num_chunks + 1 << ", nc=1.";
        crash();
      }

      this->n_chunks = num_chunks;
      LOG(INFO) << "Loaded PQ Pivots: #ctrs: " << NUM_PQ_CENTROIDS_ODIN << ", #dims: " << this->ndims
                << ", #chunks: " << this->n_chunks;
    }

    void post_load_pq_table() {
      // alloc and compute transpose
      pipeann::alloc_aligned((void **) &tables_T, 256 * ndims * sizeof(float), 64);
      // tables_T = new float[256 * ndims];
      for (_u64 i = 0; i < 256; i++) {
        for (_u64 j = 0; j < ndims; j++) {
          tables_T[j * 256 + i] = tables[i * ndims + j];
        }
      }

      // added this for easy PQ-PQ squared-distance calculations
      // TODO: Create only for StreamingMerger.
      all_to_all_dists = new float[256 * 256 * n_chunks];
      std::memset(all_to_all_dists, 0, 256 * 256 * n_chunks * sizeof(float));
      // should perhaps optimize later
      for (_u32 i = 0; i < 256; i++) {
        for (_u32 j = 0; j < 256; j++) {
          for (_u32 c = 0; c < n_chunks; c++) {
            for (_u64 d = chunk_offsets[c]; d < chunk_offsets[c + 1]; d++) {
              float diff = (tables[i * ndims + d] - tables[j * ndims + d]);
              all_to_all_dists[i * 256 * n_chunks + j * n_chunks + c] += diff * diff;
            }
          }
        }
      }
    }

    void load_pq_centroid_bin(const char *pq_table_file, size_t num_chunks, size_t offset = 0) {
      std::string pq_pivots_path(pq_table_file);
      _u64 nr, nc;

      get_bin_metadata(pq_table_file, nr, nc, offset);
      std::ifstream reader(pq_table_file, std::ios::binary | std::ios::ate);
      reader.seekg(0);
      load_pq_pivots_new(reader, num_chunks, offset);
      post_load_pq_table();
      LOG(INFO) << "Finished optimizing for PQ-PQ distance compuation";
    }

    void populate_chunk_distances(const T *query_vec, float *dist_vec) {
      memset(dist_vec, 0, 256 * n_chunks * sizeof(float));
#ifdef USE_SVE2
      static int32_t max_lanes = SVE_MAX_SUPPORT_BITS / (sizeof(float) * 8);
      svbool_t op_pred = svwhilelt_b32(0, max_lanes);
      // chunk wise distance computation
      for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
        // sum (q-c)^2 for the dimensions associated with this chunk
        float *chunk_dists = dist_vec + (256 * chunk);
        for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
          _u64 permuted_dim_in_query = rearrangement[j];
          const float *centers_dim_vec = tables_T + (256 * j);
          float query_centroid_gap = query_vec[permuted_dim_in_query] - centroid[permuted_dim_in_query];
          for (_u64 idx = 0; idx < 256; idx += max_lanes) {
            svfloat32_t diff_sv = svsub_n_f32_z(op_pred, svld1_f32(op_pred, centers_dim_vec + idx), query_centroid_gap);
            svfloat32_t chunk_dist_accu_sv = svmla_f32_z(op_pred, svld1_f32(op_pred, chunk_dists + idx), diff_sv, diff_sv);
            svst1_f32(op_pred, chunk_dists + idx, chunk_dist_accu_sv);
          }
        }
      }
#else
      // 遍历每个PQ子空间（chunk）
      for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
        // sum (q-c)^2 for the dimensions associated with this chunk
        float *chunk_dists = dist_vec + (256 * chunk);
        // 遍历该子空间包含的所有维度
        for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
          // 查询向量的维度重排（适配PQ分块的维度顺序），比如原始向量维度为768，分块固定为128，那么两个子空间的偏移为6，
          // 则第一个子空间包含的维度为rearrangement=[0,1,2,3,4,5]，第二个子空间包含的维度为rearrangement=[6,7,8,9,10,11]，以此类推
          _u64 permuted_dim_in_query = rearrangement[j];
          // 该维度下256个聚类中心的取值数组
          const float *centers_dim_vec = tables_T + (256 * j);
          // 遍历该维度的256个聚类中心
          for (_u64 idx = 0; idx < 256; idx++) {
            // 核心：计算去中心后的该子空间下查询向量值与聚类中心向量值的差值
            double diff = centers_dim_vec[idx] - (query_vec[permuted_dim_in_query] - centroid[permuted_dim_in_query]);
            chunk_dists[idx] += (float) (diff * diff);
          }
        }
      }
#endif
    }

    void populate_chunk_distances_nt(const T *query_vec, float *dist_vec) {
#ifdef USE_AVX512
      memset(dist_vec, 0, 256 * n_chunks * sizeof(float));
      // chunk wise distance computation
      for (_u64 chunk = 0; chunk < n_chunks; chunk++) {
        // sum (q-c)^2 for the dimensions associated with this chunk
        float *chunk_dists = dist_vec + (256 * chunk);
        for (_u64 j = chunk_offsets[chunk]; j < chunk_offsets[chunk + 1]; j++) {
          _u64 permuted_dim_in_query = rearrangement[j];
          float *centers_dim_vec = tables_T + (256 * j);
          for (_u64 idx = 0; idx < 256; idx += 16) {
            __m512i center_i = _mm512_stream_load_si512(centers_dim_vec + idx);  // avoid cache thrashing
            __m512 center_f = _mm512_castsi512_ps(center_i);
            __m512 query_f = _mm512_set1_ps(query_vec[permuted_dim_in_query] - centroid[permuted_dim_in_query]);
            __m512 diff = _mm512_sub_ps(center_f, query_f);
            __m512 diff_sq = _mm512_mul_ps(diff, diff);
            __m512 chunk_dists_v = _mm512_load_ps(chunk_dists + idx);
            chunk_dists_v = _mm512_add_ps(chunk_dists_v, diff_sq);
            _mm512_store_ps(chunk_dists + idx, chunk_dists_v);  // dist_vec should be in cache.
          }
        }
      }
#else
      return populate_chunk_distances(query_vec, dist_vec);
#endif
    }

    // computes PQ distance between comp_src and comp_dsts in efficient manner
    // comp_src: [nchunks]
    // comp_dsts: count * [nchunks]
    // dists: [count]
    // TODO (perf) :: re-order computation to get better locality
    void compute_distances_alltoall(const _u8 *comp_src, const _u8 *comp_dsts, float *dists, const _u32 count) {
      std::memset(dists, 0, count * sizeof(float));
      for (_u64 i = 0; i < count; i++) {
        for (_u64 c = 0; c < n_chunks; c++) {
          dists[i] +=
              all_to_all_dists[(_u64) comp_src[c] * 256 * n_chunks + (_u64) comp_dsts[i * n_chunks + c] * n_chunks + c];
        }
      }
    }

    // fp_vec: [ndims]
    // out_pq_vec : [nchunks]
    void deflate_vec(const float *fp_vec, _u8 *out_pq_vec) {
      // permute the vector according to PQ rearrangement, compute all distances
      // to 256 centroids and choose the closest (for each chunk)
      for (_u32 c = 0; c < n_chunks; c++) {
        float closest_dist = std::numeric_limits<float>::max();
        for (_u32 i = 0; i < 256; i++) {
          float cur_dist = 0;
          for (_u64 d = chunk_offsets[c]; d < chunk_offsets[c + 1]; d++) {
            float diff = (tables[i * ndims + d] - ((float) fp_vec[rearrangement[d]] - centroid[rearrangement[d]]));
            cur_dist += diff * diff;
          }
          if (cur_dist < closest_dist) {
            closest_dist = cur_dist;
            out_pq_vec[c] = (_u8) i;
          }
        }
      }
    }
  };  // namespace pipeann
}  // namespace pipeann