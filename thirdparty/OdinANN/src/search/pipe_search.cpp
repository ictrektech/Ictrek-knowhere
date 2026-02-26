#include "aligned_file_reader.h"
#include "libcuckoo/cuckoohash_map.hh"
#include "neighbor.h"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include "timer.h"
#include "tsl/robin_set.h"
#include "utils.h"
#include "v2/page_cache.h"

#include <unistd.h>
#include <sys/syscall.h>

#ifndef USE_AIO
#include "liburing.h"
#endif

#include "knowhere/comp/knowhere_config.h"

namespace pipeann {
  struct io_t {
    Neighbor nbr;
    unsigned page_id;
    unsigned loc;
    IORequest *read_req;
    bool operator>(const io_t &rhs) const {
      return nbr.distance > rhs.nbr.distance;
    }

    bool operator<(const io_t &rhs) const {
      return nbr.distance < rhs.nbr.distance;
    }

    bool finished() {
      return read_req->finished;
    }
  };

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::pipe_search(const T *query1, const _u64 k_search, const _u32 mem_L, const _u64 l_search,
                                        TagT *res_tags, float *distances, const _u64 beam_width, QueryStats *stats) {

// send_best_read_req() -> send_read_req()
// calc_best_node() | -> compute_and_push_nbrs() -> compute_pq_dists() -> pq_dist_lookup()
// 			            | -> compute_exact_dists_and_push() -> dist_cmp->compare()
                                          
    QueryBuffer<T> *query_buf = pop_query_buf(query1);
#ifdef USE_AIO
    void *ctx = reader->get_ctx();
#else
    void *ctx = reader->get_ctx(IORING_SETUP_SQPOLL);  // use SQ polling only for pipe search.
#endif

    if (beam_width > MAX_N_SECTOR_READS_ODIN) {
      LOG(ERROR) << "Beamwidth can not be higher than MAX_N_SECTOR_READS_ODIN";
      crash();
    }

    // copy query to thread specific aligned and allocated memory (for distance
    // calculations we need aligned data)
    const T *query = query_buf->aligned_query_T;

    // reset query
    query_buf->reset();

    // pointers to buffers for data
    T *data_buf = query_buf->coord_scratch; // 存储从磁盘读取的原始向量数据副本，用于精确距离计算
#if defined(__ARM_NEON) && defined(__aarch64__)
    __builtin_prefetch((const char *) data_buf, 0, 2);
#else
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);
#endif
    // sector scratch
    char *sector_scratch = query_buf->sector_scratch; // 存储从磁盘读取的原始扇区数据，后续再从这里解析出向量节点的邻居列表、PQ 编码等信息

    // query <-> neighbor list
    float *dist_scratch = query_buf->aligned_dist_scratch; // 存储批量节点的 PQ 估算距离（临时结果）
    _u8 *pq_coord_scratch = query_buf->aligned_pq_coord_scratch; // 存储批量节点的PQ 编码序列（每个子空间 1 字节编码）,方便后续批量查表计算 PQ 距离

    Timer query_timer;
    std::vector<Neighbor> retset(mem_L + l_search * 10); // 当前查询向量的所有候选邻居节点(包含内存图的PQ距离筛选)，当搜索的节点数超过 l_search 时会动态扩容
    auto &visited = *(query_buf->visited);
    unsigned cur_list_size = 0;

    std::vector<Neighbor> full_retset; // 当前查询向量的所有候选邻居节点
    full_retset.reserve(l_search * 10);

    // query <-> PQ chunk centers distances
    float *pq_dists = query_buf->aligned_pqtable_dist_scratch; // 存储查询向量与所有 PQ 子空间聚类中心的距离表（预计算的核心表）

#ifndef OVERLAP_INIT
    pq_table.populate_chunk_distances(query, pq_dists);  // overlap with the first I/O.
#endif

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_pq_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids, const _u64 n_ids,
                                                               float *dists_out) {
      // 从索引数据中提取对应的PQ编码并保存到pq_coord_scratch
      ::aggregate_coords(ids, n_ids, this->data.data(), this->n_chunks, pq_coord_scratch);
      // 根据 pq_coord_scratch 中的编码，从 pq_dists 中查表得到子空间距离，累加后得到最终 PQ 距离
      ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists, dists_out);
    };

    // 计算精确距离并将邻居节点插入完整结果列表full_retset
    auto compute_exact_dists_and_push = [&](const char *node_buf, const unsigned id) -> float {
      T *node_fp_coords_copy = data_buf;
      memcpy(node_fp_coords_copy, node_buf, data_dim * sizeof(T));
      float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);
      full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
      return cur_expanded_dist;
    };

    uint64_t n_computes = 0;
    // 计算PQ累加距离（快速计算，模糊距离）并将邻居节点插入候选列表retset
    auto compute_and_push_nbrs = [&](const char *node_buf, unsigned &nk) {
      unsigned *node_nbrs = offset_to_node_nhood(node_buf);
      unsigned nnbrs = *(node_nbrs++); // 这里的邻居数固定为索引构建时设置的MAX_DEGREE
      unsigned nbors_cand_size = 0;
      for (unsigned m = 0; m < nnbrs; ++m) {
        // 找出未被访问的邻居向量节点放到node_nbrs
        if (visited.find(node_nbrs[m]) == visited.end()) {
          node_nbrs[nbors_cand_size++] = node_nbrs[m];
          visited.insert(node_nbrs[m]);
        }
      }

      n_computes += nbors_cand_size;
      if (nbors_cand_size) {
        // auto cpu1_st = std::chrono::high_resolution_clock::now();
        compute_pq_dists(node_nbrs, nbors_cand_size, dist_scratch);
        for (unsigned m = 0; m < nbors_cand_size; ++m) {
          const int nbor_id = node_nbrs[m];
          const float nbor_dist = dist_scratch[m];
          if (stats != nullptr) {
            stats->n_cmps++;
          }
          // 核心筛选条件：PQ距离超过候选集最差值，且候选集已满（这里的l_search从L_ONDISK_INDEX="100 150 200"取值） → 直接跳过
          if (nbor_dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search))
            continue;
          // 否则：将当前的邻居节点加入候选集
          Neighbor nn(nbor_id, nbor_dist, true);
          // Return position in sorted list where nn inserted
          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);  // may be overflow in retset...
          if (cur_list_size < l_search) {
            ++cur_list_size;
            if (unlikely(cur_list_size >= retset.size())) {
              retset.resize(2 * cur_list_size);
            }
          }
          // nk logs the best position in the retset that was updated due to
          // neighbors of n.
          if (r < nk)
            nk = r;
        }
        // auto cpu1_ed = std::chrono::high_resolution_clock::now();
        // stats->cpu_us1 += std::chrono::duration_cast<std::chrono::microseconds>(cpu1_ed - cpu1_st).count();
      }
    };

    auto add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids, float *dists) {
      for (_u64 i = 0; i < n_ids; ++i) {
        retset[cur_list_size++] = Neighbor(node_ids[i], dists[i], true);
        visited.insert(node_ids[i]);
      }
    };

    // stats.
    stats->io_us = 0;
    stats->io_us1 = 0;
    stats->cpu_us = 0;
    stats->cpu_us1 = 0;
    stats->cpu_us2 = 0;
    // search in in-memory index.

#ifdef DYN_PIPE_WIDTH
    int64_t cur_beam_width = 4;  // before converge.
#else
    int64_t cur_beam_width = beam_width;  // before converge.
#endif
    std::vector<unsigned> mem_tags(mem_L);
    std::vector<float> mem_dists(mem_L);

#ifdef OVERLAP_INIT
    if (mem_L) {
      mem_index_->search_with_tags_fast(query, mem_L, mem_tags.data(), mem_dists.data());
      add_to_retset(mem_tags.data(), std::min((_u64) mem_L, l_search), mem_dists.data());
    } else {
      // cannot overlap.
      pq_table.populate_chunk_distances_nt(query, pq_dists);
      // 添加第一个候选邻居节点初始化候选集
      compute_pq_dists(&medoids[0], 1, dist_scratch);
      add_to_retset(&medoids[0], 1, dist_scratch);
    }
#else
    if (mem_L) {
      mem_index_->search_with_tags_fast(query, mem_L, mem_tags.data(), mem_dists.data());
      compute_pq_dists(mem_tags.data(), mem_L, dist_scratch);
      add_to_retset(mem_tags.data(), std::min((_u64) mem_L, l_search), dist_scratch);
    } else {
      compute_pq_dists(&medoids[0], 1, dist_scratch);
      add_to_retset(&medoids[0], 1, dist_scratch);
    }
    std::sort(retset.begin(), retset.begin() + cur_list_size);
#endif

    std::queue<io_t> on_flight_ios;
    auto send_read_req = [&](Neighbor &item) -> bool {
      // 步骤1：标记节点状态为"已发送IO"，避免重复发送
      item.flag = false;

      // 步骤2：锁定节点对应的磁盘页面，保证并发安全
      // lock the corresponding page.
      this->lock_idx(idx_lock_table, item.id, std::vector<uint32_t>(), true);
      // 步骤3：计算节点对应的磁盘位置信息（核心映射关系）
      const unsigned loc = id2loc(item.id), pid = loc_sector_no(loc);

      // 步骤4：获取并计算IO缓冲区的地址（循环复用扇区缓冲区sector_scratch）
      // 按扇区粒度读取：磁盘 IO 的最小高效单位是扇区（4KB），按扇区读取能减少 IO 次数，提升磁盘吞吐量
      uint64_t &cur_buf_idx = query_buf->sector_idx;
      auto buf = sector_scratch + cur_buf_idx * size_per_io;
      // 步骤5：获取并构造IO请求对象（复用预分配的reqs数组）
      auto &req = query_buf->reqs[cur_buf_idx];
      req = IORequest(static_cast<_u64>(pid) * SECTOR_LEN_ODIN, size_per_io, buf, u_loc_offset(loc), max_node_len);
      // 步骤6：发送异步IO请求（无内存分配版）
      reader->send_read_no_alloc(req, ctx);

      // 步骤7：记录飞行中IO（便于后续轮询完成状态）
      on_flight_ios.push(io_t{item, pid, loc, &req});
      // 步骤8：更新缓冲区索引（循环复用，取模避免越界）
      cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS_ODIN;

      // 步骤9：统计IO次数（如果需要输出统计信息）
      if (stats != nullptr) {
        stats->n_ios++;
      }
      return true;
    };

    std::unordered_map<unsigned, char *> id_buf_map; // 向量id与其向量数据(真正从磁盘读取)缓冲区的映射
    auto poll_all = [&]() -> std::pair<int, int> {
      // poll once.
      reader->poll_all(ctx);
      unsigned n_in = 0, n_out = 0;
      while (!on_flight_ios.empty() && on_flight_ios.front().finished()) {
        io_t &io = on_flight_ios.front();
        id_buf_map.insert(std::make_pair(io.nbr.id, offset_to_loc((char *) io.read_req->buf, io.loc)));
        io.nbr.distance <= retset[cur_list_size - 1].distance ? ++n_in : ++n_out;
        // unlock the corresponding page.
        this->unlock_idx(idx_lock_table, io.nbr.id);
        on_flight_ios.pop();
      }
      return std::make_pair(n_in, n_out);
    };

    auto send_best_read_req = [&](uint32_t n) -> bool {
      // auto io_st = std::chrono::high_resolution_clock::now();
      unsigned n_sent = 0, marker = 0;
      while (marker < cur_list_size && n_sent < n) {
        while (marker < cur_list_size /* pool size */ &&
               (retset[marker].flag == false /* on flight */ ||
                id_buf_map.find(retset[marker].id) != id_buf_map.end() /* already read */)) {
          retset[marker].flag = false;  // even out the id_buf_map cost to O(1)
          ++marker;
        }
        if (marker >= cur_list_size) {
          break;  // nothing to send.
        }
        n_sent += send_read_req(retset[marker]);
      }
      // auto io_ed = std::chrono::high_resolution_clock::now();
      // stats->io_us += std::chrono::duration_cast<std::chrono::microseconds>(io_ed - io_st).count();
      return n_sent != 0;  // nothing to send.
    };

    auto calc_best_node = [&]() -> int {  // if converged.
      // auto cpu_st = std::chrono::high_resolution_clock::now();
      unsigned marker = 0, nk = cur_list_size, first_unvisited_eager = cur_list_size;
      /* calculate one from "already read" */
      for (marker = 0; marker < cur_list_size; ++marker) {
        if (!retset[marker].visited && id_buf_map.find(retset[marker].id) != id_buf_map.end()) {
          retset[marker].flag = false;  // even out the id_buf_map cost to O(1)
          retset[marker].visited = true;
          auto it = id_buf_map.find(retset[marker].id);
          auto [id, buf] = *it;
          compute_exact_dists_and_push(buf, id);
          compute_and_push_nbrs(buf, nk);
          break;
        }
      }

      /* guess the first unvisited vector (eager) */
      for (unsigned i = 0; i < cur_list_size; ++i) {
        if (!retset[i].visited && retset[i].flag /* not on-fly */
            && id_buf_map.find(retset[i].id) == id_buf_map.end() /* not already read */) {
          first_unvisited_eager = i;
          break;
        }
      }
      return first_unvisited_eager;
      // auto cpu_ed = std::chrono::high_resolution_clock::now();
      // stats->cpu_us += std::chrono::duration_cast<std::chrono::microseconds>(cpu_ed - cpu_st).count();
    };

    auto get_first_unvisited = [&]() -> int {
      int ret = -1;
      for (unsigned i = 0; i < cur_list_size; ++i) {
        if (!retset[i].visited) {
          ret = i;
          break;
        }
      }
      return ret;
    };

    auto print_state = [&]() {
      LOG(INFO) << "cur_list_size: " << cur_list_size;
      for (unsigned i = 0; i < cur_list_size; ++i) {
        LOG(INFO) << "retset[" << i << "]: " << retset[i].id << ", " << retset[i].distance << ", " << retset[i].flag
                  << ", " << retset[i].visited << ", " << (id_buf_map.find(retset[i].id) != id_buf_map.end());
      }
      LOG(INFO) << "On flight IOs: " << on_flight_ios.size();
      if (on_flight_ios.size() != 0) {
        auto &io = on_flight_ios.front();
        LOG(INFO) << "on_flight_io: " << io.nbr.id << ", " << io.nbr.distance << ", " << io.nbr.flag << ", "
                  << io.page_id << ", " << io.loc << ", " << io.finished();
      }
      usleep(500);
    };

    std::ignore = print_state;

    auto cpu2_st = std::chrono::high_resolution_clock::now();
    send_best_read_req(cur_beam_width - on_flight_ios.size());
    unsigned marker = 0, max_marker = 0; // 已处理的候选节点的最大索引（标记搜索进度，>=5 表示搜索进入稳定阶段）
#ifdef OVERLAP_INIT
    if (likely(mem_L != 0)) {
      pq_table.populate_chunk_distances_nt(query, pq_dists);  // overlap with the first I/O.
      compute_pq_dists(mem_tags.data(), mem_L, dist_scratch);
      for (unsigned i = 0; i < cur_list_size; ++i) {
        retset[i].distance = dist_scratch[i];
      }
      std::sort(retset.begin(), retset.begin() + cur_list_size);
    }
#endif

#ifndef STATIC_POLICY
    int cur_n_in = 0, cur_tot = 0;
#endif

    while (get_first_unvisited() != -1) {
      // poll to heap (best-effort) -> calc best from heap (skip if heap is empty) -> send IO (if can send) -> ...
      // auto io1_st = std::chrono::high_resolution_clock::now();
      auto [n_in, n_out] = poll_all();
      std::ignore = n_in;
      std::ignore = n_out;

#ifdef DYN_PIPE_WIDTH
#ifdef STATIC_POLICY
      constexpr int kBeamWidths[] = {4, 4, 8, 8, 16, 16, 24, 24, 32};
      cur_beam_width = kBeamWidths[std::min(max_marker / 5, 8u)];
#else
      if (max_marker >= 5 && n_in + n_out > 0) {
        cur_n_in += n_in;
        cur_tot += n_in + n_out;
        // converged, tune beam width.
        constexpr double kWasteThreshold = 0.1;  // 0.1 * 10
        if ((cur_tot - cur_n_in) * 1.0 / cur_tot <= kWasteThreshold) {
          cur_beam_width = cur_beam_width + 1;
          cur_beam_width = std::max(cur_beam_width, 4l);
          cur_beam_width = std::min((int64_t) beam_width, cur_beam_width);
        }
      }
#endif
#endif

      if ((int64_t) on_flight_ios.size() < cur_beam_width) {
#ifdef NAIVE_PIPE
        send_best_read_req(cur_beam_width - on_flight_ios.size());
#else
        send_best_read_req(1);
#endif
      }
      // auto io1_ed = std::chrono::high_resolution_clock::now();
      // stats->io_us1 += std::chrono::duration_cast<std::chrono::microseconds>(io1_ed - io1_st).count();
      marker = calc_best_node();
      max_marker = std::max(max_marker, marker);
    }
    auto cpu2_ed = std::chrono::high_resolution_clock::now();
    stats->cpu_us2 = std::chrono::duration_cast<std::chrono::microseconds>(cpu2_ed - cpu2_st).count();
    stats->cpu_us = n_computes;

    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    // copy k_search values
    _u64 t = 0;
    for (_u64 i = 0; i < full_retset.size() && t < k_search; i++) {
      if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
        continue;  // deduplicate.
      }
      res_tags[t] = id2tag(full_retset[i].id);
      if (distances != nullptr) {
        distances[t] = full_retset[i].distance;
      }
      t++;
    }

    push_query_buf(query_buf);

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
    return t;
  }

  template class SSDIndex<float>;
  template class SSDIndex<_s8>;
  template class SSDIndex<_u8>;

  // Implementation of pipe_search_with_outer_memindex: uses external mem_index instead of internal _mem_index
  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::pipe_search_with_outer_memindex(const T *query1, const _u64 k_search, const _u32 mem_L,
                                                            const _u64 l_search, TagT *res_tags, float *distances,
                                                            const _u64 beam_width, Index<T> *outer_mem_index,
                                                            QueryStats *stats) {
    // Sanity check: if outer_mem_index is nullptr but mem_L > 0, fall back to regular pipe_search
    if (outer_mem_index == nullptr && mem_L > 0) {
      LOG(WARNING) << "outer_mem_index is nullptr but mem_L > 0, falling back to regular pipe_search";
      return pipe_search(query1, k_search, mem_L, l_search, res_tags, distances, beam_width, stats);
    }

    QueryBuffer<T> *query_buf = pop_query_buf(query1);
#ifdef USE_AIO
    void *ctx = reader->get_ctx();
#else
    void *ctx = reader->get_ctx(IORING_SETUP_SQPOLL);
#endif

    if (beam_width > MAX_N_SECTOR_READS_ODIN) {
      LOG(ERROR) << "Beamwidth can not be higher than MAX_N_SECTOR_READS_ODIN";
      crash();
    }

    const T *query = query_buf->aligned_query_T;
    query_buf->reset();

    T *data_buf = query_buf->coord_scratch;
#if defined(__ARM_NEON) && defined(__aarch64__)
    __builtin_prefetch((const char *) data_buf, 0, 2);
#else
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);
#endif
    char *sector_scratch = query_buf->sector_scratch;
    float *dist_scratch = query_buf->aligned_dist_scratch;
    _u8 *pq_coord_scratch = query_buf->aligned_pq_coord_scratch;

    Timer query_timer;
    std::vector<Neighbor> retset(mem_L + l_search * 10);
    auto &visited = *(query_buf->visited);
    unsigned cur_list_size = 0;

    std::vector<Neighbor> full_retset;
    full_retset.reserve(l_search * 10);

    float *pq_dists = query_buf->aligned_pqtable_dist_scratch;

    // Lambda to batch compute query<-> node distances in PQ space
    auto compute_pq_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids, const _u64 n_ids,
                                                               float *dists_out) {
      ::aggregate_coords(ids, n_ids, this->data.data(), this->n_chunks, pq_coord_scratch);
      ::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists, dists_out);
    };

    auto compute_exact_dists_and_push = [&](const char *node_buf, const unsigned id) -> float {
      T *node_fp_coords_copy = data_buf;
      memcpy(node_fp_coords_copy, node_buf, data_dim * sizeof(T));
      float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);
      full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
      return cur_expanded_dist;
    };

    uint64_t n_computes = 0;
    auto compute_and_push_nbrs = [&](const char *node_buf, unsigned &nk) {
      unsigned *node_nbrs = offset_to_node_nhood(node_buf);
      unsigned nnbrs = *(node_nbrs++);
      unsigned nbors_cand_size = 0;
      for (unsigned m = 0; m < nnbrs; ++m) {
        if (visited.find(node_nbrs[m]) == visited.end()) {
          node_nbrs[nbors_cand_size++] = node_nbrs[m];
          visited.insert(node_nbrs[m]);
        }
      }

      n_computes += nbors_cand_size;
      if (nbors_cand_size) {
        compute_pq_dists(node_nbrs, nbors_cand_size, dist_scratch);
        for (unsigned m = 0; m < nbors_cand_size; ++m) {
          const int nbor_id = node_nbrs[m];
          const float nbor_dist = dist_scratch[m];
          if (stats != nullptr) {
            stats->n_cmps++;
          }
          if (nbor_dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search))
            continue;
          Neighbor nn(nbor_id, nbor_dist, true);
          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
          if (cur_list_size < l_search) {
            ++cur_list_size;
            if (unlikely(cur_list_size >= retset.size())) {
              retset.resize(2 * cur_list_size);
            }
          }
          if (r < nk)
            nk = r;
        }
      }
    };

    auto add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids, float *dists) {
      for (_u64 i = 0; i < n_ids; ++i) {
        retset[cur_list_size++] = Neighbor(node_ids[i], dists[i], true);
        visited.insert(node_ids[i]);
      }
    };

    if (stats != nullptr) {
      stats->io_us = 0;
      stats->io_us1 = 0;
      stats->cpu_us = 0;
      stats->cpu_us1 = 0;
      stats->cpu_us2 = 0;
    }

#ifdef DYN_PIPE_WIDTH
    int64_t cur_beam_width = 4;
#else
    int64_t cur_beam_width = beam_width;
#endif

    std::vector<unsigned> mem_tags(mem_L);
    std::vector<float> mem_dists(mem_L);

    // Search in external mem_index if provided and mem_L > 0
#ifdef OVERLAP_INIT
    if (mem_L && outer_mem_index != nullptr) {
      outer_mem_index->search_with_tags_fast(query, mem_L, mem_tags.data(), mem_dists.data());
      add_to_retset(mem_tags.data(), std::min((_u64) mem_L, l_search), mem_dists.data());
    } else {
      pq_table.populate_chunk_distances_nt(query, pq_dists);
      compute_pq_dists(&medoids[0], 1, dist_scratch);
      add_to_retset(&medoids[0], 1, dist_scratch);
    }
#else
    if (mem_L && outer_mem_index != nullptr) {
      outer_mem_index->search_with_tags_fast(query, mem_L, mem_tags.data(), mem_dists.data());
      compute_pq_dists(mem_tags.data(), mem_L, dist_scratch);
      add_to_retset(mem_tags.data(), std::min((_u64) mem_L, l_search), dist_scratch);
    } else {
      compute_pq_dists(&medoids[0], 1, dist_scratch);
      add_to_retset(&medoids[0], 1, dist_scratch);
    }
    std::sort(retset.begin(), retset.begin() + cur_list_size);
#endif

    std::queue<io_t> on_flight_ios;
    auto send_read_req = [&](Neighbor &item) -> bool {
      item.flag = false;

      this->lock_idx(idx_lock_table, item.id, std::vector<uint32_t>(), true);
      const unsigned loc = id2loc(item.id), pid = loc_sector_no(loc);

      uint64_t &cur_buf_idx = query_buf->sector_idx;
      auto buf = sector_scratch + cur_buf_idx * size_per_io;
      auto &req = query_buf->reqs[cur_buf_idx];
      req = IORequest(static_cast<_u64>(pid) * SECTOR_LEN_ODIN, size_per_io, buf, u_loc_offset(loc), max_node_len);
      reader->send_read_no_alloc(req, ctx);

      on_flight_ios.push(io_t{item, pid, loc, &req});
      cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS_ODIN;

      if (stats != nullptr) {
        stats->n_ios++;
      }
      return true;
    };

    std::unordered_map<unsigned, char *> id_buf_map;
    auto poll_all = [&]() -> std::pair<int, int> {
      reader->poll_all(ctx);
      unsigned n_in = 0, n_out = 0;
      while (!on_flight_ios.empty() && on_flight_ios.front().finished()) {
        io_t &io = on_flight_ios.front();
        id_buf_map.insert(std::make_pair(io.nbr.id, offset_to_loc((char *) io.read_req->buf, io.loc)));
        io.nbr.distance <= retset[cur_list_size - 1].distance ? ++n_in : ++n_out;
        this->unlock_idx(idx_lock_table, io.nbr.id);
        on_flight_ios.pop();
      }
      return std::make_pair(n_in, n_out);
    };

    auto send_best_read_req = [&](uint32_t n) -> bool {
      unsigned n_sent = 0, marker = 0;
      while (marker < cur_list_size && n_sent < n) {
        while (marker < cur_list_size &&
               (retset[marker].flag == false || id_buf_map.find(retset[marker].id) != id_buf_map.end())) {
          retset[marker].flag = false;
          ++marker;
        }
        if (marker >= cur_list_size) {
          break;
        }
        n_sent += send_read_req(retset[marker]);
      }
      return n_sent != 0;
    };

    auto calc_best_node = [&]() -> int {
      unsigned marker = 0, nk = cur_list_size, first_unvisited_eager = cur_list_size;
      for (marker = 0; marker < cur_list_size; ++marker) {
        if (!retset[marker].visited && id_buf_map.find(retset[marker].id) != id_buf_map.end()) {
          retset[marker].flag = false;
          retset[marker].visited = true;
          auto it = id_buf_map.find(retset[marker].id);
          auto [id, buf] = *it;
          compute_exact_dists_and_push(buf, id);
          compute_and_push_nbrs(buf, nk);
          break;
        }
      }

      for (unsigned i = 0; i < cur_list_size; ++i) {
        if (!retset[i].visited && retset[i].flag && id_buf_map.find(retset[i].id) == id_buf_map.end()) {
          first_unvisited_eager = i;
          break;
        }
      }
      return first_unvisited_eager;
    };

    auto get_first_unvisited = [&]() -> int {
      int ret = -1;
      for (unsigned i = 0; i < cur_list_size; ++i) {
        if (!retset[i].visited) {
          ret = i;
          break;
        }
      }
      return ret;
    };

    auto cpu2_st = std::chrono::high_resolution_clock::now();
    send_best_read_req(cur_beam_width - on_flight_ios.size());
    unsigned marker = 0, max_marker = 0;

#ifdef OVERLAP_INIT
    if (likely(mem_L != 0 && outer_mem_index != nullptr)) {
      pq_table.populate_chunk_distances_nt(query, pq_dists);
      compute_pq_dists(mem_tags.data(), mem_L, dist_scratch);
      for (unsigned i = 0; i < cur_list_size; ++i) {
        retset[i].distance = dist_scratch[i];
      }
      std::sort(retset.begin(), retset.begin() + cur_list_size);
    }
#endif

#ifndef STATIC_POLICY
    int cur_n_in = 0, cur_tot = 0;
#endif

    while (get_first_unvisited() != -1) {
      auto [n_in, n_out] = poll_all();
      std::ignore = n_in;
      std::ignore = n_out;

#ifdef DYN_PIPE_WIDTH
#ifdef STATIC_POLICY
      constexpr int kBeamWidths[] = {4, 4, 8, 8, 16, 16, 24, 24, 32};
      cur_beam_width = kBeamWidths[std::min(max_marker / 5, 8u)];
#else
      if (max_marker >= 5 && n_in + n_out > 0) {
        cur_n_in += n_in;
        cur_tot += n_in + n_out;
        constexpr double kWasteThreshold = 0.1;
        if ((cur_tot - cur_n_in) * 1.0 / cur_tot <= kWasteThreshold) {
          cur_beam_width = cur_beam_width + 1;
          cur_beam_width = std::max(cur_beam_width, 4l);
          cur_beam_width = std::min((int64_t) beam_width, cur_beam_width);
        }
      }
#endif
#endif

      if ((int64_t) on_flight_ios.size() < cur_beam_width) {
#ifdef NAIVE_PIPE
        send_best_read_req(cur_beam_width - on_flight_ios.size());
#else
        send_best_read_req(1);
#endif
      }
      marker = calc_best_node();
      max_marker = std::max(max_marker, marker);
    }
    auto cpu2_ed = std::chrono::high_resolution_clock::now();
    if (stats != nullptr) {
      stats->cpu_us2 = std::chrono::duration_cast<std::chrono::microseconds>(cpu2_ed - cpu2_st).count();
      stats->cpu_us = n_computes;
    }

    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    _u64 t = 0;
    for (_u64 i = 0; i < full_retset.size() && t < k_search; i++) {
      if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
        continue;
      }
      res_tags[t] = id2tag(full_retset[i].id);
      if (distances != nullptr) {
        distances[t] = full_retset[i].distance;
      }
      t++;
    }

    push_query_buf(query_buf);

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
    return t;
  }

  template size_t pipeann::SSDIndex<knowhere::bf16, unsigned int>::pipe_search_with_outer_memindex(
      knowhere::bf16 const *, unsigned long, unsigned int, unsigned long, unsigned int *, float *, unsigned long,
      pipeann::Index<knowhere::bf16, unsigned int> *, pipeann::QueryStats *);
  template size_t pipeann::SSDIndex<knowhere::fp16, unsigned int>::pipe_search_with_outer_memindex(
      knowhere::fp16 const *, unsigned long, unsigned int, unsigned long, unsigned int *, float *, unsigned long,
      pipeann::Index<knowhere::fp16, unsigned int> *, pipeann::QueryStats *);
}  // namespace pipeann
