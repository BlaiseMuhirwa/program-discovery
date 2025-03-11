#pragma once

#include <flatnav/index/Index.h>
#include <functional>

namespace flatnav {

template <typename dist_t, typename label_t>
class FlexibleIndex : public Index<dist_t, label_t> {
 public:
  using typename Index<dist_t, label_t>::PriorityQueue;
  using typename Index<dist_t, label_t>::node_id_t;
  using typename Index<dist_t, label_t>::dist_node_t;

  // Bring in needed methods
  using Index<dist_t, label_t>::initializeSearch;
  using Index<dist_t, label_t>::allocateNode;
  using Index<dist_t, label_t>::getNodeLinks;
  using Index<dist_t, label_t>::getNodeData;
  using Index<dist_t, label_t>::defaultSelectNeighbors;
  using Index<dist_t, label_t>::beamSearch;

  // Bring in data members
  using Index<dist_t, label_t>::_cur_num_nodes;
  using Index<dist_t, label_t>::_max_node_count;
  using Index<dist_t, label_t>::_M;
  using Index<dist_t, label_t>::_distance;
  using Index<dist_t, label_t>::_index_data_guard;
  using Index<dist_t, label_t>::_node_links_mutexes;
  using Index<dist_t, label_t>::_num_threads;

  using PruningFunction = std::function<void(PriorityQueue&, int)>;
  PruningFunction _custom_pruning_function;

 public:
  FlexibleIndex(std::unique_ptr<DistanceInterface<dist_t>> distance, size_t dataset_size, size_t max_edges,
                bool collect_stats = false, DataType data_type = DataType::float32)
      : Index<dist_t, label_t>(std::move(distance), dataset_size, max_edges, collect_stats, data_type) {}

  void setPruningFunction(PruningFunction func) { this->_custom_pruning_function = func; }

  void selectNeighbors(PriorityQueue& neighbors, int M) {
    if (_custom_pruning_function) {
      std::cout << "Using custom pruning function" << "\n" << std::flush;
      _custom_pruning_function(neighbors, M);
    } else {
      std::cout << "Using default pruning function" << "\n" << std::flush;
      this->defaultSelectNeighbors(neighbors, M);
    }
  }

  template <typename data_type>
  void addBatch(void* data, std::vector<label_t>& labels, int ef_construction,
                int num_initializations = 100) {
    if (num_initializations <= 0) {
      throw std::invalid_argument("num_initializations must be greater than 0.");
    }
    uint32_t total_num_nodes = labels.size();
    uint32_t data_dimension = this->_distance->dimension();

    // Don't spawn any threads if we are only using one.
    if (this->_num_threads == 1) {
      for (uint32_t row_index = 0; row_index < total_num_nodes; row_index++) {
        uint64_t offset = static_cast<uint64_t>(row_index) * static_cast<uint64_t>(data_dimension);
        void* vector = (data_type*)data + offset;
        label_t label = labels[row_index];
        this->add(vector, label, ef_construction, num_initializations);
      }
      return;
    }

    flatnav::executeInParallel(
        /* start_index = */ 0, /* end_index = */ total_num_nodes,
        /* num_threads = */ this->_num_threads, /* function = */
        [&](uint32_t row_index) {
          uint64_t offset = static_cast<uint64_t>(row_index) * static_cast<uint64_t>(data_dimension);
          void* vector = (data_type*)data + offset;
          label_t label = labels[row_index];
          this->add(vector, label, ef_construction, num_initializations);
        });
  }

  void add(void* data, label_t& label, int ef_construction, int num_initializations) {
    if (this->_cur_num_nodes >= this->_max_node_count) {
      throw std::runtime_error(
          "Maximum number of nodes reached. Consider "
          "increasing the `max_node_count` parameter to "
          "create a larger index.");
    }
    std::unique_lock<std::mutex> global_lock(this->_index_data_guard);
    auto entry_node = this->initializeSearch(data, num_initializations);
    node_id_t new_node_id;
    this->allocateNode(data, label, new_node_id);
    global_lock.unlock();

    if (new_node_id == 0) {
      return;
    }

    auto neighbors = this->beamSearch(
        /* query = */ data, /* entry_node = */ entry_node,
        /* buffer_size = */ ef_construction);

    int selection_M = std::max(static_cast<int>(_M / 2), 1);
    this->selectNeighbors(neighbors, selection_M);
    this->connectNeighbors(neighbors, new_node_id);
  }

  void connectNeighbors(PriorityQueue& neighbors, node_id_t new_node_id) {
    // connects neighbors according to the HSNW heuristic

    // Lock all operations on this node
    std::unique_lock<std::mutex> lock(this->_node_links_mutexes[new_node_id]);

    node_id_t* new_node_links = this->getNodeLinks(new_node_id);
    int i = 0;  // iterates through links for "new_node_id"

    while (neighbors.size() > 0) {
      node_id_t neighbor_node_id = neighbors.top().second;
      // add link to the current new node
      new_node_links[i] = neighbor_node_id;
      // now do the back-connections (a little tricky)

      std::unique_lock<std::mutex> neighbor_lock(this->_node_links_mutexes[neighbor_node_id]);
      node_id_t* neighbor_node_links = getNodeLinks(neighbor_node_id);
      bool is_inserted = false;
      for (size_t j = 0; j < _M; j++) {
        if (neighbor_node_links[j] == neighbor_node_id) {
          // If there is a self-loop, replace the self-loop with
          // the desired link.
          neighbor_node_links[j] = new_node_id;
          is_inserted = true;
          break;
        }
      }
      if (!is_inserted) {
        // now, we may to replace one of the links. This will disconnect
        // the old neighbor and create a directed edge, so we have to be
        // very careful. To ensure we respect the pruning heuristic, we
        // construct a candidate set including the old links AND our new
        // one, then prune this candidate set to get the new neighbors.

        float max_dist = this->_distance->distance(/* x = */ getNodeData(neighbor_node_id),
                                                   /* y = */ getNodeData(new_node_id));

        PriorityQueue candidates;
        candidates.emplace(max_dist, new_node_id);
        for (size_t j = 0; j < _M; j++) {
          if (neighbor_node_links[j] != neighbor_node_id) {
            auto label = neighbor_node_links[j];
            auto distance = this->_distance->distance(/* x = */ getNodeData(neighbor_node_id),
                                                      /* y = */ getNodeData(label));
            candidates.emplace(distance, label);
          }
        }
        // 2X larger than the previous call to defaultSelectNeighbors.
        selectNeighbors(candidates, this->_M);
        // connect the pruned set of candidates, including self-loops:
        size_t j = 0;
        while (candidates.size() > 0) {  // candidates
          neighbor_node_links[j] = candidates.top().second;
          candidates.pop();
          j++;
        }
        while (j < _M) {  // self-loops (unused links)
          neighbor_node_links[j] = neighbor_node_id;
          j++;
        }
      }

      // Unlock the current node we are iterating over
      neighbor_lock.unlock();

      // loop increments:
      i++;
      neighbors.pop();
    }
  }
};

}  // namespace flatnav