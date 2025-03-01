#pragma once 


#include <flatnav/index/Index.h>
#include <functional>



namespace flatnav {

template<typename dist_t, typename label_t>
class FlexibleIndex : public Index<dist_t, label_t> {
    using PruningFunction = std::function<void(PriorityQueue&, int)>;
    PruningFunction _custom_pruning_function;
public:
    void setPruningFunction(PruningFunction func) {
        _custom_pruning_function = func;
    }

    void selectNeighbors(PriorityQueue& neighbors, int M) {
        if (_custom_pruning_function) {
            _custom_pruning_function(neighbors, M);
        } else {
            defaultSelectNeighbors(neighbors, M);
        }
    }

};


}  // namespace flatnav