#include "local_map.hpp"

namespace map_management {

    Local_Map::Local_Map()
    {

    }

    const matched_point_container Local_Map::find_matches(const keypoint_container& detectedPoints) 
    {


    }


    void Local_Map::update()
    {
        // add staged points to local map
        update_staged();

        // add local map points to global map
        update_local_to_global();
    }

    void Local_Map::reset()
    {
        _localMap.clear();
        _stagedPoints.clear();
    }

    void Local_Map::update_staged() 
    {
        // TODO when we have a global map
    }

}



