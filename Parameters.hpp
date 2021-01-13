#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <math.h>

//global detection parameters (main)
const float COS_ANGLE_MAX = cos(M_PI/10.0);
const float MAX_MERGE_DIST = 100;//50.0f;
const unsigned int PATCH_SIZE = 20;   //depth grid cell size


//Plane detection algo
const unsigned int MIN_SEED_COUNT = 6;  //min planes in batch to use it
const unsigned int MIN_CELL_ACTIVATED = 5;
//depth estimated std uncertainty in a depth segment (depends on depth)
const double DEPTH_SIGMA_COEFF = 1.425e-6; //2.76e-6;  // 
//depth discontinuity tolerance coefficient
const double DEPTH_DISCONTINUITY_LIMIT = 4;     //max number of discontinuities in cell before rejection
const double DEPTH_SIGMA_MARGIN = 12;           //[3, 8]
const double DEPTH_ALPHA = 0.06;                //[0.02, 0.04]


//cylinder segmentation 
const float CYLINDER_RANSAC_SQR_MAX_DIST = 0.04;//0.0225;    //square of 15%
const float CYLINDER_SCORE_MIN = 75;
//index offset of a cylinder to a plane: used for masks display purposes
const unsigned int CYLINDER_CODE_OFFSET = 50;

#endif
