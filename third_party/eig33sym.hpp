//
// Copyright 2014 Mitsubishi Electric Research Laboratories All
// Rights Reserved.
//
// Permission to use, copy and modify this software and its
// documentation without fee for educational, research and non-profit
// purposes, is hereby granted, provided that the above copyright
// notice, this paragraph, and the following three paragraphs appear
// in all copies.
//
// To request permission to incorporate this software into commercial
// products contact: Director; Mitsubishi Electric Research
// Laboratories (MERL); 201 Broadway; Cambridge, MA 02139.
//
// IN NO EVENT SHALL MERL BE LIABLE TO ANY PARTY FOR DIRECT,
// INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
// LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
// DOCUMENTATION, EVEN IF MERL HAS BEEN ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGES.
//
// MERL SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN
// "AS IS" BASIS, AND MERL HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE,
// SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
//
#pragma once
//
#include "dsyevh3.h"
#include <cmath>
#include <limits>

namespace LA {
    //s[0]<=s[1]<=s[2], V[:][i] correspond to s[i]
    inline static bool eig33sym(double K[3][3], double s[3], double V[3])
    {
        double tmpV[3][3];
        if(dsyevh3(K, tmpV, s) != 0) return false;

        int order[] = {0,1,2};
        for(int i=0; i < 3; ++i) {
            for(int j = i+1; j < 3; ++j) {
                if(s[i] > s[j]) {
                    double tmp=s[i];
                    s[i] = s[j];
                    s[j] = tmp;
                    int tmpor=order[i];
                    order[i]=order[j];
                    order[j]=tmpor;
                }
            }
        }
        V[0] = tmpV[0][order[0]];
        V[1] = tmpV[1][order[0]];
        V[2] = tmpV[2][order[0]];
        return true;
    }
}//end of namespace LA

