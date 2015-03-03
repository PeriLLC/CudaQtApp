/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * volumeRender_kernel.cu
 * 
 * Copyright (c) 2015, Peri, LLC. All rights reserved.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301  USA
 */

//! Simple 3D volume renderer
//  Based on NVidia Cuda example, add three filters.

#ifndef _VOLUMERENDER_KERNEL_CU_
#define _VOLUMERENDER_KERNEL_CU_



//***************START Includes for IntelliSense**********************
#if !defined(__INTELLISENSE_ON) //only defined to have collapsable section.
#define __INTELLISENSE_ON
#define _SIZE_T_DEFINED
#define __THROW
#if !defined(__cplusplus)
#define __cplusplus
#endif
#include "float.h"
#include <cuda.h>
#include <device_launch_parameters.h>
#include <builtin_types.h>
#include <vector_functions.h>
#include <vector_types.h>
//#include "include/helper_print.h"  
#if !defined(__CUDA_INTERNAL_COMPILATION__)
#define __CUDA_INTERNAL_COMPILATION__
#endif
#include <math_functions.h>
#include <math.h>
#include "helper_math.h"  
#if !defined(__CUDACC__)
#define __CUDACC__
#endif
#include <texture_fetch_functions.h>
#include <device_functions.h>
#endif
//***************END Includes for IntelliSense**********************
#include "globals.h"

//typedef unsigned int  uint;
//typedef unsigned char uchar;


texture<VolumeType, 3, cudaReadModeNormalizedFloat> tex=0;         // 3D texture, first is VolumeType
texture<float4, 1, cudaReadModeElementType>         transferTex=0; // 1D transfer function texture

typedef struct
{
    float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

struct Ray
{
    float3 o;   // origin
    float3 d;   // direction
};

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

__global__ void
d_render(uint *d_output, uint imageW, uint imageH,
         float density, float brightness,
         float transferOffset, float transferScale, 
		 char renderringType, float minExclude, float maxExclude)
{
    const int maxSteps = 500;
    const float tstep = 0.01f;
    const float opacityThreshold = 0.95f;
    const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
    const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if ((x >= imageW) || (y >= imageH)) return;

    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // find intersection with box
    float tnear, tfar;
    int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

    if (!hit) return;

    if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0.0f);
    float t = tnear;
    float3 pos = eyeRay.o + eyeRay.d*tnear;
    float3 step = eyeRay.d*tstep;

    for (int i=0; i<maxSteps; i++)
    {
        // read from 3D texture
        // remap position to [0, 1] coordinates
        float sample = tex3D(tex, pos.x*0.5f+0.5f, pos.y*0.5f+0.5f, pos.z*0.5f+0.5f);
        //sample *= 64.0f;    // scale for 10-bit data
		
		if ((sample<minExclude)||(sample>maxExclude))
			continue;

		if (renderringType==TYPE_COMPOSITED){
			// mode transfer function
			// lookup in transfer function texture
			float4 col = tex1D(transferTex, (sample-transferOffset)*transferScale);
			col.w *= .05*density;

			// "under" operator for back-to-front blending
			//sum = lerp(sum, col, col.w);

			// pre-multiply alpha
			col.x *= col.w;
			col.y *= col.w;
			col.z *= col.w;
			// "over" operator for front-to-back blending
			sum = sum + col*(1.0f - sum.w);
		}
		else if(renderringType==TYPE_MIP){
			//MIP mode
			if (sum.w<sample*density)
				sum=make_float4(sample*density);
		}
		else
		{
			//X Ray mode
			sum = sum + make_float4(sample*tstep*density);
		}
        // exit early if opaque
        if (sum.w > opacityThreshold)
            break;

        t += tstep;

        if (t > tfar) break;

        pos += step;
    }

    sum *= brightness;

    // write output color
    d_output[y*imageW + x] = rgbaFloatToInt(sum);
}

extern "C"
void setTextureFilterMode(bool bLinearFilter)
{
    tex.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
}

extern "C"
cudaError_t bindArray2Texture(cudaArray * d_volumeArray,cudaChannelFormatDesc* pchannelDesc,
							cudaArray * d_transferFuncArray,cudaChannelFormatDesc* pchannelDesc2)
{
	cudaError_t retcode;
    // set texture parameters
    tex.normalized = true;                      // access with normalized texture coordinates
    tex.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
    tex.addressMode[1] = cudaAddressModeClamp;

    // bind array to 3D texture
    retcode=cudaBindTextureToArray(tex, d_volumeArray, *pchannelDesc);
	if(retcode!=cudaSuccess)
		return retcode;

    transferTex.filterMode = cudaFilterModeLinear;
    transferTex.normalized = true;    // access with normalized texture coordinates
    transferTex.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates

    // Bind the array to the texture
    retcode=cudaBindTextureToArray(transferTex, d_transferFuncArray, *pchannelDesc2);
	return cudaSuccess;
}



extern "C"
void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
                   float density, float brightness, float transferOffset, float transferScale, 
				   char renderringType, float minExclude, float maxExclude)
{
    d_render<<<gridSize, blockSize>>>(d_output, imageW, imageH, density,
                                      brightness, transferOffset, transferScale,
									  renderringType, minExclude, maxExclude);
}

extern "C"
cudaError_t copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
    return cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix);
}


#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
