/*
 * globals.h
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

/*
 * defines all functions that provided from cudacc source files
 */


#ifndef __GLOBALS_H
#define __GLOBALS_H

//! \def MIP mode
#define TYPE_MIP 0

//! \def XRAY mode
#define TYPE_XRAY 1

//! \def Composited mode
#define TYPE_COMPOSITED 2

//! \typedef Should be one of TYPE_MIP TYPE_XRAY or TYPE_COMPOSITED
typedef unsigned char VolumeType;

//! cuda function to bind array to textures
extern "C" cudaError_t bindArray2Texture(cudaArray * d_volumeArray,cudaChannelFormatDesc* pchannelDesc,
							cudaArray * d_transferFuncArray,cudaChannelFormatDesc* pchannelDesc2);

//! cuda function to upload the view matrx for renderring
extern "C" cudaError_t copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);

//! cuda function to upload filter mode
extern "C" void setTextureFilterMode(bool bLinearFilter);

//! cuda function to invoke the kernel function
extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
                              float density, float brightness, float transferOffset, float transferScale, 
							  char renderringType, float minExclude, float maxExclude);

#endif
