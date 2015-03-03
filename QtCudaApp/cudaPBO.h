/*
 * cudaPBO.h
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

#ifndef __CUDAPBO_H
#define __CUDAPBO_H

#include <qglbuffer.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include "RenderSetting.h"
/**
* \brief Encapsulates pixel buffer object and 2d texturing.
*
* It runs cuda and launches the kernel as well.
* Reference: http://drdobbs.com/cpp/222600097
* (took source and adapted it to this class)
*/
class CudaPBO
{
public:
	CudaPBO(void);
	~CudaPBO(void);

	/**
	 * Initial Cuda Device.
	 */	
	void initCuda();
	
	/**
	 * Resize PBO demension.
	 * @param w Window width
	 * @param h Window height
	 */	
	void resize(int w, int h);
	
	/**
	 * Bind to memory.
	 */	
	void bind();

	/**
	 * Unbind/release from memory.
	 */	
	void release();

	/**
	 * Call CUDA kernal function.
	 * @param time the elapsed time
	 */
	void runCuda(int time);
	
	/**
	 * Clean up Cuda Device.
	 */	
	void cleanupCuda();

	/**
	 * Return cuda availablity.
	 */
	bool isCudaAvailable()
	{
		return cudaAvailable;
	}

	/**
	 * Return a list, showing whether such device is available.
	 * The index is device id. 
	 * The device is not available maybe due to:
	 *	- sm is 1.0. Cuda 6.5 put sm <2.0 deprecated and sm 1.0 is totally out of support (cannot compile).
	 *  - device report not usable, maybe malfunction requires reset or occpyed by other processes.
	 */
	bool * getDeviceAvailable(){
		return isDeviceAvailable;
	}

	/**
	 * Return how many cuda device in the machine.
	 */
	int getDeviceCounts()
	{
		return deviceCounts;
	}
	
	/**
	 * Return all cuda device's properties, in an array.
	 */
	cudaDeviceProp *getDevicesProps()
	{
		return devicesProps;
	}
	
	/**
	 * Returun Cuda driver version.
	 */	
	int getDriverVersion()
	{
		return driverVersion;
	}

	/**
	 * Returun Cuda runtime version.
	 */	
	int getRuntimeVersion()
	{
		return runtimeVersion;
	}

	/**
	 * Select the device to run cuda kernel.
	 */	
	int selectDevice(int deviceID);
	
	/**
	 * Return which device is running cuda kernel.
	 */	
	int getSelectedDevice();

	/**
	 * Load volume into cuda device texture buffer.
	 */	
	bool loadCudaBuffers(void *h_volume, cudaExtent volumeSize);
	
	/**
	 * Clean cuda device texture buffer.
	 */		
	bool freeCudaBuffers();

	/**
	 *  This function find the fastest and available device.
	 *  return device id. If no available, return -1.
	 */
	int findFastestDevice();
private:
	void createPBO();
	void deletePBO();
	void createTexture();
	void deleteTexture();

	int iDivUp(int a, int b);

private:
	bool cudaAvailable;
	int deviceCounts;
	int selectedDevice;
	cudaDeviceProp *devicesProps;
	bool *isDeviceAvailable;
	int driverVersion;
	int runtimeVersion;
	unsigned int image_width;
	unsigned int image_height;
	QGLBuffer*   pixelBuffer;
	GLuint*      textureID;
	struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)
	dim3 blockSize;//(16, 16);
	dim3 gridSize;

private: // from cu files
	cudaArray *d_volumeArray;
	cudaArray *d_transferFuncArray;


public:
	CRenderSetting renderSetting;
	/*
	float invViewMatrix[12]; //!< The view matrix for cuda kernel to render
	float3 viewRotation;//!< rotation of view point
	float3 viewTranslation;//!< translation of view point
	char renderringType;//!< Renderring type setting
	float density;//!< Density setting
	float brightness;//!< Brightness setting
	float minExclude;//!< minExclude setting
	float maxExclude;//!< maxExclude setting
	float transferOffset;//!< transfer function Offset
	float transferScale;//!< transfer function scale
	*/
};

#endif
