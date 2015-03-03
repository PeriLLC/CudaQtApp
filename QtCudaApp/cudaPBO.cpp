/*
* cudaPBO.cpp
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
#include "cudaPBO.h"

// CUDA Runtime, Interop, and includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "helper_cuda_qt.h"
#include "globals.h"
/*
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// CUDA utilities
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

// Helper functions
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>
#include <cuda_runtime_api.h>
*/

#pragma comment(lib, "cudart") 


#ifndef GL_BGRA
#define GL_BGRA 0x80E1
#endif

//-----------------------------------------------------------------------------
// Constructor
//-----------------------------------------------------------------------------
CudaPBO::CudaPBO(void)
	: pixelBuffer(0), textureID(0),cudaAvailable(false),
	deviceCounts(0),devicesProps(NULL),selectedDevice(-1),d_volumeArray(0),d_transferFuncArray(0),
	isDeviceAvailable(NULL)
{
	image_width  = 512;
	image_height = 512;
	blockSize.x=16;
	blockSize.y=16;
}
//-----------------------------------------------------------------------------
// Destructor
//-----------------------------------------------------------------------------
CudaPBO::~CudaPBO(void)
{
	cleanupCuda();
	if (devicesProps!=NULL)
		delete []devicesProps;
	if (isDeviceAvailable!=NULL)
		delete []isDeviceAvailable;
}
//-----------------------------------------------------------------------------
// initCuda
//-----------------------------------------------------------------------------
void CudaPBO::initCuda(){

	if(!checkCudaErrorsQT(cudaDriverGetVersion(&driverVersion)))
		return;
	if(!checkCudaErrorsQT(cudaRuntimeGetVersion(&runtimeVersion)))
		return;

	deviceCounts=gpuGLDeviceInit(&devicesProps);
	if(deviceCounts>0)
	{
		isDeviceAvailable=new bool[deviceCounts];
		for (int i=0;i<deviceCounts;i++)
		{
			if ((devicesProps[i].computeMode==cudaComputeModeProhibited )||((devicesProps[i].major<2)&&(devicesProps[i].minor<1)))
				isDeviceAvailable[i]=false;
			else
				isDeviceAvailable[i]=true;
		}
		int fastestdevice=findFastestDevice();
		if (fastestdevice>=0)
			if(checkCudaErrorsQT(cudaGLSetGLDevice(fastestdevice)))
			{
				selectedDevice=fastestdevice;
				cudaAvailable=true;
			}
	}
	//	qDebug("initCuda");
}
//-----------------------------------------------------------------------------
// resize
//-----------------------------------------------------------------------------
void CudaPBO::resize(int w, int h)
{
	// sizes must be a multiple of 16
	image_width = w+16-(w%16); 
	image_height = h+16-(h%16);

	// calculate new grid size
	gridSize = dim3(iDivUp(image_width, blockSize.x), iDivUp(image_height, blockSize.y));

	// delete pixelBuffer and textures if they already exist
	deletePBO();
	deleteTexture();
	// create pixel buffer object and register to cude
	createPBO();
	// create and allocate 2d texture buffer
	createTexture();
	// deactive pixelbuffer and texture object
	release();

	//	qDebug("resizePBO");

}

int CudaPBO::iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}


//-----------------------------------------------------------------------------
// runCuda -  Run the Cuda part of the computation
//-----------------------------------------------------------------------------
void CudaPBO::runCuda(int time)
{
	// uchar4 *dptr=NULL;

	Q_ASSERT(pixelBuffer);

	copyInvViewMatrix(renderSetting.invViewMatrix, sizeof(float4)*3);

	// map PBO to get CUDA device pointer
	uint *d_output;
	checkCudaErrorsQT(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));

	size_t num_bytes;
	checkCudaErrorsQT(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes, cuda_pbo_resource));
	//printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

	// clear image
	checkCudaErrorsQT(cudaMemset(d_output, 0, image_width*image_height*4));

	// call CUDA kernel, writing results to PBO
	render_kernel(gridSize, blockSize, d_output, image_width, image_height, renderSetting.density, 
		renderSetting.brightness, renderSetting.transferOffset, renderSetting.transferScale,
		renderSetting.renderringType, renderSetting.minExclude, renderSetting.maxExclude);

	getLastCudaErrorQT("kernel failed");

	checkCudaErrorsQT(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

	/*
	// map OpenGL buffer object for writing from CUDA 
	// on a single GPU no data is moved (Win & Linux). 
	// When mapped to CUDA, OpenGL should not use this buffer
	checkCudaErrors( cudaGLMapBufferObject((void**)&dptr, pixelBuffer->bufferId()) );
	Q_ASSERT(dptr);
	// execute the kernel
	launch_kernel(dptr, image_width, image_height, time);

	// unmap buffer object
	HANDLE_ERROR( cudaGLUnmapBufferObject(pixelBuffer->bufferId()) );
	*/
}
//-----------------------------------------------------------------------------
// createPBO
//-----------------------------------------------------------------------------
void CudaPBO::createPBO()
{

	// set up vertex data parameter
	int size_tex_data = sizeof(GLubyte) * image_width * image_height * 4;

	if(!pixelBuffer)
	{
		pixelBuffer = new QGLBuffer(QGLBuffer::PixelUnpackBuffer);

		/*
		FROM Qt Doc:
		The data will be modified repeatedly and used
		many times for reading data back from the GL server for
		use in further drawing operations.
		*/

		pixelBuffer->setUsagePattern(QGLBuffer::DynamicCopy);
		pixelBuffer->create();
	}
	pixelBuffer->bind();
	pixelBuffer->allocate(size_tex_data);

	checkCudaErrorsQT( cudaGraphicsGLRegisterBuffer( &cuda_pbo_resource, pixelBuffer->bufferId() ,cudaGraphicsMapFlagsWriteDiscard) );

}
//-----------------------------------------------------------------------------
// deletePBO
//-----------------------------------------------------------------------------
void CudaPBO::deletePBO()
{

	if (pixelBuffer) {
		// unregister this buffer object with CUDA
		checkCudaErrorsQT(cudaGraphicsUnregisterResource(cuda_pbo_resource));

		delete pixelBuffer;
		pixelBuffer = 0;
	}

}
//-----------------------------------------------------------------------------
// createTexture
//-----------------------------------------------------------------------------
void CudaPBO::createTexture()
{

	// delete texture object if necessary
	//  for reallocating tex mem, e.g. at different size
	deleteTexture();

	// Generate a texture identifier
	textureID = new GLuint[1]; // increase if u need more
	glGenTextures(1, textureID);

	// Make this the current texture (remember that GL is state-based)
	glBindTexture( GL_TEXTURE_2D, textureID[0]);

	// Allocate the texture memory. The last parameter is NULL since we only
	// want to allocate memory, not initialize it
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, image_width, image_height, 0,
		GL_BGRA,GL_UNSIGNED_BYTE, NULL);

	// Must set the filter mode, GL_LINEAR enables interpolation when scaling
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	// Note: GL_TEXTURE_RECTANGLE_ARB may be used instead of
	// GL_TEXTURE_2D for improved performance if linear interpolation is
	// not desired. Replace GL_LINEAR with GL_NEAREST in the
	// glTexParameteri() call

	// normally needed for deactivating but we will take release() later
	// glBindTexture(GL_TEXTURE_2D, 0);

}
//-----------------------------------------------------------------------------
// deleteTexture
//-----------------------------------------------------------------------------
void CudaPBO::deleteTexture()
{

	if(textureID){
		glDeleteTextures(1, textureID);
		delete[] textureID;
		textureID = 0;
	}

}
//-----------------------------------------------------------------------------
// bind
//-----------------------------------------------------------------------------
void CudaPBO::bind()
{

	Q_ASSERT(pixelBuffer);
	// Create a texture from the buffer
	pixelBuffer->bind();
	// bind texture from PBO
	glBindTexture(GL_TEXTURE_2D, textureID[0]);

	// Note: glTexSubImage2D will perform a format conversion if the
	// buffer is a different format from the texture. We created the
	// texture with format GL_RGBA8. In glTexSubImage2D we specified
	// GL_BGRA and GL_UNSIGNED_BYTE. This is a fast-path combination

	// Note: NULL indicates the data resides in device memory
	// hence data is coming from PBO
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, 
		GL_RGBA, GL_UNSIGNED_BYTE, NULL);

}
//-----------------------------------------------------------------------------
// release
//-----------------------------------------------------------------------------
void CudaPBO::release()
{

	Q_ASSERT(pixelBuffer);
	// deactivate pixelbuffer object
	pixelBuffer->release();
	// deactivate texture object
	glBindTexture(GL_TEXTURE_2D, 0);

}
//-----------------------------------------------------------------------------
// cleanupCuda
//-----------------------------------------------------------------------------
void CudaPBO::cleanupCuda()
{

	deletePBO();
	deleteTexture();

	freeCudaBuffers();

	checkCudaErrorsQT( cudaThreadExit() );
}

int CudaPBO::selectDevice(int deviceID){
	// delete pixelBuffer and textures if they already exist
	deletePBO();
	deleteTexture();

	freeCudaBuffers();

	if(!checkCudaErrorsQT(cudaDeviceReset()))
		return -2;
	if(checkCudaErrorsQT(cudaSetDevice(deviceID)))
	{
		//		int dd;
		//		cudaGetDevice (&dd);
		//		qDebug(QString("so device is%1").arg(dd).toLatin1());

		selectedDevice=deviceID;
		// create pixel buffer object and register to cude
		createPBO();
		// create and allocate 2d texture buffer
		createTexture();
		// deactive pixelbuffer and texture object
		release();

		qDebug(QString("select Device:%1\n").arg(deviceID).toLatin1());
		return deviceID;
	}
	return -1;
}

int CudaPBO::getSelectedDevice(){
	return selectedDevice;
}

bool CudaPBO::loadCudaBuffers(void *h_volume, cudaExtent volumeSize)
{
	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
	checkCudaErrorsQT(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr(h_volume, volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
	copyParams.dstArray = d_volumeArray;
	copyParams.extent   = volumeSize;
	copyParams.kind     = cudaMemcpyHostToDevice;
	checkCudaErrorsQT(cudaMemcpy3D(&copyParams));

	// create transfer function texture
	float4 transferFunc[] =
	{
		{  0.0, 0.0, 0.0, 0.0, },
		{  1.0, 0.0, 0.0, 1.0, },
		{  1.0, 0.5, 0.0, 1.0, },
		{  1.0, 1.0, 0.0, 1.0, },
		{  0.0, 1.0, 0.0, 1.0, },
		{  0.0, 1.0, 1.0, 1.0, },
		{  0.0, 0.0, 1.0, 1.0, },
		{  1.0, 0.0, 1.0, 1.0, },
		{  0.0, 0.0, 0.0, 0.0, },
	};

	cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
	cudaArray *d_transferFuncArray;
	checkCudaErrorsQT(cudaMallocArray(&d_transferFuncArray, &channelDesc2, sizeof(transferFunc)/sizeof(float4), 1));
	checkCudaErrorsQT(cudaMemcpyToArray(d_transferFuncArray, 0, 0, transferFunc, sizeof(transferFunc), cudaMemcpyHostToDevice));

	checkCudaErrorsQT(bindArray2Texture(d_volumeArray, &channelDesc,
		d_transferFuncArray, &channelDesc2));
	return true;
}

bool CudaPBO::freeCudaBuffers()
{
	if(d_volumeArray)
	{
		checkCudaErrorsQT(cudaFreeArray(d_volumeArray));
		d_volumeArray=0;
	}
	if(d_transferFuncArray)
	{
		checkCudaErrorsQT(cudaFreeArray(d_transferFuncArray));
		d_transferFuncArray=0;
	}
	return true;
}

int CudaPBO::findFastestDevice()
{
	int num_devices=getDeviceCounts();
	int max_multiprocessors = 0, max_device = -1;
	if (num_devices > 0) {
		for (int device = 0; device < num_devices; device++) {
			if (getDeviceAvailable()[device])
			{
				cudaDeviceProp properties=getDevicesProps()[device];
				if (max_multiprocessors < _ConvertSMVer2Cores(properties.major, properties.minor) * properties.multiProcessorCount) {
					max_multiprocessors = _ConvertSMVer2Cores(properties.major, properties.minor) * properties.multiProcessorCount;
					max_device = device;
				}
			}
		}
	}
	return max_device;
}