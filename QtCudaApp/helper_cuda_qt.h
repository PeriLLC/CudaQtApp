/*
 * helper_cuda_qt.h
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

#ifndef HELPER_CUDA_QT_H
#define HELPER_CUDA_QT_H

#pragma once

#include <qmessagebox.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

/*! \brief This is helper functions for cuda error string, properties string decoder, with Qt supports.
 *
 *  
 */

/**
 * A const char* for all the error code, retrived from helper_qt.h
 */
static const char *_cudaGetErrorEnum(cudaError_t error)
{
	switch (error)
	{
	case cudaSuccess:
		return "cudaSuccess";

	case cudaErrorMissingConfiguration:
		return "cudaErrorMissingConfiguration";

	case cudaErrorMemoryAllocation:
		return "cudaErrorMemoryAllocation";

	case cudaErrorInitializationError:
		return "cudaErrorInitializationError";

	case cudaErrorLaunchFailure:
		return "cudaErrorLaunchFailure";

	case cudaErrorPriorLaunchFailure:
		return "cudaErrorPriorLaunchFailure";

	case cudaErrorLaunchTimeout:
		return "cudaErrorLaunchTimeout";

	case cudaErrorLaunchOutOfResources:
		return "cudaErrorLaunchOutOfResources";

	case cudaErrorInvalidDeviceFunction:
		return "cudaErrorInvalidDeviceFunction";

	case cudaErrorInvalidConfiguration:
		return "cudaErrorInvalidConfiguration";

	case cudaErrorInvalidDevice:
		return "cudaErrorInvalidDevice";

	case cudaErrorInvalidValue:
		return "cudaErrorInvalidValue";

	case cudaErrorInvalidPitchValue:
		return "cudaErrorInvalidPitchValue";

	case cudaErrorInvalidSymbol:
		return "cudaErrorInvalidSymbol";

	case cudaErrorMapBufferObjectFailed:
		return "cudaErrorMapBufferObjectFailed";

	case cudaErrorUnmapBufferObjectFailed:
		return "cudaErrorUnmapBufferObjectFailed";

	case cudaErrorInvalidHostPointer:
		return "cudaErrorInvalidHostPointer";

	case cudaErrorInvalidDevicePointer:
		return "cudaErrorInvalidDevicePointer";

	case cudaErrorInvalidTexture:
		return "cudaErrorInvalidTexture";

	case cudaErrorInvalidTextureBinding:
		return "cudaErrorInvalidTextureBinding";

	case cudaErrorInvalidChannelDescriptor:
		return "cudaErrorInvalidChannelDescriptor";

	case cudaErrorInvalidMemcpyDirection:
		return "cudaErrorInvalidMemcpyDirection";

	case cudaErrorAddressOfConstant:
		return "cudaErrorAddressOfConstant";

	case cudaErrorTextureFetchFailed:
		return "cudaErrorTextureFetchFailed";

	case cudaErrorTextureNotBound:
		return "cudaErrorTextureNotBound";

	case cudaErrorSynchronizationError:
		return "cudaErrorSynchronizationError";

	case cudaErrorInvalidFilterSetting:
		return "cudaErrorInvalidFilterSetting";

	case cudaErrorInvalidNormSetting:
		return "cudaErrorInvalidNormSetting";

	case cudaErrorMixedDeviceExecution:
		return "cudaErrorMixedDeviceExecution";

	case cudaErrorCudartUnloading:
		return "cudaErrorCudartUnloading";

	case cudaErrorUnknown:
		return "cudaErrorUnknown";

	case cudaErrorNotYetImplemented:
		return "cudaErrorNotYetImplemented";

	case cudaErrorMemoryValueTooLarge:
		return "cudaErrorMemoryValueTooLarge";

	case cudaErrorInvalidResourceHandle:
		return "cudaErrorInvalidResourceHandle";

	case cudaErrorNotReady:
		return "cudaErrorNotReady";

	case cudaErrorInsufficientDriver:
		return "cudaErrorInsufficientDriver";

	case cudaErrorSetOnActiveProcess:
		return "cudaErrorSetOnActiveProcess";

	case cudaErrorInvalidSurface:
		return "cudaErrorInvalidSurface";

	case cudaErrorNoDevice:
		return "cudaErrorNoDevice";

	case cudaErrorECCUncorrectable:
		return "cudaErrorECCUncorrectable";

	case cudaErrorSharedObjectSymbolNotFound:
		return "cudaErrorSharedObjectSymbolNotFound";

	case cudaErrorSharedObjectInitFailed:
		return "cudaErrorSharedObjectInitFailed";

	case cudaErrorUnsupportedLimit:
		return "cudaErrorUnsupportedLimit";

	case cudaErrorDuplicateVariableName:
		return "cudaErrorDuplicateVariableName";

	case cudaErrorDuplicateTextureName:
		return "cudaErrorDuplicateTextureName";

	case cudaErrorDuplicateSurfaceName:
		return "cudaErrorDuplicateSurfaceName";

	case cudaErrorDevicesUnavailable:
		return "cudaErrorDevicesUnavailable";

	case cudaErrorInvalidKernelImage:
		return "cudaErrorInvalidKernelImage";

	case cudaErrorNoKernelImageForDevice:
		return "cudaErrorNoKernelImageForDevice";

	case cudaErrorIncompatibleDriverContext:
		return "cudaErrorIncompatibleDriverContext";

	case cudaErrorPeerAccessAlreadyEnabled:
		return "cudaErrorPeerAccessAlreadyEnabled";

	case cudaErrorPeerAccessNotEnabled:
		return "cudaErrorPeerAccessNotEnabled";

	case cudaErrorDeviceAlreadyInUse:
		return "cudaErrorDeviceAlreadyInUse";

	case cudaErrorProfilerDisabled:
		return "cudaErrorProfilerDisabled";

	case cudaErrorProfilerNotInitialized:
		return "cudaErrorProfilerNotInitialized";

	case cudaErrorProfilerAlreadyStarted:
		return "cudaErrorProfilerAlreadyStarted";

	case cudaErrorProfilerAlreadyStopped:
		return "cudaErrorProfilerAlreadyStopped";

		/* Since CUDA 4.0*/
	case cudaErrorAssert:
		return "cudaErrorAssert";

	case cudaErrorTooManyPeers:
		return "cudaErrorTooManyPeers";

	case cudaErrorHostMemoryAlreadyRegistered:
		return "cudaErrorHostMemoryAlreadyRegistered";

	case cudaErrorHostMemoryNotRegistered:
		return "cudaErrorHostMemoryNotRegistered";

		/* Since CUDA 5.0 */
	case cudaErrorOperatingSystem:
		return "cudaErrorOperatingSystem";

	case cudaErrorPeerAccessUnsupported:
		return "cudaErrorPeerAccessUnsupported";

	case cudaErrorLaunchMaxDepthExceeded:
		return "cudaErrorLaunchMaxDepthExceeded";

	case cudaErrorLaunchFileScopedTex:
		return "cudaErrorLaunchFileScopedTex";

	case cudaErrorLaunchFileScopedSurf:
		return "cudaErrorLaunchFileScopedSurf";

	case cudaErrorSyncDepthExceeded:
		return "cudaErrorSyncDepthExceeded";

	case cudaErrorLaunchPendingCountExceeded:
		return "cudaErrorLaunchPendingCountExceeded";

	case cudaErrorNotPermitted:
		return "cudaErrorNotPermitted";

	case cudaErrorNotSupported:
		return "cudaErrorNotSupported";

		/* Since CUDA 6.0 */
	case cudaErrorHardwareStackError:
		return "cudaErrorHardwareStackError";

	case cudaErrorIllegalInstruction:
		return "cudaErrorIllegalInstruction";

	case cudaErrorMisalignedAddress:
		return "cudaErrorMisalignedAddress";

	case cudaErrorInvalidAddressSpace:
		return "cudaErrorInvalidAddressSpace";

	case cudaErrorInvalidPc:
		return "cudaErrorInvalidPc";

	case cudaErrorIllegalAddress:
		return "cudaErrorIllegalAddress";

		/* Since CUDA 6.5*/
	case cudaErrorInvalidPtx:
		return "cudaErrorInvalidPtx";

	case cudaErrorInvalidGraphicsContext:
		return "cudaErrorInvalidGraphicsContext";

	case cudaErrorStartupFailure:
		return "cudaErrorStartupFailure";

	case cudaErrorApiFailureBase:
		return "cudaErrorApiFailureBase";
	}

	return "<unknown>";
}

/**
 * Function to do the check cuda function calling response. Will show Qt dialog of error msg and return boolean status.
 * Do not directly call this function, use macro checkCudaErrorsQT instead.
 */
template< typename T >
bool check(T result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		//        fdesc.append(QString(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
		//               file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
		QString msg="CUDA error at %1:%2 code=%3(%4) \"%5\" \n";
		QMessageBox::critical(NULL,"CUDA Error!",msg.arg(file).arg(line)
			.arg(static_cast<unsigned int>(result)).arg( _cudaGetErrorEnum(result)).arg(func));

		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		return false;
	}
	return true;
}

/**
 * Function to do the check cuda function last calling response. Will show Qt dialog of error msg and return boolean status.
 * Do not directly call this function, use macro getLastCudaErrorQT instead.
 */
inline bool __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err)
	{
		//        fdesc.append(QString(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
		//                file, line, errorMessage, (int)err, cudaGetErrorString(err));
		QString msg="%1(%2) : getLastCudaError() CUDA error : %3 : (%4) %5.\n";
		QMessageBox::critical(NULL,"CUDA Error!",msg.arg(file).arg(line).arg(errorMessage).arg((int)err).arg(cudaGetErrorString(err)));

		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		return false;
	}
	return true;
}

//! \fn This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrorsQT(val)           check ( (val), #val, __FILE__, __LINE__ )

//! \fn This will output the proper error string when calling cudaGetLastError
#define getLastCudaErrorQT(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

/*! \brief initial and read the cuda devices information.
 *
 * \param prop	A point to an array of device properties. This funcion will allocate memory for the array, 
 *				and it is the caller's responsibility to delete [] it.
 * \return 
 *		- -1	get device count error
 *		- -2	get device properties error
 *		- -3	set device error
 */
inline int gpuGLDeviceInit(cudaDeviceProp **prop)
{
	int deviceCount;
	if(!checkCudaErrorsQT(cudaGetDeviceCount(&deviceCount)))
		return -1;
	if (deviceCount>0)
	{
		cudaDeviceProp *devices=new cudaDeviceProp[deviceCount];
		for (int i=0;i<deviceCount;i++)
		{
			if(!checkCudaErrorsQT(cudaSetDevice(i)))
			{
				delete []devices;
				return -3;
			}
			if(!checkCudaErrorsQT(cudaGetDeviceProperties(&(devices[i]), i)))
			{
				delete []devices;
				return -2;
			}
		}
		*prop=devices;
	}
	return deviceCount;
}


//! Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
        { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
        { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
//    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}
// end of GPU Architecture definitions


/*! Will produce a human readable text of device properties.
 *	\param deviceProp	A point to the cudaDeviceProp structure for decode the information
 *	\return				QString type of readable text.
 */
inline QString cudaDeviceDescQt(cudaDeviceProp * deviceProp)
{
	QString desc=QString("\nDevice name: \"%1\"\n").arg(deviceProp->name);

	desc.append(QString("  CUDA Capability Major/Minor version number:    %1.%2\n").arg( deviceProp->major).arg(deviceProp->minor));

	desc.append(QString("  Total amount of global memory:                 %1 MBytes (%2 bytes)\n")
		.arg((float)deviceProp->totalGlobalMem/1048576.0f).arg((unsigned long long) deviceProp->totalGlobalMem));

	desc.append(QString("  (%1) Multiprocessors, (%2) CUDA Cores/MP:     %3 CUDA Cores\n")
		.arg(deviceProp->multiProcessorCount)
		.arg(_ConvertSMVer2Cores(deviceProp->major, deviceProp->minor))
		.arg(_ConvertSMVer2Cores(deviceProp->major, deviceProp->minor) * deviceProp->multiProcessorCount));
	desc.append(QString("  GPU Clock rate:                                %1 MHz (%2 GHz)\n")
		.arg(deviceProp->clockRate * 1e-3f).arg(deviceProp->clockRate * 1e-6f));


	desc.append(QString("  Memory Clock rate:                             %1 Mhz\n").arg(deviceProp->memoryClockRate * 1e-3f));
	desc.append(QString("  Memory Bus Width:                              %1-bit\n").arg(deviceProp->memoryBusWidth));
	
	if (deviceProp->l2CacheSize)
	{
		desc.append(QString("  L2 Cache Size:                                 %1 bytes\n").arg(deviceProp->l2CacheSize));
	}

	desc.append(QString("  Maximum Texture Dimension Size (x,y,z)         1D=(%1), 2D=(%2, %3), 3D=(%4, %5, %6)\n")
		.arg(deviceProp->maxTexture1D).arg(deviceProp->maxTexture2D[0]).arg(deviceProp->maxTexture2D[1])
		.arg(deviceProp->maxTexture3D[0]).arg(deviceProp->maxTexture3D[1]).arg(deviceProp->maxTexture3D[2]));
	desc.append(QString("  Maximum Layered 1D Texture Size, (num) layers  1D=(%1), %2 layers\n")
		.arg(deviceProp->maxTexture1DLayered[0]).arg(deviceProp->maxTexture1DLayered[1]));
	desc.append(QString("  Maximum Layered 2D Texture Size, (num) layers  2D=(%1, %2), %3 layers\n")
		.arg(deviceProp->maxTexture2DLayered[0]).arg(deviceProp->maxTexture2DLayered[1]).arg(deviceProp->maxTexture2DLayered[2]));


	desc.append(QString("  Total amount of constant memory:               %1 bytes\n").arg( deviceProp->totalConstMem));
	desc.append(QString("  Total amount of shared memory per block:       %1 bytes\n").arg( deviceProp->sharedMemPerBlock));
	desc.append(QString("  Total number of registers available per block: %1\n").arg( deviceProp->regsPerBlock));
	desc.append(QString("  Warp size:                                     %1\n").arg( deviceProp->warpSize));
	desc.append(QString("  Maximum number of threads per multiprocessor:  %1\n").arg( deviceProp->maxThreadsPerMultiProcessor));
	desc.append(QString("  Maximum number of threads per block:           %1\n").arg( deviceProp->maxThreadsPerBlock));
	desc.append(QString("  Max dimension size of a thread block (x,y,z): (%1, %2, %3)\n")
		.arg(deviceProp->maxThreadsDim[0]).arg(deviceProp->maxThreadsDim[1]).arg(deviceProp->maxThreadsDim[2]));

	desc.append(QString("  Max dimension size of a grid size    (x,y,z): (%1, %2, %3)\n")
		.arg(deviceProp->maxGridSize[0]).arg(deviceProp->maxGridSize[1]).arg(deviceProp->maxGridSize[2]));

	desc.append(QString("  Maximum memory pitch:                          %1 bytes\n").arg( deviceProp->memPitch));
	desc.append(QString("  Texture alignment:                             %1 bytes\n").arg( deviceProp->textureAlignment));
	desc.append(QString("  Concurrent copy and kernel execution:          %1 with %2 copy engine(s)\n")
		.arg( (deviceProp->deviceOverlap ? "Yes" : "No")).arg(deviceProp->asyncEngineCount));
	desc.append(QString("  Run time limit on kernels:                     %1\n").arg( deviceProp->kernelExecTimeoutEnabled ? "Yes" : "No"));
	desc.append(QString("  Integrated GPU sharing Host Memory:            %1\n").arg( deviceProp->integrated ? "Yes" : "No"));
	desc.append(QString("  Support host page-locked memory mapping:       %1\n").arg( deviceProp->canMapHostMemory ? "Yes" : "No"));
	desc.append(QString("  Alignment requirement for Surfaces:            %1\n").arg( deviceProp->surfaceAlignment ? "Yes" : "No"));
	desc.append(QString("  Device has ECC support:                        %1\n").arg( deviceProp->ECCEnabled ? "Enabled" : "Disabled"));
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	desc.append(QString("  CUDA Device Driver Mode (TCC or WDDM):         %1\n").arg( deviceProp->tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)"));
#endif
	desc.append(QString("  Device supports Unified Addressing (UVA):      %1\n").arg( deviceProp->unifiedAddressing ? "Yes" : "No"));
	desc.append(QString("  Device PCI Bus ID / PCI location ID:           %1 / %2\n").arg( deviceProp->pciBusID).arg( deviceProp->pciDeviceID));

	const char *sComputeMode[] =
	{
		"Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
		"Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
		"Prohibited (no host thread can use ::cudaSetDevice() with this device)",
		"Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
		"Unknown",
		NULL
	};
	desc.append(QString("  Compute Mode:\n"));
	desc.append(QString("     < %1 >\n").arg( sComputeMode[deviceProp->computeMode]));
	
	return desc;
}
#endif