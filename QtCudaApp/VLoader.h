/*
 * VLoader.h
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


#pragma once

#include <stdio.h>

#define DDS_ISINTEL (*((unsigned char *)(&DDS_INTEL)+1)==0)


/*! \class CVLoader
 *  \brief The class that loads and decode volume files.
 *
 *
 *  CVLoader class provides all helper functions as static. So you do not need to create instance.
 *  It supports three formats now: PVM, DDS and RAW.
 */

class CVLoader
{
public:

	
    //! Function that reads PVM format volume file.
    /*!
	 *  \param filename		The name with its path of the volume file.
	 *  \param width		A point to where the width of volume is reported.
	 *  \param height		A point to where the height of volume is reported.
	 *  \param depth		A point to where the depth of volume is reported.
	 *  \param components	A point to where the components of volume is reported. 
	 *  \return				A point to where the decoded volume is stored, it is your duty to free() 
	 *                      the memory after finishing use it, to avoid memory leak.
	 */
	static unsigned char *readPVMvolume(const char *filename,
		unsigned int *width,unsigned int *height,unsigned int *depth,unsigned int *components=NULL,
		float *scalex=NULL,float *scaley=NULL,float *scalez=NULL,
		unsigned char **description=NULL,
		unsigned char **courtesy=NULL,
		unsigned char **parameter=NULL,
		unsigned char **comment=NULL);
	
	
	
    //! Function that reads DDS format volume file.
    /*!
	 *  \param filename		The name with its path of the volume file.
	 *  \param bytes		A point to where the total bytes decoded.
	 *  \return				A point to where the decoded volume is stored, it is your duty to free() 
	 *                      the memory after finishing use it, to avoid memory leak.
	 */
	static unsigned char *readDDSfile(const char *filename,unsigned int *bytes);

	//! Function that reads RAW format volume file.
    /*!
	 *  \param filename		The name with its path of the volume file.
	 *  \param bytes		A point to where the total bytes decoded.
	 *  \return				A point to where the decoded volume is stored, it is your duty to free() 
	 *                      the memory after finishing use it, to avoid memory leak.
	 */	
	static unsigned char *readRAWfile(const char *filename,unsigned int *bytes);

	//! Function that reads RAW format volume file.
    /*!
	 *  \param fp			The point to file of the volume file. This function will not fclose() the file.
	 *  \param bytes		A point to where the total bytes decoded.
	 *  \return				A point to where the decoded volume is stored, it is your duty to free() 
	 *                      the memory after finishing use it, to avoid memory leak.
	 */		
	static unsigned char *readRAWfile(FILE *fp,unsigned int *bytes);

protected:
	static unsigned char* DDS_decode(unsigned char *chunk,unsigned int size,
		unsigned char **data,unsigned int *bytes,
		unsigned int block=0);
	
	static void DDS_initbuffer();

	static unsigned char* DDS_deinterleave(unsigned char *data,unsigned int bytes,unsigned int skip,unsigned int block=0, bool restore=false);

	static unsigned char* DDS_loadbits(unsigned char *data,unsigned int size);

	static inline void DDS_clearbits()
	{
		DDS_cache=NULL;
		DDS_cachepos=0;
		DDS_cachesize=0;
	}

	static inline int DDS_decode(int bits)
	{return(bits>=1?bits+1:bits);}

	static inline void DDS_interleave(unsigned char *data,unsigned int bytes,unsigned int skip,unsigned int block=0)
	{DDS_deinterleave(data,bytes,skip,block,true);}

	static inline unsigned int DDS_shiftl(const unsigned int value,const unsigned int bits)
	{return((bits>=32)?0:value<<bits);}

	static inline unsigned int DDS_shiftr(const unsigned int value,const unsigned int bits)
	{return((bits>=32)?0:value>>bits);}

	static inline unsigned int DDS_readbits(unsigned int bits)
	{
		unsigned int value;

		if (bits<DDS_bufsize)
		{
			DDS_bufsize-=bits;
			value=DDS_shiftr(DDS_buffer,DDS_bufsize);
		}
		else
		{
			value=DDS_shiftl(DDS_buffer,bits-DDS_bufsize);

			if (DDS_cachepos>=DDS_cachesize) DDS_buffer=0;
			else
			{
				DDS_buffer=*((unsigned int *)&DDS_cache[DDS_cachepos]);
				if (DDS_ISINTEL) DDS_swapuint(&DDS_buffer);
				DDS_cachepos+=4;
			}

			DDS_bufsize+=32-bits;
			value|=DDS_shiftr(DDS_buffer,DDS_bufsize);
		}

		DDS_buffer&=DDS_shiftl(1,DDS_bufsize)-1;

		return(value);
	}

	static inline void DDS_swapuint(unsigned int *x)
	{
		unsigned int tmp=*x;

		*x=((tmp&0xff)<<24)|
			((tmp&0xff00)<<8)|
			((tmp&0xff0000)>>8)|
			((tmp&0xff000000)>>24);
	}

private:
	static unsigned char *DDS_cache;
	static unsigned int DDS_cachepos,DDS_cachesize;

	static unsigned int DDS_buffer;
	static unsigned int DDS_bufsize;

	static unsigned short int DDS_INTEL;


};

