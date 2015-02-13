/*
 * RenderSetting.h
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

#include <qstring.h>
#include <cuda_runtime.h>

/*! \brief Settings for renderring the volume. Can be saved and read from the file.
 *
 */
class CRenderSetting
{
public:
	CRenderSetting(void);
	~CRenderSetting(void);

	QString settingIdentity;

	QString volumeFilename;
	QString remark;
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

	//! reset all setting to default
	void clear();
	
	//! print the setting in debug
	void printSetting();
};

//! << operator
QDataStream &operator<<(QDataStream &out, const CRenderSetting &setting);

//! >> operator
QDataStream &operator>>(QDataStream &in, CRenderSetting &setting);
