/*
 * RenderSetting.cpp
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

#include "RenderSetting.h"
#include "globals.h"
#include "qdatastream.h"

CRenderSetting::CRenderSetting(void)
{
	settingIdentity="QtCudaAppSetting 1.00.00";
	clear();
}


CRenderSetting::~CRenderSetting(void)
{
}

void CRenderSetting::clear()
{
	density = 1.0f;// 0.05f;
	brightness = 1.0f;
	transferOffset = 0.0f;
	transferScale = 1.0f;
	for (int i=0;i<12;i++)
		invViewMatrix[i]=0;

	viewTranslation = make_float3(0.0, 0.0, -4.0f);
	viewRotation = make_float3(0.0, 0.0, 0.0);
	renderringType=TYPE_MIP;
	minExclude=0.0f;
	maxExclude=1.0f;
	volumeFilename="";
	remark="";
}

QDataStream &operator<<(QDataStream &out, const CRenderSetting &setting)
{
    out << setting.settingIdentity << setting.volumeFilename <<setting.remark;
	
	for (int i=0;i<12;i++)
		out <<setting.invViewMatrix[i];

	out << setting.viewRotation.x;
	out << setting.viewRotation.y;
	out << setting.viewRotation.z;

	out << setting.viewTranslation.x;
	out << setting.viewTranslation.y;
	out << setting.viewTranslation.z;


	out << qint8(setting.renderringType);
	out << setting.density;
	out << setting.brightness;
	out << setting.minExclude;
	out << setting.maxExclude;
	out << setting.transferOffset;
	out << setting.transferScale;

	return out;
}

QDataStream &operator>>(QDataStream &in, CRenderSetting &setting)
{
	CRenderSetting settingtemp;
    in >> settingtemp.settingIdentity >> settingtemp.volumeFilename >>settingtemp.remark;
	
	for (int i=0;i<12;i++)
		in >>settingtemp.invViewMatrix[i];

	in >> settingtemp.viewRotation.x;
	in >> settingtemp.viewRotation.y;
	in >> settingtemp.viewRotation.z;

	in >> settingtemp.viewTranslation.x;
	in >> settingtemp.viewTranslation.y;
	in >> settingtemp.viewTranslation.z;

	qint8 t;
	in >> t;
	settingtemp.renderringType = t;

	in >> settingtemp.density;
	in >> settingtemp.brightness;
	in >> settingtemp.minExclude;
	in >> settingtemp.maxExclude;
	in >> settingtemp.transferOffset;
	in >> settingtemp.transferScale;

	setting=settingtemp;
	return in;

}

void CRenderSetting::printSetting()
{
	QString settingInfo="Settings:\n";
	settingInfo.append(QString("%1\n%2\n%3\n").arg(settingIdentity).arg(volumeFilename).arg(remark));

	settingInfo.append("\ninvViewMatrix\n");
	for (int i=0;i<12;i++)
		settingInfo.append(QString("%1\n").arg(invViewMatrix[i]));

	settingInfo.append(QString("\nviewRotation%1,%2,%3\n").arg(viewRotation.x).arg(viewRotation.y).arg(viewRotation.z));
	settingInfo.append(QString("\nviewTranslation%1,%2,%3\n").arg(viewTranslation.x).arg(viewTranslation.y).arg(viewTranslation.z));

	settingInfo.append(QString("renderringType %1\n").arg((qint8)renderringType));

	settingInfo.append(QString("density %1\n").arg(density));
	settingInfo.append(QString("brightness %1\n").arg(brightness));
	settingInfo.append(QString("minExclude %1\n").arg(minExclude));
	settingInfo.append(QString("maxExclude %1\n").arg(maxExclude));
	settingInfo.append(QString("transferOffset %1\n").arg(transferOffset));
	settingInfo.append(QString("transferScale %1\n").arg(transferScale));

	qDebug(settingInfo.toLatin1());
}