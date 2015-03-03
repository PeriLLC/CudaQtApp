/*
 * qtcudaapp.h
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


#ifndef QTCUDAAPP_H
#define QTCUDAAPP_H

#include <QtWidgets/QMainWindow>
#include "cudaglwidget.h"
#include "ui_qtcudaapp.h"

/*! \class QtCudaApp
 *  \brief The main window.
 *
 *
 *  QtCudaApp is the main window. The menu, toolbar and dockable control tool panel are all
 *  designed within qtcudaapp.ui . The main widget for showing the GL rendering result
 *  is created in this class. All action from the user interface is responded in this class.
 */
class QtCudaApp : public QMainWindow
{
	Q_OBJECT

public:
	QtCudaApp(QWidget *parent = 0);
	~QtCudaApp();


protected:
	void closeEvent(QCloseEvent *event);

protected slots:
	//! Function that responds to New File action
	void newFile();
	
	//! Function that responds to Open File action
	void open();
	
	//! Function that responds to Save File action
	bool save();
	
	//! Function that responds to Save File As action
	bool saveAs();
	
	//! Function that responds to Help->About
	void about();
	
	//! Slots Function that receive notification of changed on document.
	void documentWasModified();
	
	//! Function that shows cuda device information
	void cudaInfo();
	
	//! Function that check Cuda status after application starts, to enable rendering
	void checkCuda();
	
	//! Function that responds to Switch among available Cuda devices
	void switchDevice();
	
	//! Function that responds to Import Volume action
	void importVolume();

	//! Function that responds to Toggle Redner Type action
	void toggleRenderType();
	
	//! Function that responds to Reset View Point action
	void resetVP();
	
	//! Function that responds to Reset Renderring Setting action
	void resetRS();

	//! Function that responds to changing on brightness slide bar
	void bMoved(int);

	//! Function that responds to changing on density slide bar
	void dMoved(int);

	//! Function that responds to changing on maximal exclude slide bar
	void maxMoved(int);

	//! Function that responds to changing on minimal exclude slide bar
	void minMoved(int);

	//! Function that responds to changing on transfunction offset slide bar
	void toMoved(int);

	//! Function that responds to changing on transfunction scale slide bar
	void tsMoved(int);

protected:
	
	QString curFile;//!< The current file
	QString volumeFile; //!< The last loaded Volume File name
	
	//! Function that loads volume file into Cuda device
	bool loadVolume(QString fileName);
	
	//! whether the setting is modified
	bool isModified;

	//! Read the RenderSetting and load them in the dockable control panel
	void updateRenderSettingToControls();

	/*! Map slider value to render value linearly, now it is 0..99 -> 0..0.9
	 *  Linear mapping y=x/100;
	 */
	float slider2renderLinear(float v);

	/*! Map slider value to render value non-linearly, not it is 0..50..99 -> 0.01..1..100
	 *  It Is an exponential function. y=10^((x-50)/25)
	 */
	float slider2renderNonlinear(float v);

	//! Map render value to slider value linearly, reversly mapping
	float render2sliderLinear(float v);

	//! Map render value to slider value non-linearly, reversly mapping
	float render2sliderNonlinear(float v);

private:
	Ui::QtCudaAppClass ui;
	CUDAGLWidget *pCUDAGLWidget;

	QAction **deviceActions;

	QActionGroup* deviceActionGrp;

	void createActions();
	void readSettings();
	void writeSettings();
	bool maybeSave();
	void loadFile(const QString &fileName);
	bool saveFile(const QString &fileName);
	void setCurrentFile(const QString &fileName);
	QString strippedName(const QString &fullFileName);

};

#endif // QTCUDAAPP_H
