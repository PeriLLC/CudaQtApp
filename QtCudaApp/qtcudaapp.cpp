/*
* QtCudaApp.cpp
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

#include <QMessageBox>
#include <QCloseEvent>
#include <QFileDialog>
#include <QSettings>
#include <QTextStream>
#include <QTimer>
#include <QPlainTextEdit>

#include "qtcudaapp.h"
#include "textmessagebox.h"
#include "helper_cuda_qt.h"
#include "VLoader.h"
#include "globals.h"

QtCudaApp::QtCudaApp(QWidget *parent)
	: QMainWindow(parent),deviceActions(NULL),isModified(false)
{
	ui.setupUi(this);
	deviceActionGrp = new QActionGroup( this );
	pCUDAGLWidget = new CUDAGLWidget(this);
	setCentralWidget(pCUDAGLWidget);
	createActions();
	QTimer::singleShot(100, this, SLOT(checkCuda()));
}

QtCudaApp::~QtCudaApp()
{
	delete pCUDAGLWidget;
	if (deviceActions!=NULL)
		delete deviceActions;
}

void QtCudaApp::createActions()
{
	connect(ui.action_Device_Info,SIGNAL(triggered()),this, SLOT(cudaInfo()));
	connect(ui.action_About,SIGNAL(triggered()),this,SLOT(about()));
	connect(ui.action_Import_Volume,SIGNAL(triggered()),this,SLOT(importVolume()));

	connect(ui.action_New,SIGNAL(triggered()),this,SLOT(newFile()));
	connect(ui.action_Open,SIGNAL(triggered()),this,SLOT(open()));
	connect(ui.action_Save,SIGNAL(triggered()),this,SLOT(save()));
	connect(ui.action_Save_As,SIGNAL(triggered()),this,SLOT(saveAs()));

	connect(ui.radioButtonC,SIGNAL(toggled(bool)),this,SLOT(toggleRenderType()));
	connect(ui.radioButtonM,SIGNAL(toggled(bool)),this,SLOT(toggleRenderType()));
	connect(ui.radioButtonR,SIGNAL(toggled(bool)),this,SLOT(toggleRenderType()));
	connect(ui.btn_RVP,SIGNAL(clicked()), this, SLOT(resetVP()));
	connect(ui.btn_RRS,SIGNAL(clicked()), this, SLOT(resetRS()));
	connect(ui.sliderB,SIGNAL(sliderMoved(int)),this,SLOT(bMoved(int)));
	connect(ui.sliderD,SIGNAL(sliderMoved(int)),this,SLOT(dMoved(int)));
	connect(ui.sliderMax,SIGNAL(sliderMoved(int)),this,SLOT(maxMoved(int)));
	connect(ui.sliderMin,SIGNAL(sliderMoved(int)),this,SLOT(minMoved(int)));
	connect(ui.sliderTO,SIGNAL(sliderMoved(int)),this,SLOT(toMoved(int)));
	connect(ui.sliderTS,SIGNAL(sliderMoved(int)),this,SLOT(tsMoved(int)));

}

void QtCudaApp::checkCuda()
{
	if (pCUDAGLWidget->isCudaAvailable())
	{
		QMessageBox::about(this, tr("Cuda check"),tr("Cuda is ready!"));
		ui.action_New->setEnabled(true);
		ui.action_Open->setEnabled(true);
		ui.action_Save->setEnabled(true);
		ui.action_Save_As->setEnabled(true);
		ui.action_Import_Volume->setEnabled(true);

		ui.menu_Cuda->addSeparator();
		int deviceCount=pCUDAGLWidget->getCudaPBO()->getDeviceCounts();
		cudaDeviceProp * props = pCUDAGLWidget->getCudaPBO()->getDevicesProps();
		bool *isDeviceOk=pCUDAGLWidget->getCudaPBO()->getDeviceAvailable();
		deviceActions=new QAction*[deviceCount];
		for (int i=0;i<deviceCount;i++)
		{
			deviceActions[i]=ui.menu_Cuda->addAction(props[i].name);
			if(isDeviceOk[i])
			{
				deviceActions[i]->setCheckable(true);
				deviceActions[i]->setActionGroup(deviceActionGrp);
				connect(deviceActions[i],SIGNAL(triggered ()),this,SLOT(switchDevice()));
			}
			else
				deviceActions[i]->setEnabled(false);
		}
		int selectedDevice=pCUDAGLWidget->getCudaPBO()->getSelectedDevice();
		if ((selectedDevice>=0)&&(selectedDevice<deviceCount))
			deviceActions[selectedDevice]->setChecked(true);
	}
	else
		QMessageBox::about(this, tr("Cuda check failed"),tr("Cuda not available, please check hardware!\n Notice: CUDA 6.5 does not support sm1.0 anymore!"));
}

void QtCudaApp::switchDevice()
{
	int deviceCount=pCUDAGLWidget->getCudaPBO()->getDeviceCounts();
	int selectedDevice=pCUDAGLWidget->getCudaPBO()->getSelectedDevice();
	int wantToSel=-1;
	if (deviceActions[selectedDevice]->isChecked()){
		// it is what we are using.
		return;
	}

	bool wasVolumeReady=pCUDAGLWidget->isVolumeReady();
	pCUDAGLWidget->setVolumeReady(false);
	for (int i=0;i<deviceCount;i++)
	{
		if (deviceActions[i]->isChecked())
		{
			//We want to do this.
			wantToSel=i;
			break;
		}
	}

	if (pCUDAGLWidget->getCudaPBO()->selectDevice(wantToSel)<0)
	{
		QMessageBox::critical(this,tr("Cuda Fail"),tr("Swith device failed"));
		return;
	}

	if(wasVolumeReady)
		loadVolume(volumeFile);

	pCUDAGLWidget->setVolumeReady(wasVolumeReady);

}

void QtCudaApp::cudaInfo()
{
	if ((pCUDAGLWidget==NULL)||(pCUDAGLWidget->getCudaPBO()==NULL))
	{
		QMessageBox::critical(this,tr("Error"),tr("Qt widget failed!"));
		return;
	}
	QString cudainfo;
	TextMessageBox tmb;
	int dev=0;
	int deviceCount=pCUDAGLWidget->getCudaPBO()->getDeviceCounts();
	if (deviceCount == 0)
	{
		cudainfo.append("There are no available device(s) that support CUDA\n");
	}
	else
	{
		int driverVersion = pCUDAGLWidget->getCudaPBO()->getDriverVersion();
		int runtimeVersion = pCUDAGLWidget->getCudaPBO()->getRuntimeVersion();
		cudainfo.append(QString("CUDA Driver Version / Runtime Version          %1.%2 / %3.%4\n\n")
			.arg((int)(driverVersion/1000)).arg((int)((driverVersion%100)/10))
			.arg((int)(runtimeVersion/1000)).arg((int)((runtimeVersion%100)/10)));
		cudainfo.append(tr("Detected %1 CUDA Capable device(s)\n").arg(deviceCount));
		cudaDeviceProp * props = pCUDAGLWidget->getCudaPBO()->getDevicesProps();
		for (dev=0;dev<deviceCount;dev++){
			cudainfo.append(QString("\n=========================\nDevice: %1\n").arg(dev));
			cudainfo.append(cudaDeviceDescQt(&(props[dev]) ));
		}
	}

	tmb.setInfo("Cuda Device Info",cudainfo);
	tmb.exec();

}


void QtCudaApp::newFile()
{
	if (maybeSave()) {
		//		textEdit->clear();
		pCUDAGLWidget->getCudaPBO()->renderSetting.clear();
		pCUDAGLWidget->setVolumeReady(false);
		pCUDAGLWidget->getCudaPBO()->freeCudaBuffers();
		resetRS();
		this->volumeFile="";
		setCurrentFile("");
	}
}

void QtCudaApp::open()
{
	if (maybeSave()) {
		QString fileName = QFileDialog::getOpenFileName(this);
		if (!fileName.isEmpty())
			loadFile(fileName);
	}
}

bool QtCudaApp::save()
{
	if (curFile.isEmpty()) {
		return saveAs();
	} else {
		return saveFile(curFile);
	}
}

bool QtCudaApp::saveAs()
{
	QString fileName = QFileDialog::getSaveFileName(this);
	if (fileName.isEmpty())
		return false;

	return saveFile(fileName);
}

void QtCudaApp::about()
{
	QMessageBox::about(this, tr("About Application"),
		tr("The <b>QtCudaApp</b> is a framework that provide cuda "
		"a cross platform window widget supported development environment."));
	this->pCUDAGLWidget->getCudaPBO()->renderSetting.printSetting();
}

void QtCudaApp::documentWasModified()
{
	isModified=true;
}

void QtCudaApp::closeEvent(QCloseEvent *event)
{
	if (maybeSave()) {
		writeSettings();
		event->accept();
	} else {
		event->ignore();
	}
}

void QtCudaApp::readSettings()
{
	QSettings settings("Peri.LLC", "QtCudaApp");
	QPoint pos = settings.value("pos", QPoint(200, 200)).toPoint();
	QSize size = settings.value("size", QSize(400, 400)).toSize();
	resize(size);
	move(pos);
}

void QtCudaApp::writeSettings()
{
	QSettings settings("Peri.LLC", "QtCudaApp");
	settings.setValue("pos", pos());
	settings.setValue("size", size());
}

bool QtCudaApp::maybeSave()
{

	if (isModified) {
		QMessageBox::StandardButton ret;
		ret = QMessageBox::warning(this, tr("Application"),
			tr("The renderring setting document has been modified.\n"
			"Do you want to save your changes?"),
			QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);
		if (ret == QMessageBox::Save)
			return save();
		else if (ret == QMessageBox::Cancel)
			return false;
	}
	return true;
}

void QtCudaApp::loadFile(const QString &fileName)
{
	QFile file(fileName);
	if (!file.open(QFile::ReadOnly | QFile::Text)) {
		QMessageBox::warning(this, tr("Application"),
			tr("Cannot read file %1:\n%2.")
			.arg(fileName)
			.arg(file.errorString()));
		return;
	}

	QDataStream in(&file);
	in>>pCUDAGLWidget->getCudaPBO()->renderSetting;
	if((pCUDAGLWidget->getCudaPBO()->renderSetting.volumeFilename!=NULL)&&(pCUDAGLWidget->getCudaPBO()->renderSetting.volumeFilename!=""))
		loadVolume(pCUDAGLWidget->getCudaPBO()->renderSetting.volumeFilename);

	setCurrentFile(fileName);
	statusBar()->showMessage(tr("File loaded"), 2000);
}

bool QtCudaApp::saveFile(const QString &fileName)
{
	QFile file(fileName);
	if (!file.open(QFile::WriteOnly | QFile::Text)) {
		QMessageBox::warning(this, tr("Application"),
			tr("Cannot write file %1:\n%2.")
			.arg(fileName)
			.arg(file.errorString()));
		return false;
	}

	QDataStream out(&file);
	out<<pCUDAGLWidget->getCudaPBO()->renderSetting;

	setCurrentFile(fileName);
	statusBar()->showMessage(tr("File saved"), 2000);
	return true;
}

void QtCudaApp::setCurrentFile(const QString &fileName)
{
	curFile = fileName;
	
	isModified=false;
	setWindowModified(false);

	QString shownName = curFile;
	if (curFile.isEmpty())
		shownName = "untitled.txt";
	setWindowFilePath(shownName);
}

QString QtCudaApp::strippedName(const QString &fullFileName)
{
	return QFileInfo(fullFileName).fileName();
}

void QtCudaApp::importVolume()
{
	QString fileName = QFileDialog::getOpenFileName(this,
		tr("Open Volume File"), "./volume/", tr("Volume Files (*.pvm)"));

	if (!fileName.isNull())
	{
		if(loadVolume(fileName))
		{
			volumeFile=fileName;
			pCUDAGLWidget->getCudaPBO()->renderSetting.volumeFilename=fileName;
			resetRS();
		}
	}
	documentWasModified();

}

bool QtCudaApp::loadVolume(QString fileName)
{
	unsigned char *volume;

	unsigned int width,height,depth,components;

	float scalex,scaley,scalez;

	if ((volume=CVLoader::readPVMvolume(fileName.toLocal8Bit().data(),&width,&height,&depth,&components,&scalex,&scaley,&scalez))==NULL)
	{
		QMessageBox::critical(this,tr("Open PVM file"),tr("failed to open the file"));
		return false;
	}
	if (volume==NULL) 
	{
		QMessageBox::critical(this,tr("Open PVM file"),tr("failed to read the file"));
		return false;
	}
	qDebug()<<tr("found volume with width=%1 height=%2 depth=%3 components=%4\n").arg(width).arg(height).arg(depth).arg(components);

	if (scalex!=1.0f || scaley!=1.0f || scalez!=1.0f)
		qDebug()<<tr("and edge length %1/%2/%3\n").arg(scalex).arg(scaley).arg(scalez);

	//qDebug()<<tr("and data checksum=%1\n").arg(checksum(volume,width*height*depth*components));
	if (components!=1)
	{
		QMessageBox::information(this,tr("Open PVM file"),tr("Sorry, only 8 bits volume files are supported"));
		free(volume);
		return false;
	}
	else
	{
		cudaExtent volumeSize = make_cudaExtent(width, height, depth);

		// load volume data
		this->pCUDAGLWidget->setVolumeReady(false);
		//initCuda(volume,volumeSize);
		this->pCUDAGLWidget->getCudaPBO()->freeCudaBuffers();
		this->pCUDAGLWidget->getCudaPBO()->loadCudaBuffers(volume,volumeSize);
		this->pCUDAGLWidget->setVolumeReady(true);
		this->setWindowTitle(fileName);
	}

	free(volume);
	return true;
}

void QtCudaApp::toggleRenderType()
{
	char newRType=TYPE_MIP;;
	if (ui.radioButtonC->isChecked())
		newRType=TYPE_COMPOSITED;
	else if (ui.radioButtonR->isChecked())
		newRType=TYPE_XRAY;
	pCUDAGLWidget->getCudaPBO()->renderSetting.renderringType=newRType;
	//qDebug()<<tr("toggleRenderType:%1").arg((int)newRType);
	documentWasModified();
}

void QtCudaApp::bMoved(int v)
{
	pCUDAGLWidget->getCudaPBO()->renderSetting.brightness=slider2renderNonlinear(v);
	documentWasModified();
}
void QtCudaApp::dMoved(int v)
{
	pCUDAGLWidget->getCudaPBO()->renderSetting.density=slider2renderNonlinear(v);
	documentWasModified();
}
void QtCudaApp::maxMoved(int v)
{
	pCUDAGLWidget->getCudaPBO()->renderSetting.maxExclude=slider2renderLinear(v);
	documentWasModified();
}
void QtCudaApp::minMoved(int v)
{
	pCUDAGLWidget->getCudaPBO()->renderSetting.minExclude=slider2renderLinear(v);
	documentWasModified();
}

void QtCudaApp::resetVP()
{
	pCUDAGLWidget->getCudaPBO()->renderSetting.viewTranslation = make_float3(0.0, 0.0, -4.0f);
	pCUDAGLWidget->getCudaPBO()->renderSetting.viewRotation = make_float3(0.0, 0.0, 0.0);
}

void QtCudaApp::resetRS()
{
	pCUDAGLWidget->getCudaPBO()->renderSetting.clear();
	updateRenderSettingToControls();
}

void QtCudaApp::tsMoved(int v)
{
	pCUDAGLWidget->getCudaPBO()->renderSetting.transferScale=slider2renderNonlinear(v);
	documentWasModified();
}
void QtCudaApp::toMoved(int v)
{
	pCUDAGLWidget->getCudaPBO()->renderSetting.transferOffset=slider2renderLinear(v);
	documentWasModified();
}


float QtCudaApp::slider2renderLinear(float v)
{
	return v/100.0f;
}

float QtCudaApp::slider2renderNonlinear(float v)
{
	qreal x=(qreal)v;
	return qPow(10.0,((x-50.0f)/25.0f));
}

float QtCudaApp::render2sliderLinear(float v)
{
	return v*100.0f;
}

float QtCudaApp::render2sliderNonlinear(float v)
{
	qreal x=(qreal)v;
	return qLn(x)/qLn(2.618)*25.0f+50.0f;
}

void QtCudaApp::updateRenderSettingToControls()
{
	ui.radioButtonM->setChecked(pCUDAGLWidget->getCudaPBO()->renderSetting.renderringType==TYPE_MIP);
	ui.radioButtonR->setChecked(pCUDAGLWidget->getCudaPBO()->renderSetting.renderringType==TYPE_XRAY);
	ui.radioButtonC->setChecked(pCUDAGLWidget->getCudaPBO()->renderSetting.renderringType==TYPE_COMPOSITED);

	ui.sliderB->setValue(render2sliderNonlinear(pCUDAGLWidget->getCudaPBO()->renderSetting.brightness));
	ui.sliderD->setValue(render2sliderNonlinear(pCUDAGLWidget->getCudaPBO()->renderSetting.density));

	ui.sliderMax->setValue(render2sliderLinear(pCUDAGLWidget->getCudaPBO()->renderSetting.maxExclude));
	ui.sliderMin->setValue(render2sliderLinear(pCUDAGLWidget->getCudaPBO()->renderSetting.minExclude));

	ui.sliderTO->setValue(render2sliderLinear(pCUDAGLWidget->getCudaPBO()->renderSetting.transferOffset));
	ui.sliderTS->setValue(render2sliderNonlinear(pCUDAGLWidget->getCudaPBO()->renderSetting.transferScale));
}
