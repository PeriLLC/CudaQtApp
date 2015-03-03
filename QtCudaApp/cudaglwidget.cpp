/*
 * cudaglwidget.cpp
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

#include "cudaglwidget.h"

#include <QString>
#include <QMessageBox>

//-----------------------------------------------------------------------------
// AppGLWidget
//-----------------------------------------------------------------------------
CUDAGLWidget::CUDAGLWidget(QWidget *parent)
	: QGLWidget(QGLFormat(QGL::SampleBuffers), parent), spbo(NULL),volumeReady(false)
{
	spbo=new CudaPBO;
	strFrames = "Frames per second: 0";
	frames    = 0;
	tiupdate  = new QTimer( this );
	tifps     = new QTimer( this );
	connect ( tiupdate, SIGNAL( timeout() ), this, SLOT ( update() ) );
	connect ( tifps,    SIGNAL( timeout() ), this, SLOT ( framesCount() ) );

	tifps->start(1000);
	timeElapse.start();
	glError = GL_NO_ERROR;

	mousedown=0;
	volumeReady=false;
}
//-----------------------------------------------------------------------------
// AppGLWidget
//-----------------------------------------------------------------------------
CUDAGLWidget::~CUDAGLWidget()
{
	delete tiupdate;
	delete tifps;
	if (spbo)
		delete spbo;
}

bool CUDAGLWidget::isCudaAvailable()
{
	if (spbo)
		return spbo->isCudaAvailable();
	else
		return false;
}

//-----------------------------------------------------------------------------
// initializeGL
//-----------------------------------------------------------------------------
void CUDAGLWidget::initializeGL()
{

	int vmaj = format().majorVersion();
	int vmin = format().minorVersion();
	if( vmaj < 2 ){
		QMessageBox::critical(this, 
			tr("Wrong OpenGL version"), 
			tr("OpenGL version 2.0 or higher needed. You have %1.%2, so some functions may not work properly.").arg(vmaj).arg(vmin));
		return;
	}

	if(spbo)
		spbo->initCuda();

	glClearColor(0, 0, 0, 1);
	glDisable(GL_DEPTH_TEST);
	//  glShadeModel(GL_FLAT);
	//  glDisable(GL_LIGHTING);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	startUpdateTimer();
}
//-----------------------------------------------------------------------------
// resizeGL
//-----------------------------------------------------------------------------
void CUDAGLWidget::resizeGL(int w, int h){

	int l=w;
	if (l>h) l=h;
	glViewport(0, 0, l, l);
	glEnable(GL_TEXTURE_2D);
	spbo->resize(l, l);

	glMatrixMode(GL_MODELVIEW); //Select The Modelview Matrix
	glLoadIdentity(); //Reset The Modelview Matrix
	glMatrixMode(GL_PROJECTION); //Select The Projection Matrix
	glLoadIdentity(); //Reset The Projection Matrix
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}
//-----------------------------------------------------------------------------
// paintGL
//-----------------------------------------------------------------------------
void CUDAGLWidget::paintGL(){

	GLfloat modelView[16];
	if(isCudaAvailable())
	{
		if(volumeReady)
		{

			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			glLoadIdentity();

			glRotatef(-spbo->renderSetting.viewRotation.x, 1.0, 0.0, 0.0);
			glRotatef(-spbo->renderSetting.viewRotation.y, 0.0, 1.0, 0.0);
			glTranslatef(-spbo->renderSetting.viewTranslation.x, -spbo->renderSetting.viewTranslation.y, -spbo->renderSetting.viewTranslation.z);
			glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
			glPopMatrix();

			spbo->renderSetting.invViewMatrix[0] = modelView[0];
			spbo->renderSetting.invViewMatrix[1] = modelView[4];
			spbo->renderSetting.invViewMatrix[2] = modelView[8];
			spbo->renderSetting.invViewMatrix[3] = modelView[12];
			spbo->renderSetting.invViewMatrix[4] = modelView[1];
			spbo->renderSetting.invViewMatrix[5] = modelView[5];
			spbo->renderSetting.invViewMatrix[6] = modelView[9];
			spbo->renderSetting.invViewMatrix[7] = modelView[13];
			spbo->renderSetting.invViewMatrix[8] = modelView[2];
			spbo->renderSetting.invViewMatrix[9] = modelView[6];
			spbo->renderSetting.invViewMatrix[10] = modelView[10];
			spbo->renderSetting.invViewMatrix[11] = modelView[14];

			// run CUDA kernel
			spbo->runCuda( timeElapse.elapsed() );

			glClearColor(0.,0.,.2,.5);
			glClear(GL_COLOR_BUFFER_BIT);// | GL_DEPTH_BUFFER_BIT);
			// draw image from PBO
			glDisable(GL_DEPTH_TEST);
			glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

			// now bind buffer after cuda is done
			spbo->bind();

			glColor3f(1,1,1);
			// draw textured quad
			glEnable(GL_TEXTURE_2D);
			glBegin(GL_QUADS);
			glTexCoord2f(0, 0);
			glVertex2f(0, 0);
			glTexCoord2f(1, 0);
			glVertex2f(1, 0);
			glTexCoord2f(1, 1);
			glVertex2f(1, 1);
			glTexCoord2f(0, 1);
			glVertex2f(0, 1);
			glEnd();

			spbo->release();

			glColor3f(1,1,0);
			renderText(20,20, strFrames);
		}
		else
		{
			glClearColor(0.,0.,.2,.5);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glColor3f(0,1,0);
			renderText(width()/2,height()/2,tr("No Volume Loaded!"));
		}
	}
	else
	{
		glClearColor(0.,0.,.2,.5);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glColor3f(1,0,0);
		renderText(width()/2,height()/2,tr("CUDA not available!"));
	}
	++frames;

	glError = glGetError();
}
//-----------------------------------------------------------------------------
// startUpdateTimer
//-----------------------------------------------------------------------------
void CUDAGLWidget::startUpdateTimer()
{
	if( tiupdate->isActive() == false ){
		timeElapse.restart();
		tiupdate->start(0);
	}
}
//-----------------------------------------------------------------------------
// stopUpdateTimer
//-----------------------------------------------------------------------------
void CUDAGLWidget::stopUpdateTimer()
{
	tiupdate->stop();
	timeElapse.restart();
}
//-----------------------------------------------------------------------------
// framesCount
//-----------------------------------------------------------------------------
void CUDAGLWidget::framesCount(){
	strFrames.setNum(frames);
	strFrames.prepend(tr("Frames per second: "));
	frames=0;
}
//-----------------------------------------------------------------------------
// update - SLOT Timer tiupdate
//-----------------------------------------------------------------------------
void CUDAGLWidget::update(){
	if( glError != GL_NO_ERROR ){ // OpenGL ocurred
		stopUpdateTimer();

		QMessageBox::warning(this, "OpenGL Error: "+QString::number(glError), getGLError());
		glError=GL_NO_ERROR;
	}else{
		updateGL();
	}
}

void CUDAGLWidget::mousePressEvent(QMouseEvent * event){
	if (event->button()==Qt::LeftButton)
	{
		ox = event->x();
		oy = event->y();
		mousedown=Qt::LeftButton;
	}
	else
		if (event->button()==Qt::RightButton)
		{
			ox = event->x();
			oy = event->y();
			mousedown=Qt::RightButton;
		}

}
void CUDAGLWidget::mouseReleaseEvent(QMouseEvent * event){
	if ((event->button()==Qt::LeftButton)||(event->button()==Qt::RightButton))
	{
		mousedown=0;
	}

}

void CUDAGLWidget::wheelEvent(QWheelEvent *event){
	QPoint numPixels = event->pixelDelta();
	QPoint numDegrees = event->angleDelta() / 8;


	if (!numPixels.isNull()) {
		//        scrollWithPixels(numpixels);
		spbo->renderSetting.viewTranslation.z += numPixels.ry() / 10.0f;
	} else if (!numDegrees.isNull()) {
		QPoint numSteps = numDegrees / 15;
		//        scrollWithDegrees(numSteps);
		spbo->renderSetting.viewTranslation.z += numSteps.ry() / 10.0f;
	}

	event->accept();	

}

void CUDAGLWidget::mouseMoveEvent(QMouseEvent * event){

	float dx, dy;
	dx = (float)(event->x() - ox);
	dy = (float)(event->y() - oy);

	if(mousedown==Qt::LeftButton)
	{
		// left = rotate
		spbo->renderSetting.viewRotation.x += dy / 5.0f;
		spbo->renderSetting.viewRotation.y += dx / 5.0f;
		//qDebug()<<tr("mm%1,%2\n").arg(dx).arg(dy);
	}else if (event->buttons()==Qt::RightButton)
	{
		// right = translate
		spbo->renderSetting.viewTranslation.x += dx / 100.0f;
		spbo->renderSetting.viewTranslation.y -= dy / 100.0f;
	}

	ox = event->x();
	oy = event->y();
}

//-----------------------------------------------------------------------------
// getGLError
//-----------------------------------------------------------------------------
QString CUDAGLWidget::getGLError(){

	QString exception = "No error";
	switch (glError)
	{
		// see macro on top
		GLERROR(GL_INVALID_ENUM)
			GLERROR(GL_INVALID_VALUE)
			GLERROR(GL_INVALID_OPERATION)
			GLERROR(GL_STACK_OVERFLOW)
			GLERROR(GL_STACK_UNDERFLOW)
			GLERROR(GL_OUT_OF_MEMORY)
#ifdef GL_INVALID_INDEX
			GLERROR(GL_INVALID_INDEX)
#endif

#ifdef GL_INVALID_FRAMEBUFFER_OPERATION_EXT
			GLERROR(GL_INVALID_FRAMEBUFFER_OPERATION_EXT)
#endif
	default:
		exception.sprintf("Unknown GL error: %04x\n", glError);
		break;
	}
	return exception;
}