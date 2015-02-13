/*
 * cudaglwidget.h
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


#ifndef __CUDAGLWIDGET_H
#define __CUDAGLWIDGET_H

// first 
#include "cudaPBO.h"

#include <QtGui>
#include <QtOpenGL>
#include <QGLWidget>
#include <QTimer>

// for AppGLWidget::getGLError()
#ifndef GLERROR
#define GLERROR(e) case e: exception=#e; break;
#endif

/**
* \brief OpenGL frame widget, designed for Cuda Renderring.
* Shows opengl scene being updated by frame timer.
*/
class CUDAGLWidget : public QGLWidget
{
	Q_OBJECT

public:
	/**
	* Constructor.
	* @param parent Parent widget.
	*/
	CUDAGLWidget(QWidget *parent);

	/**
	* Destructor (delete timers on heap).
	*/
	~CUDAGLWidget();

	/**
	 * Report if the cuda is available 
	 */
	bool isCudaAvailable();
	
	
	/**
	 * Report if the volume is ready to render 
	 */
	bool isVolumeReady()
	{
		return volumeReady;
	}
	
	/**
	 * set if the volue is ready to render 
	 */
	void setVolumeReady(bool vReady)
	{
		volumeReady=vReady;
	}
	
	/**
	* Starts Update Timer.
	* Timer is running at max speed 
	albeit frame rate may be capped by drivers (~60fps).
	*/
	void startUpdateTimer();

	/// Stop the update timer.
	void stopUpdateTimer();

protected:
	virtual void mouseMoveEvent(QMouseEvent * event);
	virtual void mousePressEvent(QMouseEvent * event);
	virtual void mouseReleaseEvent(QMouseEvent * event);
	virtual void wheelEvent(QWheelEvent *event);

	//-----------------------------------------------------------------------------
	public slots:
		/**
		* Update scene by calling updateGL(). 
		* If OpenGL Error ocurred, show it and do not update 
		(stopUpdateTimer).
		*/
		void update();

	private slots:
		/**
		* Count frames and create debug string (strFrames). Reset frame counter.
		*/
		void framesCount();

	//-----------------------------------------------------------------------------
private:
	/**
	* Initialize context, check for OpenGL driver and start Update Timer.
	*/
	void initializeGL();

	/**
	* Resize OpenGL frame.
	* @param w Window width
	* @param h Window height
	*/
	void resizeGL(int w, int h);

	/**
	* Paint OpenGL Frame.
	*/
	void paintGL();

	/**
	* @return OpenGL error enum as QString.
	*/
	QString getGLError();


public:
	/**
	 * @return Pixel Buffer Object that is related to this GL Widget
	 */
	CudaPBO *getCudaPBO()
	{
		return spbo;
	}

protected:
	/// Pixel Buffer Object
	CudaPBO *spbo;

	/// is the volume Ready
	bool volumeReady;
	/// frame counter
	unsigned int frames;
	/// error opengl enum type
	GLenum  glError;
	/// time object for measuring elapsed times
	QTime   timeElapse;
	/// timer for updating frame (\see update())
	QTimer* tiupdate; 
	/// 1sec timer for counting frames
	QTimer* tifps; 
	/// string holding frames debug text
	QString strFrames; 

private:
	int ox, oy;
	int mousedown;
};

#endif
