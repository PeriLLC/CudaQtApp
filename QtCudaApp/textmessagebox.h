/*
 * TextMessageBox.h
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

#ifndef TEXTMESSAGEBOX_H
#define TEXTMESSAGEBOX_H

#include <QDialog>
#include <QPlainTextEdit>
#include <QDialogButtonBox>
#include <QPushButton>

/*! \class TextMessageBox
 *  \brief A dialog to show long text information.
 *
 *
 *  It is a resizable dialog, with a plain text editor which is read only showing the long message.
 *  It is useful for logger information or so. The CUDA device information is shown with it.
 *
 *  Usage: First create the dialog, then set the information with setInfo(), then exec() on the instance.
 */

class TextMessageBox : public QDialog
{
	Q_OBJECT

public:
	TextMessageBox(QWidget *parent=NULL);
	~TextMessageBox();

	//! Set the information with this function before exec() the dialog.
    /*!
	 *  \param title		The title informaion shows at the dialog.
	 *  \param msg			The message shown in the text editor.
	 */	
	void setInfo(QString title, QString msg);
	
private:
	QPlainTextEdit *textEdit;
	QDialogButtonBox *buttonBox;
	QWidget *extension;
};

#endif // TEXTMESSAGEBOX_H
