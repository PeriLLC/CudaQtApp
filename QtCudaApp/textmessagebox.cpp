/*
 * textmessagebox.cpp
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

#include "textmessagebox.h"
#include <QVBoxLayout>

TextMessageBox::TextMessageBox(QWidget *parent)
	: QDialog(parent)
{
	textEdit=new QPlainTextEdit();
	textEdit->setReadOnly(true);
	buttonBox = new QDialogButtonBox(Qt::Horizontal);
	buttonBox->addButton(QDialogButtonBox::Close);

	connect(buttonBox, SIGNAL(rejected()),this, SLOT(close()));

	QGridLayout *mainLayout = new QGridLayout;
//    mainLayout->setSizeConstraint(QLayout::SetFixedSize);

    mainLayout->addWidget(textEdit, 0, 0);
    mainLayout->addWidget(buttonBox, 1, 0);
    mainLayout->setRowStretch(0, 1);

    setLayout(mainLayout);
}

TextMessageBox::~TextMessageBox()
{

}

void TextMessageBox::setInfo(QString title, QString msg)
{
	setWindowTitle(title);
	textEdit->setPlainText(msg);
}