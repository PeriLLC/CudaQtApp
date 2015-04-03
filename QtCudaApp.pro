# ----------------------------------------------------
# Linux version ( Test on Ubuntu 10.04)
# ------------------------------------------------------

TEMPLATE = app
TARGET = QtCudaApp
DESTDIR = ./x64/Release
QT += core network opengl widgets gui
CONFIG += release
DEFINES += QT_DLL QT_NETWORK_LIB QT_OPENGL_LIB QT_WIDGETS_LIB
INCLUDEPATH += ./QtCudaApp/GeneratedFiles \
    ./QtCudaApp \
    ./QtCudaApp/GeneratedFiles/Release 

win32{
INCLUDEPATH +=  $CUDA_PATH/include
}

unix{	
INCLUDEPATH +=  ${CUDA_HOME}/include
}

DEPENDPATH += .

MOC_DIR += ./QtCudaApp/GeneratedFiles/release
OBJECTS_DIR += release
UI_DIR += ./QtCudaApp/GeneratedFiles
RCC_DIR += ./QtCudaApp/GeneratedFiles

###############################################
## Cuda Setting
###############################################
CUDA_SOURCES += ./QtCudaApp/volumeRender_kernel.cu

# Path to cuda toolkit install
unix{
CUDA_DIR      = /usr/local/cuda
QMAKE_LIBDIR += $$CUDA_DIR/lib64     # Note I'm using a 64 bits Operating system
}
win32{
CUDA_DIR      = $$(CUDA_PATH)
#QMAKE_LIBDIR += $$CUDA_DIR/lib/x64     # Note I'm using a 64 bits Operating system
LIBS += -L\"$$CUDA_DIR/lib/x64\"
}

# Path to header and libs files
INCLUDEPATH  += $$CUDA_DIR/include
# libs used in your code
LIBS += -lcudart -lcuda
# GPU architecture
CUDA_ARCH     = sm_20                # Yeah! I've a new device. Adjust with your compute capability

unix{
# Here are some NVCC flags I've always used by default.
NVCCFLAGS     = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v
}

win32{
# Here are some NVCC flags I've always used by default.
NVCCFLAGS     = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v -Xcompiler \"/EHsc /W1 /nologo /O2 /Zi  /MD  \"
}

# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

unix{ 
cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS \
                $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \

 
cuda.dependency_type = TYPE_C # there was a typo here. Thanks workmate!
cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}
}

win32{ 
cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS \
                -I\"$$CUDA_DIR/include\" $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
				> nul
 
cuda.dependency_type = TYPE_C # there was a typo here. Thanks workmate!
cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}
}


cuda.input = CUDA_SOURCES

unix{
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
}

win32{
cuda.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
}

# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda


include(./QtCudaApp.pri)
