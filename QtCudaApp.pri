

HEADERS += ./QtCudaApp/cudaPBO.h \
    ./QtCudaApp/globals.h \
    ./QtCudaApp/helper_cuda.h \
    ./QtCudaApp/helper_math.h \
    ./QtCudaApp/helper_string.h \
    ./QtCudaApp/RenderSetting.h \
    ./QtCudaApp/VLoader.h \
    ./QtCudaApp/helper_cuda_qt.h \
    ./QtCudaApp/qtcudaapp.h \
    ./QtCudaApp/cudaglwidget.h \
    ./QtCudaApp/textmessagebox.h
SOURCES += ./QtCudaApp/cudaglwidget.cpp \
    ./QtCudaApp/RenderSetting.cpp \
    ./QtCudaApp/VLoader.cpp \
    ./QtCudaApp/main.cpp \
    ./QtCudaApp/qtcudaapp.cpp \
    ./QtCudaApp/cudaPBO.cpp \
    ./QtCudaApp/textmessagebox.cpp
FORMS += ./QtCudaApp/qtcudaapp.ui
RESOURCES += ./QtCudaApp/qtcudaapp.qrc

#CUDA_SOURCES += ./QtCudaApp/volumeRender_kernel.cu
