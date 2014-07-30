#ifndef _CONFIG_H_
#define _CONFIG_H_

#define DLL_EXPORT
#ifdef DLL_EXPORT
#define SVM_HOG_EXPORT __declspec( dllexport )
#else
#define SVM_HOG_EXPORT
#endif //DLL_EXPORT
#endif //_CONFIG_H_